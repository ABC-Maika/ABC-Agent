"""The asyncio WebSocket server.

The third transport over the single ``World.step`` seam: hosts one
:class:`RetailParallelEnv` per game, drives ``env.step`` on a wall-clock
schedule, and accepts joint actions from N remote :class:`LiveClient`
connections. The seam is called exactly once per tick; no economics are
reimplemented here. One :func:`serve_game` call drives one game from
registration through final scores.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

import retail_simulator
from retail_simulator.core.config import CoreConfig
from retail_simulator.core.schema import ACTION_SCHEMA_VERSION
from retail_simulator.envs.parallel_env import RetailParallelEnv
from retail_simulator.live._permutation import permute_actions_by_seat
from retail_simulator.live.protocol import (
    PROTOCOL_VERSION,
    REGISTRY_DEFAULT_ACTION,
    ActionMessage,
    ErrorMessage,
    FinalScores,
    Hello,
    ObservationMessage,
    SeatAssigned,
    from_json_any,
)
from retail_simulator.live.replay import ActionLogRecord, ReplayManifest

if TYPE_CHECKING:
    # Type-only import; ``websockets`` is paid for only inside serve_game. The symbol moved
    # between websockets versions, so the parameter is typed Any at runtime and only the
    # public name is TYPE_CHECKING-imported (robust across the pinned >=12,<14 range).
    from websockets.server import WebSocketServerProtocol  # noqa: F401

_log = logging.getLogger("retail_simulator.live.server")

# ---------------------------------------------------------------------------- #
# Public dataclasses
# ---------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ServeConfig:
    """Configuration for one live game; built by the CLI and handed to :func:`serve_game`.

    All fields are required so the CLI surfaces every operator choice. ``token=None``
    is ``--no-auth``; ``tick_interval_seconds=0.0`` is CI fast mode;
    ``npc_archetypes=()`` is the all-remote mode.
    """

    n_agents: int
    n_ticks: int
    tick_interval_seconds: float
    host: str
    port: int
    registration_timeout_seconds: float
    token: str | None
    npc_archetypes: tuple[str, ...]
    scenario_name: str
    runs_dir: Path
    seed: int


@dataclass(frozen=True)
class ServeResult:
    """Outcome of one :func:`serve_game` call.

    ``exit_status`` mirrors a Unix exit code: ``0`` clean game, ``1`` registration
    timeout, ``2`` any other unrecoverable error.
    """

    game_id: str
    run_dir: Path
    rounds_completed: int
    exit_status: int
    summary: str
    final_scores: dict[str, float] = field(default_factory=dict)
    winner_seat: int | None = None


# ---------------------------------------------------------------------------- #
# Internal per-connection state
# ---------------------------------------------------------------------------- #


@dataclass
class _ClientState:
    """Per-connection mutable state held by the server during one game (keyed by seat_index).

    ``websocket`` is typed ``Any``: the websockets server protocol class is only
    imported under :data:`TYPE_CHECKING`.
    """

    seat_index: int
    websocket: Any  # WebSocketServerProtocol; ``Any`` keeps the runtime import lazy.
    client_name: str | None
    agent_name: str
    last_valid_action: npt.NDArray[np.float32] | None = None
    missed_deadlines: int = 0


# ---------------------------------------------------------------------------- #
# Helpers — action-log writer, JSON helpers, manifest persistence
# ---------------------------------------------------------------------------- #


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (sibling ``.tmp`` + ``os.replace``: a crash leaves either the prior or the new contents, never a partial file)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _flatten_action_mask(action_mask: Any) -> list[float] | None:
    """Flatten the seam's per-lever ``action_mask`` dict into the wire's flat list.

    Concatenates the per-key arrays in sorted-key order (deterministic layout).
    Returns ``None`` if the seam's mask is ``None``.
    """
    if action_mask is None:
        return None
    if not isinstance(action_mask, dict):
        # Defensive: a future seam revision might emit a flat array. Coerce.
        return [float(x) for x in np.asarray(action_mask).ravel()]
    flat: list[float] = []
    for key in sorted(action_mask.keys()):
        arr = np.asarray(action_mask[key]).ravel()
        flat.extend(float(x) for x in arr)
    return flat


def _coerce_components(info: dict[str, Any]) -> dict[str, float]:
    """Pull the reward ``components`` dict out of the seam's info and coerce to floats."""
    raw = info.get("components")
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items()}


def _jsonsafe_info(info: dict[str, Any]) -> dict[str, Any]:
    """Strip the seam's info dict of non-JSON-safe values for the wire.

    Drops ``action_mask`` and ``components`` (they have dedicated wire fields),
    coerces numpy scalars to Python types, and silently drops anything else — the
    wire is for display, not authority.
    """
    safe: dict[str, Any] = {}
    for key, value in info.items():
        if key in ("action_mask", "components"):
            continue
        if isinstance(value, (str, bool, int)) and not isinstance(value, np.bool_):
            safe[key] = value
        elif isinstance(value, float):
            safe[key] = float(value)
        elif isinstance(value, np.floating):
            safe[key] = float(value)
        elif isinstance(value, np.integer):
            safe[key] = int(value)
        elif isinstance(value, np.bool_):
            safe[key] = bool(value)
        elif value is None:
            safe[key] = None
        # Else: silently drop unsupported types — the wire is operator-display; structured data
        # goes through the protocol's dedicated fields.
    return safe


class _ActionLogWriter:
    """Per-tick JSONL action-log writer (append + flush per record)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        # Line-buffered text mode + explicit flush per write: a crash leaves complete lines.
        self._fh = path.open("a", encoding="utf-8", buffering=1)

    def write_record(self, record: ActionLogRecord) -> None:
        """Serialize one record + append + flush. Used per tick."""
        self._fh.write(record.to_json() + "\n")
        self._fh.flush()

    def close(self) -> None:
        """fsync + close; safe to call once."""
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            # fsync may not be supported on every filesystem (e.g. tmpfs in some
            # CI containers); the explicit flush above already pushed the data.
            pass
        self._fh.close()


# ---------------------------------------------------------------------------- #
# The server class — internal; not exported
# ---------------------------------------------------------------------------- #


@dataclass
class _GameServer:
    """Per-game state machine; one instance per :func:`serve_game` invocation.

    Holds the env, the connected-clients dict, the action-log writer, and the
    running per-seat reward accumulators used to compute :class:`FinalScores`.
    Callers go through :func:`serve_game`.
    """

    config: ServeConfig
    game_id: str
    run_dir: Path
    env: RetailParallelEnv
    # ``possible_agents`` mirrored to a list[str]; the env's ordering is authoritative.
    agent_ids: list[str] = field(default_factory=list)
    # seat_index -> _ClientState; populated during the registration window.
    clients: dict[int, _ClientState] = field(default_factory=dict)
    # The action-log writer; opened in serve_game(), closed at the end.
    action_log: _ActionLogWriter | None = None
    # Most-recent per-agent obs / reward / info (carried between ticks).
    current_obs: dict[str, npt.NDArray[np.float32]] = field(default_factory=dict)
    current_rewards: dict[str, float] = field(default_factory=dict)
    current_infos: dict[str, dict[str, Any]] = field(default_factory=dict)
    cumulative_score_per_seat: dict[int, float] = field(default_factory=dict)
    # Per-seat per-component cumulative breakdown (for FinalScores).
    cumulative_components_per_seat: dict[int, dict[str, float]] = field(default_factory=dict)
    # Per-tick wall-clock variance samples (fast mode skips the schedule check).
    tick_variance_samples: list[float] = field(default_factory=list)
    # Set when registration completes (all N clients in).
    registration_complete: asyncio.Event = field(default_factory=asyncio.Event)
    # Lock guards ``clients`` mutations during the concurrent registration handlers.
    registration_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ---------------------------------------------------------------------------- #
# Registration handler
# ---------------------------------------------------------------------------- #


async def _send_error_and_close(websocket: Any, error_kind: str, message: str) -> None:
    """Send an :class:`ErrorMessage` and close the connection (best-effort; swallows ConnectionClosed)."""
    try:
        await websocket.send(
            ErrorMessage(error_kind=error_kind, message=message, recoverable=False).to_json()
        )
    except Exception:
        # The peer may already have closed; nothing more we can do.
        _log.debug("failed to send ErrorMessage(%s) — peer may be gone", error_kind)
    try:
        await websocket.close()
    except Exception:
        _log.debug("failed to close peer cleanly after ErrorMessage(%s)", error_kind)


async def _handle_registration(
    websocket: Any,
    server: _GameServer,
    hello_timeout_seconds: float = 5.0,
) -> None:
    """One ``websockets.serve`` handler invocation: handshake + seat assignment.

    Awaits the client's :class:`Hello` frame (short timeout), validates protocol
    version + token, assigns the next available seat, sends
    :class:`SeatAssigned`, then parks until the socket closes (the per-tick loop
    drives all I/O after this point). Returning early implies the connection was
    rejected.
    """
    # 1. Read the Hello frame under a short timeout.
    try:
        raw = await asyncio.wait_for(websocket.recv(), timeout=hello_timeout_seconds)
    except asyncio.TimeoutError:
        await _send_error_and_close(websocket, "hello_timeout", "no Hello within timeout")
        _log.warning("registration: peer did not send Hello within %.1fs", hello_timeout_seconds)
        return
    except Exception as exc:
        _log.warning("registration: error reading Hello frame: %s", exc)
        return

    # 2. Parse + dispatch — the boundary validator raises ValueError on bad input.
    try:
        message = from_json_any(raw)
    except ValueError as exc:
        await _send_error_and_close(websocket, "malformed_hello", f"could not parse: {exc}")
        return
    if not isinstance(message, Hello):
        await _send_error_and_close(
            websocket, "unexpected_message", f"expected Hello, got {type(message).__name__}"
        )
        return
    hello: Hello = message

    # 3. Protocol-version check (HARD reject on mismatch).
    if hello.protocol_version != PROTOCOL_VERSION:
        await _send_error_and_close(
            websocket,
            "protocol_mismatch",
            f"server PROTOCOL_VERSION={PROTOCOL_VERSION}, client sent {hello.protocol_version}",
        )
        _log.warning(
            "registration: protocol mismatch (server=%d, client=%d) from %s",
            PROTOCOL_VERSION,
            hello.protocol_version,
            hello.agent_name,
        )
        return

    # 4. Schema-version check (WARN-only; the client may still be correct).
    if hello.schema_version != ACTION_SCHEMA_VERSION:
        _log.warning(
            "registration: schema version mismatch (server=%d, client=%d) from %s",
            ACTION_SCHEMA_VERSION,
            hello.schema_version,
            hello.agent_name,
        )

    # 5. Token auth (when --token was set; --no-auth bypasses).
    if server.config.token is not None:
        client_token = hello.token or ""
        if not secrets.compare_digest(client_token, server.config.token):
            await _send_error_and_close(websocket, "auth_failed", "shared-token mismatch")
            _log.warning("registration: auth failed for %s", hello.agent_name)
            return

    # 6. Assign the next available seat under the lock (concurrent registrations).
    async with server.registration_lock:
        if server.registration_complete.is_set():
            # All seats already filled; reject this late connection.
            await _send_error_and_close(
                websocket,
                "registration_complete",
                "all seats already filled",
            )
            return
        seat_index = len(server.clients)
        if seat_index >= server.config.n_agents:
            await _send_error_and_close(
                websocket,
                "registration_complete",
                "all seats already filled",
            )
            return
        agent_id = server.agent_ids[seat_index]
        server.clients[seat_index] = _ClientState(
            seat_index=seat_index,
            websocket=websocket,
            client_name=hello.client_name,
            agent_name=hello.agent_name,
        )
        if len(server.clients) == server.config.n_agents:
            server.registration_complete.set()

    # 7. Send SeatAssigned (after the lock is released — the assignment is durable; a
    #    failed send surfaces later as a missed deadline).
    seat_assigned = SeatAssigned(
        seat_index=seat_index,
        n_agents=server.config.n_agents,
        n_npcs=len(server.config.npc_archetypes),
        npc_archetypes=server.config.npc_archetypes,
        n_ticks=server.config.n_ticks,
        game_id=server.game_id,
        agent_id=agent_id,
        tick_interval_ns=int(server.config.tick_interval_seconds * 1e9),
        scenario_name=server.config.scenario_name,
    )
    try:
        await websocket.send(seat_assigned.to_json())
    except Exception as exc:
        _log.warning("registration: failed to send SeatAssigned to seat %d: %s", seat_index, exc)
        return
    _log.info(
        "registration: seat %d assigned to %s (client_name=%s)",
        seat_index,
        hello.agent_name,
        hello.client_name,
    )

    # 8. Park the handler until the game ends: the per-tick loop reuses this websocket via
    #    server.clients[seat].websocket. The handler returns when the socket closes so
    #    websockets.serve()'s wait_closed() can complete at game end.
    await websocket.wait_closed()


# ---------------------------------------------------------------------------- #
# Per-tick action collection
# ---------------------------------------------------------------------------- #


async def _receive_action_for_seat(
    client: _ClientState,
    expected_tick: int,
    timeout_seconds: float,
) -> tuple[int, npt.NDArray[np.float32]] | None:
    """Receive one on-tick :class:`ActionMessage` for the seat within the deadline.

    Returns ``(seat_index, action)`` on a valid, on-tick action; ``None`` on
    timeout, peer error, late tick, or malformed frame (the caller treats absence
    as a missed deadline).
    """
    try:
        raw = await asyncio.wait_for(client.websocket.recv(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return None
    except Exception as exc:
        _log.warning(
            "seat %d: connection error during action recv at tick %d: %s",
            client.seat_index,
            expected_tick,
            exc,
        )
        return None

    try:
        message = from_json_any(raw)
    except ValueError as exc:
        _log.warning(
            "seat %d: malformed action frame at tick %d: %s",
            client.seat_index,
            expected_tick,
            exc,
        )
        return None
    if not isinstance(message, ActionMessage):
        _log.warning(
            "seat %d: expected ActionMessage at tick %d, got %s",
            client.seat_index,
            expected_tick,
            type(message).__name__,
        )
        return None
    if message.tick != expected_tick:
        _log.warning(
            "seat %d: action tick mismatch (got %d, expected %d) — dropping",
            client.seat_index,
            message.tick,
            expected_tick,
        )
        return None

    action_array = np.asarray(message.action, dtype=np.float32)
    return client.seat_index, action_array


async def _collect_actions_for_tick(
    server: _GameServer, tick: int, deadline_seconds: float
) -> dict[int, npt.NDArray[np.float32]]:
    """Race N per-seat receivers against the wall-clock deadline.

    Returns ``{seat_index: action}`` for the seats that submitted in time; missing
    seats are filled later by :func:`permute_actions_by_seat`. Each receiver is
    its own task so a slow seat does not block a fast one.
    """
    arrived: dict[int, npt.NDArray[np.float32]] = {}
    if deadline_seconds <= 0.0:
        # Fast mode (--tick-interval=0): give each seat one zero-timeout opportunity so an
        # action that arrived during the prior broadcast still lands; do not block.
        deadline_seconds = 0.0
    tasks = [
        asyncio.create_task(_receive_action_for_seat(client, tick, max(deadline_seconds, 0.0)))
        for client in server.clients.values()
    ]
    # Wait for all tasks or the deadline (the per-receiver wait_for enforces the deadline;
    # gather just joins). return_exceptions=True shields from a per-task exception.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for outcome in results:
        if isinstance(outcome, BaseException):
            # The per-receiver path already logs; nothing to add here.
            continue
        if outcome is None:
            continue
        seat_index, action_array = outcome
        arrived[seat_index] = action_array
    return arrived


# ---------------------------------------------------------------------------- #
# Game-end helpers
# ---------------------------------------------------------------------------- #


def _build_manifest(server: _GameServer, started_at_iso: str) -> ReplayManifest:
    """Snapshot the game inputs into a :class:`ReplayManifest`."""
    return ReplayManifest(
        game_id=server.game_id,
        seed=server.config.seed,
        n_agents=server.config.n_agents,
        npc_archetypes=server.config.npc_archetypes,
        n_ticks=server.config.n_ticks,
        started_at_iso=started_at_iso,
        schema_version=ACTION_SCHEMA_VERSION,
        software_version=retail_simulator.__version__,
    )


def _compute_winner(cumulative_score_per_seat: dict[int, float]) -> int | None:
    """Return the seat with the strictly-highest score, or ``None`` on tie/empty."""
    if not cumulative_score_per_seat:
        return None
    sorted_scores = sorted(cumulative_score_per_seat.items(), key=lambda kv: kv[1], reverse=True)
    if len(sorted_scores) >= 2 and sorted_scores[0][1] == sorted_scores[1][1]:
        return None
    return sorted_scores[0][0]


def _compute_wall_clock_variance(samples: list[float], tick_interval_seconds: float) -> float:
    """Std-dev of per-tick elapsed times divided by the configured interval.

    Returns ``0.0`` in fast mode or when there are fewer than 2 samples.
    """
    if tick_interval_seconds <= 0.0:
        return 0.0
    if len(samples) < 2:
        return 0.0
    return float(np.std(samples) / tick_interval_seconds)


async def _broadcast_observation(
    server: _GameServer, tick: int, seat: int, deadline_ns: int
) -> None:
    """Send one :class:`ObservationMessage` to one connected client; a broken peer just misses its tick."""
    client = server.clients[seat]
    agent_id = server.agent_ids[seat]
    obs_array = server.current_obs[agent_id]
    reward = server.current_rewards.get(agent_id, 0.0)
    info = server.current_infos.get(agent_id, {})
    components = _coerce_components(info)
    action_mask = _flatten_action_mask(info.get("action_mask"))
    message = ObservationMessage(
        tick=tick,
        observation=[float(x) for x in obs_array.tolist()],
        reward=float(reward),
        info=_jsonsafe_info(info),
        seat_index=seat,
        action_mask=action_mask,
        deadline_ns=deadline_ns,
        components=components,
    )
    try:
        await client.websocket.send(message.to_json())
    except Exception as exc:
        _log.warning(
            "broadcast: failed to send observation to seat %d at tick %d: %s",
            seat,
            tick,
            exc,
        )


# ---------------------------------------------------------------------------- #
# The async entry point
# ---------------------------------------------------------------------------- #


async def serve_game(config: ServeConfig) -> ServeResult:
    """Run one game lifecycle end-to-end; return a :class:`ServeResult`.

    Lazy-imports ``websockets`` (raises :class:`ImportError` with an install hint
    when the ``[serve]`` extra is missing). Opens a listener, runs the registration
    window, drives the per-tick loop, broadcasts FinalScores, writes the manifest
    + action log.
    """
    try:
        import websockets
    except ImportError as exc:
        raise ImportError(
            "retail-sim serve requires the [serve] extra. "
            "Install with: pip install retail-simulator[serve]"
        ) from exc

    game_id = f"game-{uuid.uuid4().hex[:12]}"
    run_dir = config.runs_dir / game_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Build the env and seed the world.
    env = RetailParallelEnv(
        config=CoreConfig.default(),
        n_learning_agents=config.n_agents,
        npc_archetypes=config.npc_archetypes,
        seed=config.seed,
    )
    obs, infos = env.reset(seed=config.seed)

    server = _GameServer(
        config=config,
        game_id=game_id,
        run_dir=run_dir,
        env=env,
        agent_ids=list(env.possible_agents),
        current_obs=dict(obs),
        current_rewards={aid: 0.0 for aid in env.possible_agents},
        current_infos=dict(infos),
        cumulative_score_per_seat={i: 0.0 for i in range(config.n_agents)},
        cumulative_components_per_seat={i: {} for i in range(config.n_agents)},
    )

    started_at_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest = _build_manifest(server, started_at_iso)
    _atomic_write_text(run_dir / "game_manifest.json", manifest.to_json())

    server.action_log = _ActionLogWriter(run_dir / "action_log.jsonl")

    rounds_completed = 0
    exit_status = 0
    summary = ""
    ws_server: Any = None

    try:
        # ---- 2. Start the websockets server.
        async def handler(websocket: Any) -> None:
            await _handle_registration(websocket, server)

        ws_server = await websockets.serve(handler, config.host, config.port)
        _log.info(
            "serve: listening on ws://%s:%d for %d clients (game_id=%s)",
            config.host,
            config.port,
            config.n_agents,
            game_id,
        )

        # ---- 3. Wait for full registration or timeout.
        try:
            await asyncio.wait_for(
                server.registration_complete.wait(),
                timeout=config.registration_timeout_seconds,
            )
        except asyncio.TimeoutError:
            connected = len(server.clients)
            _log.warning(
                "serve: registration timeout (%d/%d clients connected after %.1fs)",
                connected,
                config.n_agents,
                config.registration_timeout_seconds,
            )
            # Best-effort notify any already-connected clients.
            for client in list(server.clients.values()):
                await _send_error_and_close(
                    client.websocket,
                    "registration_timeout",
                    f"only {connected}/{config.n_agents} clients connected",
                )
            exit_status = 1
            summary = f"game {game_id} registration_timeout: {connected}/{config.n_agents} clients"
            return ServeResult(
                game_id=game_id,
                run_dir=run_dir,
                rounds_completed=0,
                exit_status=exit_status,
                summary=summary,
                final_scores={},
                winner_seat=None,
            )

        _log.info("serve: registration complete; starting %d-tick game", config.n_ticks)

        # ---- 4. Per-tick loop.
        game_start_monotonic = time.monotonic()
        tick_interval = config.tick_interval_seconds
        # Per-tick action deadline = 80% of the tick interval (leaves 20% to step + broadcast); 0 in fast mode.
        per_tick_deadline_seconds = tick_interval * 0.8 if tick_interval > 0 else 0.0

        for tick in range(config.n_ticks):
            tick_start_monotonic = time.monotonic()
            # ---- 4a. Broadcast the current obs (reset obs at t=0; step obs at t>0).
            # deadline_ns is the absolute monotonic deadline for THIS tick's action.
            broadcast_at_ns = time.monotonic_ns()
            deadline_ns = broadcast_at_ns + int(per_tick_deadline_seconds * 1e9)
            broadcast_tasks = [
                asyncio.create_task(_broadcast_observation(server, tick, seat, deadline_ns))
                for seat in sorted(server.clients.keys())
            ]
            await asyncio.gather(*broadcast_tasks, return_exceptions=True)

            # ---- 4b. Collect actions from each seat (race against the deadline).
            arrived = await _collect_actions_for_tick(server, tick, per_tick_deadline_seconds)

            # ---- 4c. Permute by seat. Arrived seats update last_valid_action so the next
            # tick's miss falls back to THIS tick's action.
            last_valid: dict[int, npt.NDArray[np.float32]] = {
                seat: client.last_valid_action
                for seat, client in server.clients.items()
                if client.last_valid_action is not None
            }
            # joint_action MUST go through permute_actions_by_seat regardless of arrival order
            # (the load-bearing determinism primitive).
            joint_action, missing_seats = permute_actions_by_seat(
                arrived=arrived,
                last_valid=last_valid,
                n_seats=config.n_agents,
                registry_default_action=REGISTRY_DEFAULT_ACTION,
            )
            for seat in arrived:
                server.clients[seat].last_valid_action = arrived[seat]
            for seat in missing_seats:
                server.clients[seat].missed_deadlines += 1
                _log.warning(
                    "tick %d: seat %d missed deadline; falling back to %s",
                    tick,
                    seat,
                    "last_valid_action"
                    if server.clients[seat].last_valid_action is not None
                    else "registry default",
                )

            # ---- 4d. Step the env EXACTLY ONCE (the seam call).
            obs, rewards, _terms, _truncs, infos = server.env.step(joint_action)
            server.current_obs = dict(obs)
            server.current_rewards = dict(rewards)
            server.current_infos = dict(infos)
            for seat in range(config.n_agents):
                agent_id = server.agent_ids[seat]
                reward = float(rewards.get(agent_id, 0.0))
                server.cumulative_score_per_seat[seat] += reward
                components = _coerce_components(infos.get(agent_id, {}))
                seat_components = server.cumulative_components_per_seat[seat]
                for comp_name, comp_value in components.items():
                    seat_components[comp_name] = seat_components.get(comp_name, 0.0) + comp_value

            # ---- 4e. Write the action-log record.
            log_record = ActionLogRecord(
                tick=tick,
                joint_action={
                    aid: [float(x) for x in joint_action[aid].tolist()] for aid in joint_action
                },
                observations={aid: [float(x) for x in obs[aid].tolist()] for aid in obs},
                rewards={aid: float(rewards[aid]) for aid in rewards},
                missing_seats=missing_seats,
                deadline_ns=deadline_ns,
            )
            assert server.action_log is not None
            server.action_log.write_record(log_record)

            rounds_completed = tick + 1

            # ---- 4f. Wall-clock scheduling: sleep until the next tick boundary.
            tick_elapsed = time.monotonic() - tick_start_monotonic
            if tick_interval > 0:
                next_boundary = game_start_monotonic + (tick + 1) * tick_interval
                sleep_for = next_boundary - time.monotonic()
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                server.tick_variance_samples.append(tick_elapsed)
                deviation = abs(tick_elapsed - tick_interval) / tick_interval
                if deviation > 0.10:
                    _log.warning(
                        "tick %d: wall-clock variance %.1f%% over interval",
                        tick,
                        deviation * 100,
                    )
            # Log progress every 10 ticks (always for long intervals).
            if tick_interval >= 1.0 or tick % 10 == 0:
                _log.info(
                    "tick %d/%d complete (rewards=%s)",
                    tick + 1,
                    config.n_ticks,
                    {aid: round(r, 3) for aid, r in rewards.items()},
                )

        # ---- 5. End-of-game broadcast: FinalScores.
        wall_clock_variance = _compute_wall_clock_variance(
            server.tick_variance_samples, tick_interval
        )
        final_scores = {
            server.agent_ids[seat]: float(score)
            for seat, score in server.cumulative_score_per_seat.items()
        }
        per_seat_components_wire = {
            server.agent_ids[seat]: dict(components)
            for seat, components in server.cumulative_components_per_seat.items()
        }
        missed_deadlines_per_seat = {
            server.agent_ids[seat]: client.missed_deadlines
            for seat, client in server.clients.items()
        }
        winner_seat = _compute_winner(server.cumulative_score_per_seat)

        final = FinalScores(
            final_scores=final_scores,
            winner_seat=winner_seat,
            per_seat_components=per_seat_components_wire,
            total_ticks=rounds_completed,
            missed_deadlines_per_seat=missed_deadlines_per_seat,
            wall_clock_variance=wall_clock_variance,
        )
        final_payload = final.to_json()
        for client in list(server.clients.values()):
            try:
                await client.websocket.send(final_payload)
            except Exception as exc:
                _log.warning(
                    "final: failed to send FinalScores to seat %d: %s",
                    client.seat_index,
                    exc,
                )

        # Write the serve_summary.json artifact (operator-readable).
        serve_summary = {
            "game_id": game_id,
            "rounds_completed": rounds_completed,
            "exit_status": 0,
            "final_scores": final_scores,
            "winner_seat": winner_seat,
            "missed_deadlines_per_seat": missed_deadlines_per_seat,
            "wall_clock_variance": wall_clock_variance,
        }
        _atomic_write_text(run_dir / "serve_summary.json", json.dumps(serve_summary, indent=2))

        # Close all sockets gracefully.
        for client in list(server.clients.values()):
            try:
                await client.websocket.close()
            except Exception:
                pass

        winner_label = f"seat_{winner_seat}" if winner_seat is not None else "tie"
        summary = f"game {game_id} completed: {rounds_completed} ticks; winner={winner_label}"
        _log.info("serve: %s", summary)
        return ServeResult(
            game_id=game_id,
            run_dir=run_dir,
            rounds_completed=rounds_completed,
            exit_status=0,
            summary=summary,
            final_scores=final_scores,
            winner_seat=winner_seat,
        )

    except Exception as exc:
        _log.exception("serve: unhandled error during game %s", game_id)
        exit_status = 2
        summary = f"game {game_id} aborted: {exc}"
        # Best-effort error broadcast.
        for client in list(server.clients.values()):
            await _send_error_and_close(client.websocket, "server_internal", f"server error: {exc}")
        return ServeResult(
            game_id=game_id,
            run_dir=run_dir,
            rounds_completed=rounds_completed,
            exit_status=exit_status,
            summary=summary,
            final_scores={},
            winner_seat=None,
        )

    finally:
        # Always close the action log + the websockets server, even on error.
        if server.action_log is not None:
            server.action_log.close()
        if ws_server is not None:
            ws_server.close()
            try:
                await ws_server.wait_closed()
            except Exception:
                pass


__all__ = ["ServeConfig", "ServeResult", "serve_game"]
