"""CLI commands: live serve, run-series, replay, play."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:
    # Type-only: heavy live imports stay lazy inside command bodies.
    from collections.abc import Callable

    from retail_simulator.live import SeriesResult, ServeConfig, ServeResult

from . import _atomic_write_text, _echo, _log, _status, live_app


def _resolve_token(no_auth: bool) -> str | None:
    """Resolve the shared-secret token from ``--no-auth`` / ``RETAIL_SIM_TOKEN``.

    ``--no-auth`` returns ``None`` (skips the env check); otherwise
    ``RETAIL_SIM_TOKEN`` must be set or the command exits 1 with a clean
    diagnostic. Shared by ``live serve``, ``live run-series``, and ``live play``.
    """
    if no_auth:
        return None
    token = os.environ.get("RETAIL_SIM_TOKEN")
    if token is None:
        _echo(
            _status(
                "[FAIL]",
                "RETAIL_SIM_TOKEN env var must be set (or use --no-auth for local testing)",
                "red",
            )
        )
        raise typer.Exit(code=1)
    return token


def _build_serve_config(
    *,
    n_agents: int,
    tick_interval: float,
    n_ticks: int,
    host: str,
    port: int,
    registration_timeout: float,
    no_auth: bool,
    npc_archetypes: str,
    scenario_name: str,
    runs_dir: Path,
    seed: int,
) -> ServeConfig:
    """Resolve live-game operator flags into a :class:`ServeConfig`.

    Shared by ``live serve`` and ``live run-series``: parses
    ``--npc-archetypes`` (unknown names exit 1 with the known list) and resolves
    the ``--no-auth`` / ``RETAIL_SIM_TOKEN`` token.
    """
    from retail_simulator.core.config import NPC_ARCHETYPES
    from retail_simulator.live import ServeConfig

    # Empty string -> empty tuple (all-remote mode).
    parsed_archetypes = tuple(name.strip() for name in npc_archetypes.split(",") if name.strip())
    unknown = [a for a in parsed_archetypes if a not in NPC_ARCHETYPES]
    if unknown:
        _echo(_status("[FAIL]", f"unknown NPC archetype(s): {unknown}", "red"))
        _echo(f"Known archetypes: {list(NPC_ARCHETYPES)}")
        raise typer.Exit(code=1)

    # Token resolution is shared with `live play` via _resolve_token.
    token = _resolve_token(no_auth)

    return ServeConfig(
        n_agents=n_agents,
        n_ticks=n_ticks,
        tick_interval_seconds=tick_interval,
        host=host,
        port=port,
        registration_timeout_seconds=registration_timeout,
        token=token,
        npc_archetypes=parsed_archetypes,
        scenario_name=scenario_name,
        runs_dir=runs_dir,
        seed=seed,
    )


@live_app.command("serve")
def live_serve(
    n_agents: int = typer.Option(2, "--n-agents", help="Number of learning seats."),
    tick_interval: float = typer.Option(
        5.0,
        "--tick-interval",
        help="Seconds per tick; 0 = CI fast mode (F5=A — no wall-clock sleep).",
    ),
    n_ticks: int = typer.Option(104, "--n-ticks", help="Game length (per)."),
    host: str = typer.Option("localhost", "--host", help="WebSocket bind host."),
    port: int = typer.Option(8765, "--port", help="WebSocket bind port."),
    registration_timeout: float = typer.Option(
        60.0,
        "--registration-timeout",
        help="F4=B abort after this many seconds with insufficient clients.",
    ),
    no_auth: bool = typer.Option(
        False,
        "--no-auth",
        help="Disable shared-token auth. Token via RETAIL_SIM_TOKEN env var when set; F2=A.",
    ),
    npc_archetypes: str = typer.Option(
        "",
        "--npc-archetypes",
        help="F3=B comma-separated NPC archetypes (e.g. 'discounter,premium'); empty = all-remote.",
    ),
    scenario_name: str = typer.Option(
        "default", "--scenario", help="Scenario name for SeatAssigned + manifest."
    ),
    runs_dir: Path = typer.Option(
        Path("runs"), "--runs-dir", help="Where to write runs/<game-id>/."
    ),
    seed: int = typer.Option(0, "--seed", help="Core PCG64 seed."),
) -> None:
    """Run a single live game over WebSocket and print the :class:`ServeResult` as JSON.

    Exits with ``result.exit_status`` (0 clean game, 1 registration timeout,
    2 server error).
    """
    import asyncio

    from retail_simulator.live import serve_game

    config = _build_serve_config(
        n_agents=n_agents,
        tick_interval=tick_interval,
        n_ticks=n_ticks,
        host=host,
        port=port,
        registration_timeout=registration_timeout,
        no_auth=no_auth,
        npc_archetypes=npc_archetypes,
        scenario_name=scenario_name,
        runs_dir=runs_dir,
        seed=seed,
    )

    _log.info(
        "running live serve: n_agents=%d, n_ticks=%d, tick_interval=%.2fs, "
        "host=%s, port=%d, npc_archetypes=%s, scenario=%s, runs_dir=%s, seed=%d",
        n_agents,
        n_ticks,
        tick_interval,
        host,
        port,
        list(config.npc_archetypes),
        scenario_name,
        runs_dir,
        seed,
    )

    result = asyncio.run(serve_game(config))

    # ``default=str`` handles the Path on ``run_dir``.
    _echo(json.dumps(asdict(result), indent=2, default=str))

    # Honor the server's exit status: 0 clean game, 1 registration timeout, 2 error.
    if result.exit_status != 0:
        raise typer.Exit(code=result.exit_status)



def _format_series_table(series: SeriesResult) -> str:
    """Render a :class:`SeriesResult` as a plain-ASCII summary table (presentation-only)."""
    lines = [
        f"series_id={series.series_id}",
        (
            f"games: requested={series.games_requested} "
            f"played={series.games_played} failed={series.games_failed}"
        ),
        "",
    ]

    seat_header = f"{'seat':>12} {'mean':>14} {'stddev':>14}"
    lines.append(seat_header)
    lines.append("-" * len(seat_header))
    for seat in sorted(series.per_seat_mean):
        mean = series.per_seat_mean[seat]
        stddev = series.per_seat_stddev[seat]
        mean_str = "n/a" if mean is None else f"{mean:.4f}"
        stddev_str = "n/a" if stddev is None else f"{stddev:.4f}"
        lines.append(f"{seat:>12} {mean_str:>14} {stddev_str:>14}")

    lines.append("")
    game_header = f"{'idx':>4} {'game_id':>18} {'seed':>8} {'exit':>6}"
    lines.append(game_header)
    lines.append("-" * len(game_header))
    for game in series.games:
        lines.append(
            f"{game.game_index:>4} {game.game_id:>18} {game.seed:>8} {game.exit_status:>6}"
        )
    return "\n".join(lines)


async def _wait_for_listener(host: str, port: int, *, timeout_seconds: float = 10.0) -> None:
    """Poll ``host:port`` until a TCP connect succeeds or the timeout elapses.

    A bind-probe readiness check (robust against a client racing the listener);
    raises ``TimeoutError`` if the listener never comes up.
    """
    import asyncio
    import contextlib

    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout_seconds
    while loop.time() < deadline:
        try:
            _reader, writer = await asyncio.open_connection(host, port)
        except OSError:
            await asyncio.sleep(0.02)
            continue
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        return
    raise TimeoutError(f"self-play: server at {host}:{port} did not bind within {timeout_seconds}s")


async def _serve_with_clients(config: ServeConfig, policy: str) -> ServeResult:
    """Run one game's server AND its ``n_agents`` internal clients concurrently.

    One asyncio loop hosts :func:`serve_game` plus one :func:`play_one_game`
    client per learning seat (NPC seats are filled by the server), so a game
    runs hands-free with no external clients. Clients connect only after
    :func:`_wait_for_listener` confirms the bind; a failed client is logged at
    WARNING, never masking the server's :class:`ServeResult` (which is returned;
    the clients' own scores are discarded).
    """
    import asyncio
    import contextlib

    from retail_simulator.live import play_one_game, serve_game

    uri = f"ws://{config.host}:{config.port}"
    # Internal clients use the server's own token (None under --no-auth).
    token = config.token

    server_task = asyncio.create_task(serve_game(config))
    client_tasks: list[asyncio.Task[Any]] = []
    try:
        # Wait for the bind (not a fixed sleep); on timeout, awaiting server_task
        # below surfaces its ServeResult rather than a bare TimeoutError.
        with contextlib.suppress(TimeoutError):
            await _wait_for_listener(config.host, config.port)

        client_tasks = [
            asyncio.create_task(play_one_game(uri, f"selfplay-{i}", token, policy=policy))
            for i in range(config.n_agents)
        ]

        # Await the server last — it holds the authoritative ServeResult.
        result: ServeResult = await server_task
    finally:
        # Drain the clients even if the server raised, so no task is orphaned;
        # return_exceptions keeps client errors from masking the server result.
        outcomes = await asyncio.gather(*client_tasks, return_exceptions=True)
        for i, outcome in enumerate(outcomes):
            if isinstance(outcome, BaseException):
                _log.warning("self-play internal client selfplay-%d failed: %s", i, outcome)
    return result


def _make_self_play_serve_fn(policy: str) -> Callable[[ServeConfig], ServeResult]:
    """Build the ``serve_fn`` that :func:`run_series` uses under ``--self-play``.

    Each game runs :func:`_serve_with_clients` in its own ``asyncio.run`` —
    games stay sequential; only the per-game server + clients are concurrent.
    """
    import asyncio

    def _serve_fn(cfg: ServeConfig) -> ServeResult:
        return asyncio.run(_serve_with_clients(cfg, policy))

    return _serve_fn


@live_app.command("run-series")
def live_run_series(
    games: int = typer.Option(..., "--games", help="Number of games in the series."),
    n_agents: int = typer.Option(2, "--n-agents", help="Number of learning seats."),
    tick_interval: float = typer.Option(
        5.0,
        "--tick-interval",
        help="Seconds per tick; 0 = CI fast mode (F5=A — no wall-clock sleep).",
    ),
    n_ticks: int = typer.Option(104, "--n-ticks", help="Game length (per)."),
    host: str = typer.Option("localhost", "--host", help="WebSocket bind host."),
    port: int = typer.Option(8765, "--port", help="WebSocket bind port."),
    registration_timeout: float = typer.Option(
        60.0,
        "--registration-timeout",
        help="F4=B abort after this many seconds with insufficient clients.",
    ),
    no_auth: bool = typer.Option(
        False,
        "--no-auth",
        help="Disable shared-token auth. Token via RETAIL_SIM_TOKEN env var when set; F2=A.",
    ),
    npc_archetypes: str = typer.Option(
        "",
        "--npc-archetypes",
        help="F3=B comma-separated NPC archetypes (e.g. 'discounter,premium'); empty = all-remote.",
    ),
    scenario_name: str = typer.Option(
        "default", "--scenario", help="Scenario name for SeatAssigned + manifest."
    ),
    runs_dir: Path = typer.Option(
        Path("runs"), "--runs-dir", help="Where to write runs/<game-id>/."
    ),
    seed: int = typer.Option(0, "--seed", help="Core PCG64 seed (game i uses seed+i)."),
    table: bool = typer.Option(
        False, "--table", help="Render an ASCII summary table instead of JSON."
    ),
    self_play: bool = typer.Option(
        False,
        "--self-play",
        help="Run each game's server AND its n-agents internal default-policy clients "
        "concurrently in-process — a hands-free series with NO external clients. "
        "Off (default) waits for external clients exactly as before.",
    ),
    policy: str = typer.Option(
        "default",
        "--policy",
        help="Per-tick action policy for the internal --self-play clients. "
        "'default' submits the registry-default action every tick.",
    ),
) -> None:
    """Run ``--games`` sequential live games and aggregate per-seat scores.

    Game ``i`` runs with ``seed + i`` on the same port. Writes a sidecar
    ``runs/<series-id>/series_summary.json`` and prints the
    :class:`SeriesResult` as JSON (``--table`` for the ASCII view). Exits 1
    only if zero games finished cleanly.
    """
    from retail_simulator.live import run_series

    # Clean [FAIL] + exit 1 for --games < 1 (Typer's bare min= would exit 2
    # with no context).
    if games < 1:
        _echo(_status("[FAIL]", f"--games must be >= 1 (got {games})", "red"))
        raise typer.Exit(code=1)

    # Validate the --self-play policy up front so a bad value never reaches the
    # serve_fn.
    if self_play and policy != "default":
        _echo(
            _status(
                "[FAIL]",
                f"unknown --policy {policy!r}; only 'default' is supported in this slice",
                "red",
            )
        )
        raise typer.Exit(code=1)

    base_config = _build_serve_config(
        n_agents=n_agents,
        tick_interval=tick_interval,
        n_ticks=n_ticks,
        host=host,
        port=port,
        registration_timeout=registration_timeout,
        no_auth=no_auth,
        npc_archetypes=npc_archetypes,
        scenario_name=scenario_name,
        runs_dir=runs_dir,
        seed=seed,
    )

    _log.info(
        "running live run-series: games=%d, n_agents=%d, n_ticks=%d, tick_interval=%.2fs, "
        "host=%s, port=%d, npc_archetypes=%s, scenario=%s, runs_dir=%s, base_seed=%d",
        games,
        n_agents,
        n_ticks,
        tick_interval,
        host,
        port,
        list(base_config.npc_archetypes),
        scenario_name,
        runs_dir,
        seed,
    )

    # Self-play only swaps the serve_fn (in-process clients); aggregation is
    # identical to the external-client path.
    if self_play:
        series = run_series(base_config, games=games, serve_fn=_make_self_play_serve_fn(policy))
    else:
        series = run_series(base_config, games=games)

    # Sidecar under <runs-dir>/<series-id>, so it never collides with per-game dirs.
    series_dir = runs_dir / series.series_id
    series_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "series_id": series.series_id,
        "games_requested": series.games_requested,
        "games_played": series.games_played,
        "games_failed": series.games_failed,
        "per_seat_mean": series.per_seat_mean,
        "per_seat_stddev": series.per_seat_stddev,
        "games": [
            {
                "game_index": g.game_index,
                "game_id": g.game_id,
                "seed": g.seed,
                "exit_status": g.exit_status,
                "final_scores": g.final_scores,
            }
            for g in series.games
        ],
    }
    _atomic_write_text(series_dir / "series_summary.json", json.dumps(summary, indent=2))

    if table:
        _echo(_format_series_table(series))
    else:
        # ``default=str`` matches `live serve`'s echo.
        _echo(json.dumps(asdict(series), indent=2, default=str))

    # Failed games are recorded, never fatal; exit 1 only when zero games
    # finished cleanly.
    if series.games_played == 0:
        _echo(
            _status(
                "[FAIL]",
                f"series {series.series_id}: 0/{series.games_requested} games completed cleanly",
                "red",
            )
        )
        raise typer.Exit(code=1)


@live_app.command("replay")
def live_replay(
    game_dir: Path = typer.Option(
        ...,
        "--game-dir",
        help="Per-game directory (runs/<game-id>/) containing game_manifest.json "
        "+ action_log.jsonl.",
    ),
    strict: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="D-2 strict mode: truncated logs count as divergence. --no-strict accepts "
        "truncation for crashed-server replays.",
    ),
) -> None:
    """Replay a recorded game offline and verify bit-identity; exit 0/1.

    Prints the :class:`ReplayResult` as JSON. Exits 1 on a divergence or (under
    ``--strict``) a truncated action log; a missing game dir/manifest surfaces
    as a clean ``[FAIL]``.
    """
    from retail_simulator.live import replay_game_dir

    try:
        result = replay_game_dir(game_dir, strict=strict)
    except FileNotFoundError as exc:
        _echo(_status("[FAIL]", str(exc), "red"))
        raise typer.Exit(code=1) from exc

    _echo(json.dumps(asdict(result), indent=2, default=str))

    if not result.ok:
        raise typer.Exit(code=1)


@live_app.command("play")
def live_play(
    host: str = typer.Option("localhost", "--host", help="WebSocket server host."),
    port: int = typer.Option(8765, "--port", help="WebSocket server port."),
    agent_name: str = typer.Option(
        "player", "--agent-name", help="Client name echoed to the server + logs."
    ),
    no_auth: bool = typer.Option(
        False,
        "--no-auth",
        help="Connect without a token. Otherwise the token is read from RETAIL_SIM_TOKEN; F2=A.",
    ),
    policy: str = typer.Option(
        "default",
        "--policy",
        help="Per-tick action policy. 'default' submits the registry-default action every tick.",
    ),
    seed: int = typer.Option(
        0, "--seed", help="Policy RNG seed (reserved for future stochastic policies)."
    ),
) -> None:
    """Connect ONE client to a running live game, play it, print final scores.

    Submits one action per tick until FinalScores (printed as JSON); the seat
    assignment is printed when assigned. A single-client primitive — launch two
    clients (or one + an NPC seat) to fill a 2-seat game. Exits 0 on a clean
    end; 1 on ``ConnectionError`` or a game that ends without final scores.
    """
    import asyncio

    from retail_simulator.live import SeatAssigned, play_one_game

    if policy != "default":
        _echo(
            _status(
                "[FAIL]",
                f"unknown --policy {policy!r}; only 'default' is supported in this slice",
                "red",
            )
        )
        raise typer.Exit(code=1)

    token = _resolve_token(no_auth)
    uri = f"ws://{host}:{port}"

    # `seed` is reserved for future stochastic policies; the default policy
    # ignores it.

    _log.info(
        "running live play: uri=%s, agent_name=%s, policy=%s, auth=%s",
        uri,
        agent_name,
        policy,
        "off" if token is None else "on",
    )

    # Print the seat assignment as soon as the server assigns it, before the
    # first action.
    def _print_seat(seat: SeatAssigned) -> None:
        _echo(
            _status(
                "[OK]",
                (f"seat_index={seat.seat_index} game_id={seat.game_id} n_ticks={seat.n_ticks}"),
                "green",
            )
        )

    try:
        final = asyncio.run(
            play_one_game(uri, agent_name, token, policy=policy, on_seat_assigned=_print_seat)
        )
    except ConnectionError as exc:
        _echo(_status("[FAIL]", str(exc), "red"))
        raise typer.Exit(code=1) from exc

    # None means the socket closed without an end-of-game frame — treat as an
    # aborted game.
    if final is None:
        _echo(_status("[FAIL]", "game ended without final scores", "red"))
        raise typer.Exit(code=1)

    _echo(json.dumps(asdict(final), indent=2, default=str))
