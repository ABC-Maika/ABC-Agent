"""The offline replay validator.

Pure Python — no asyncio, no ``websockets``; importable without the
``[serve]`` extra (stdlib + numpy + ``core`` + ``envs``). Rebuilds the env
from the on-disk manifest + action log and diffs each tick. The
reconstruction path pins ``CoreConfig.default()`` and seats only the
manifest's ``n_agents`` + ``npc_archetypes`` (``config_summary`` is a reserved
extension hook).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from retail_simulator.core.config import CoreConfig
from retail_simulator.envs.parallel_env import RetailParallelEnv


@dataclass(frozen=True)
class ReplayManifest:
    """Per-game manifest written alongside the action log.

    Carries every input the determinism contract needs to rebuild the same env:
    the seam's seed, the seat plan, the schema version, and the package version.
    """

    game_id: str
    seed: int
    n_agents: int
    npc_archetypes: tuple[str, ...]
    n_ticks: int
    started_at_iso: str
    schema_version: int
    software_version: str
    # Reserved for richer scenario reconstruction; empty dict for now.
    config_summary: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize as a single JSON object string."""
        payload = asdict(self)
        # Tuples are not JSON-native; ``asdict`` flattens to lists and ``from_json`` rebuilds the tuple.
        return json.dumps(payload, sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, s: str) -> ReplayManifest:
        """Parse a manifest JSON string; raise ``ValueError`` on missing keys.

        Unknown extra keys are ignored (forward-compat).
        """
        try:
            data = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ReplayManifest.from_json: malformed JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"ReplayManifest.from_json: expected JSON object, got {type(data).__name__}"
            )
        try:
            return cls(
                game_id=str(data["game_id"]),
                seed=int(data["seed"]),
                n_agents=int(data["n_agents"]),
                npc_archetypes=tuple(str(a) for a in data["npc_archetypes"]),
                n_ticks=int(data["n_ticks"]),
                started_at_iso=str(data["started_at_iso"]),
                schema_version=int(data["schema_version"]),
                software_version=str(data["software_version"]),
                config_summary=dict(data.get("config_summary", {})),
            )
        except KeyError as exc:
            raise ValueError(
                f"ReplayManifest.from_json: missing required key {exc.args[0]!r}"
            ) from exc


@dataclass(frozen=True)
class ActionLogRecord:
    """One JSONL record per tick (``runs/<game-id>/action_log.jsonl``).

    Carries both halves of a tick — the joint action applied AND the per-seat
    observations + rewards the seam returned. ``missing_seats`` records which
    seats fell back to last-valid-or-default; ``deadline_ns`` is the wall-clock
    deadline the server collected against. Fields use Python lists so the
    dataclass is JSON-safe.
    """

    tick: int
    joint_action: dict[str, list[float]]
    observations: dict[str, list[float]]
    rewards: dict[str, float]
    missing_seats: list[int]
    deadline_ns: int

    def to_json(self) -> str:
        """Serialize as a single-line JSON object (the JSONL contract)."""
        # No indent: JSONL is one record per line (the writer adds the newline).
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> ActionLogRecord:
        """Parse one JSONL line into a record; raise ``ValueError`` on malformed input."""
        try:
            data = json.loads(s)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ActionLogRecord.from_json: malformed JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"ActionLogRecord.from_json: expected JSON object, got {type(data).__name__}"
            )
        try:
            tick = int(data["tick"])
            joint_raw = data["joint_action"]
            observations_raw = data["observations"]
            rewards_raw = data["rewards"]
            missing_raw = data["missing_seats"]
            deadline_ns = int(data["deadline_ns"])
        except KeyError as exc:
            raise ValueError(
                f"ActionLogRecord.from_json: missing required key {exc.args[0]!r}"
            ) from exc
        if not isinstance(joint_raw, dict):
            raise ValueError("ActionLogRecord.from_json: 'joint_action' must be a JSON object")
        if not isinstance(observations_raw, dict):
            raise ValueError("ActionLogRecord.from_json: 'observations' must be a JSON object")
        if not isinstance(rewards_raw, dict):
            raise ValueError("ActionLogRecord.from_json: 'rewards' must be a JSON object")
        if not isinstance(missing_raw, list):
            raise ValueError("ActionLogRecord.from_json: 'missing_seats' must be a JSON array")
        return cls(
            tick=tick,
            joint_action={str(k): [float(x) for x in v] for k, v in joint_raw.items()},
            observations={str(k): [float(x) for x in v] for k, v in observations_raw.items()},
            rewards={str(k): float(v) for k, v in rewards_raw.items()},
            missing_seats=[int(s) for s in missing_raw],
            deadline_ns=deadline_ns,
        )


@dataclass(frozen=True)
class ReplayResult:
    """Outcome of a replay run.

    ``ok`` is ``True`` iff every per-seat ``(observation, reward)`` matched the
    recorded log bit-for-bit. On failure ``first_divergence_tick`` localizes the
    divergence. ``n_ticks_replayed`` counts records processed (1-based).
    """

    ok: bool
    n_ticks_replayed: int
    first_divergence_tick: int | None
    divergence_summary: str | None


def read_manifest(manifest_path: Path) -> ReplayManifest:
    """Read + parse a manifest JSON file; ``FileNotFoundError`` names the missing path."""
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"read_manifest: no manifest at {manifest_path}; "
            f"the file is written by the live server at game start"
        )
    return ReplayManifest.from_json(manifest_path.read_text(encoding="utf-8"))


def read_action_log(log_path: Path) -> Iterator[ActionLogRecord]:
    """Lazily yield one :class:`ActionLogRecord` per JSONL line.

    A trailing blank line is tolerated; a truncated last line surfaces as
    :class:`ValueError` (loud failure preferred).
    """
    if not log_path.exists():
        raise FileNotFoundError(
            f"read_action_log: no action log at {log_path}; "
            f"the file is written by the live server per tick"
        )
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                # Blank line: a trailing editor newline / whitespace; not a record.
                continue
            yield ActionLogRecord.from_json(stripped)


def replay(manifest_path: Path, log_path: Path) -> ReplayResult:
    """Rebuild the env from the manifest, re-step the log, diff each tick; return the verdict."""
    manifest = read_manifest(manifest_path)

    # Reconstruction pins CoreConfig.default(); a richer scenario can switch on manifest.config_summary.
    env = RetailParallelEnv(
        config=CoreConfig.default(),
        n_learning_agents=manifest.n_agents,
        npc_archetypes=manifest.npc_archetypes,
        seed=manifest.seed,
    )
    env.reset(seed=manifest.seed)

    n_processed = 0
    for record in read_action_log(log_path):
        # Build the joint action dict in the same agent-id keying the live server used (retailer_<seat>).
        joint_action = {
            agent_id: np.asarray(action, dtype=np.float32)
            for agent_id, action in record.joint_action.items()
        }
        obs, rewards, _terms, _truncs, _infos = env.step(joint_action)

        # Per-seat verification: bit-exact np.array_equal for obs, exact == for the reward.
        for agent_id, logged_obs_list in record.observations.items():
            logged_obs = np.asarray(logged_obs_list, dtype=np.float32)
            live_obs = obs[agent_id]
            if not np.array_equal(logged_obs, live_obs):
                return ReplayResult(
                    ok=False,
                    n_ticks_replayed=n_processed + 1,
                    first_divergence_tick=record.tick,
                    divergence_summary=(
                        f"observation divergence at tick {record.tick} seat {agent_id}: "
                        f"first delta at dim "
                        f"{int(np.argmax(logged_obs != live_obs))} "
                        f"logged={logged_obs[np.argmax(logged_obs != live_obs)]!r} "
                        f"live={live_obs[np.argmax(logged_obs != live_obs)]!r}"
                    ),
                )

            logged_reward = record.rewards[agent_id]
            live_reward = rewards[agent_id]
            if logged_reward != live_reward:
                return ReplayResult(
                    ok=False,
                    n_ticks_replayed=n_processed + 1,
                    first_divergence_tick=record.tick,
                    divergence_summary=(
                        f"reward divergence at tick {record.tick} seat {agent_id}: "
                        f"logged={logged_reward!r} live={live_reward!r}"
                    ),
                )

        n_processed += 1

    return ReplayResult(
        ok=True,
        n_ticks_replayed=n_processed,
        first_divergence_tick=None,
        divergence_summary=None,
    )


def replay_game_dir(game_dir: Path, *, strict: bool = True) -> ReplayResult:
    """Replay the game directory (manifest + action log); raise ``FileNotFoundError`` naming what's missing.

    With ``strict=True`` (default) a log shorter than the manifest's ``n_ticks``
    is reported as a divergence; ``strict=False`` accepts partial trajectories
    (crashed-server audit).
    """
    # Boundary validation: a typo'd path surfaces with a greppable error.
    if not game_dir.is_dir():
        raise FileNotFoundError(f"game_dir not found: {game_dir}")

    manifest_path = game_dir / "game_manifest.json"
    log_path = game_dir / "action_log.jsonl"

    # Surface each missing file separately: the manifest is written at game start, the log per-tick.
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest at {manifest_path}")
    if not log_path.exists():
        raise FileNotFoundError(f"missing action log at {log_path}")

    result = replay(manifest_path, log_path)

    if not strict:
        # ``strict=False`` accepts whatever ticks were captured (crashed-server audit).
        return result

    # ``strict=True``: an OK result with fewer ticks than the manifest promised is a
    # truncation — upgrade to a divergence so the CLI exits non-zero.
    # itself does not detect).
    if result.ok:
        manifest = read_manifest(manifest_path)
        if result.n_ticks_replayed < manifest.n_ticks:
            return ReplayResult(
                ok=False,
                n_ticks_replayed=result.n_ticks_replayed,
                first_divergence_tick=result.n_ticks_replayed,
                divergence_summary=(
                    f"action log truncated at tick {result.n_ticks_replayed} — "
                    f"manifest expected {manifest.n_ticks}"
                ),
            )
    return result


__all__ = [
    "ActionLogRecord",
    "ReplayManifest",
    "ReplayResult",
    "read_action_log",
    "read_manifest",
    "replay",
    "replay_game_dir",
]
