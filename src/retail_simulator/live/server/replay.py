"""Action-log persistence and end-of-game score reducers for the live server.

The per-tick JSONL :class:`_ActionLogWriter`, the atomic text write it shares
with the manifest/summary artifacts, the :class:`ReplayManifest` snapshot of
one game's inputs, and the pure reducers that turn the per-seat accumulators
into the final winner + wall-clock variance numbers.

Depends only on the seam's replay domain (``retail_simulator.live.replay``)
plus the seam outputs — :mod:`retail_simulator.live.server.game` imports
these helpers, never the reverse.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import retail_simulator
from retail_simulator.core.schema import ACTION_SCHEMA_VERSION
from retail_simulator.live.replay import ActionLogRecord, ReplayManifest

if TYPE_CHECKING:
    from retail_simulator.live.server.game import _GameServer


# ---------------------------------------------------------------------------- #
# Persistence — atomic writes + the per-tick JSONL action log
# ---------------------------------------------------------------------------- #


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically.

    A sibling ``.tmp`` + ``os.replace`` means a crash leaves either the prior
    or the new contents, never a partial file.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


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
# Manifest snapshot + end-of-game reducers
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
