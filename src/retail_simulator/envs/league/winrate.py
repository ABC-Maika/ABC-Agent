"""Head-to-head win-rate tracking and opponent-distribution samplers for league rounds.

WinRateMatrix persists per-(learner, opponent) wins/total; the
OpponentDistribution samplers (uniform, latest-N, PFSP) draw the next opponent
each round from the frozen pool.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from .pool import FrozenPolicyPool, PoolMember

# Win-rate matrix on-disk format version: detected at load time so a
# future format change surfaces as a clear ValueError.
WIN_RATE_MATRIX_VERSION: int = 1


@dataclass(frozen=True)
class WinRateEntry:
    """One cell of the win-rate matrix: head-to-head wins/total for a (learner, opponent) pair.

    Frozen value type; :meth:`WinRateMatrix.record` rebuilds entries instead of
    mutating them. Invariants (``0 <= wins <= total``) are checked at
    construction so a corrupted on-disk record fails at load time.
    """

    wins: int
    total: int

    def __post_init__(self) -> None:
        if self.wins < 0:
            raise ValueError(f"WinRateEntry: wins must be >= 0, got {self.wins}")
        if self.total < 0:
            raise ValueError(f"WinRateEntry: total must be >= 0, got {self.total}")
        if self.wins > self.total:
            raise ValueError(f"WinRateEntry: wins ({self.wins}) cannot exceed total ({self.total})")

    @property
    def rate(self) -> float:
        """Empirical win-rate; raises if ``total == 0`` (caller should check via ``total`` first)."""
        if self.total == 0:
            raise ValueError("WinRateEntry.rate: total == 0; check `total` before accessing rate")
        return self.wins / self.total


@dataclass
class WinRateMatrix:
    """Per-(learner, opponent) head-to-head win-rate tracker.

    Sparse dict keyed by ``(learner_id, opponent_id)``; recording a win adds to
    that entry only (no symmetric reverse record). Persisted atomically as JSON
    at ``<pool_dir>/win_rate_matrix.json``.
    """

    # NOT frozen: record() mutates the internal dict; the public API
    # (record/win_rate/save/load) is the surface contract.
    _entries: dict[tuple[str, str], WinRateEntry] = field(default_factory=dict)

    def record(self, learner_id: str, opponent_id: str, learner_won: bool) -> None:
        """Record one head-to-head outcome; repeated calls accumulate on the existing entry."""
        key = (learner_id, opponent_id)
        prev = self._entries.get(key, WinRateEntry(wins=0, total=0))
        self._entries[key] = WinRateEntry(
            wins=prev.wins + (1 if learner_won else 0),
            total=prev.total + 1,
        )

    def win_rate(self, learner_id: str, opponent_id: str) -> float | None:
        """Win-rate for a pair, or None if there is no record yet (cold start)."""
        key = (learner_id, opponent_id)
        entry = self._entries.get(key)
        if entry is None or entry.total == 0:
            return None
        return entry.rate

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> tuple[tuple[tuple[str, str], WinRateEntry], ...]:
        """Sorted immutable snapshot of all entries (for testing / persistence)."""
        return tuple(sorted(self._entries.items()))

    def save(self, path: Path) -> None:
        """Atomic JSON write (tmp file + ``fsync`` + ``os.replace``).

        Entries serialize as a list of records sorted by (learner_id, opponent_id)
        for diff-ability.
        """
        payload = {
            "version": WIN_RATE_MATRIX_VERSION,
            "entries": [
                {
                    "learner_id": learner_id,
                    "opponent_id": opponent_id,
                    "wins": entry.wins,
                    "total": entry.total,
                }
                for (learner_id, opponent_id), entry in self.entries()
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, sort_keys=True, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> WinRateMatrix:
        """Load the matrix JSON; missing file raises ``FileNotFoundError``, an unknown version or malformed record raises ``ValueError``."""
        if not path.is_file():
            raise FileNotFoundError(f"WinRateMatrix.load: file not found at {path}")
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError(
                f"WinRateMatrix.load: expected JSON object, got {type(payload).__name__}"
            )
        version = payload.get("version")
        if version != WIN_RATE_MATRIX_VERSION:
            raise ValueError(
                f"WinRateMatrix.load: unknown version {version!r} at {path}; "
                f"expected {WIN_RATE_MATRIX_VERSION}"
            )
        entries_raw = payload.get("entries", [])
        if not isinstance(entries_raw, list):
            raise ValueError("WinRateMatrix.load: 'entries' must be a JSON array")
        matrix = cls()
        for record in entries_raw:
            if not isinstance(record, dict):
                raise ValueError(
                    f"WinRateMatrix.load: entry must be an object, got {type(record).__name__}"
                )
            try:
                learner_id = str(record["learner_id"])
                opponent_id = str(record["opponent_id"])
                wins = int(record["wins"])
                total = int(record["total"])
            except KeyError as exc:
                raise ValueError(
                    f"WinRateMatrix.load: missing required key {exc.args[0]!r}"
                ) from exc
            matrix._entries[(learner_id, opponent_id)] = WinRateEntry(wins=wins, total=total)
        return matrix



class OpponentDistribution(Protocol):
    """Sampling strategy: pick which pool member faces the learner each episode."""

    def sample(
        self,
        rng: np.random.Generator,
        pool: FrozenPolicyPool,
        *,
        win_rate_matrix: WinRateMatrix | None = None,
        learner_id: str | None = None,
    ) -> PoolMember:  # pragma: no cover - Protocol
        ...


@dataclass(frozen=True)
class UniformOpponentDistribution:
    """Uniform sampler over the pool's current members.

    Each ``sample(rng, pool)`` returns one member chosen uniformly via
    ``rng.integers(len(pool))``; determinism is exactly the rng's. Raises
    :class:`ValueError` at sample time on an empty pool. The
    ``win_rate_matrix``/``learner_id`` kwargs are accepted and ignored.
    """

    def sample(
        self,
        rng: np.random.Generator,
        pool: FrozenPolicyPool,
        *,
        win_rate_matrix: WinRateMatrix | None = None,
        learner_id: str | None = None,
    ) -> PoolMember:
        """Draw ONE :class:`PoolMember` uniformly at random from ``pool``."""
        # win_rate_matrix/learner_id are accepted for Protocol compatibility
        # with PFSPOpponentDistribution; uniform sampling ignores them.
        del win_rate_matrix, learner_id
        if len(pool) == 0:
            raise ValueError("UniformOpponentDistribution: pool is empty; sample() is ill-defined")
        index = int(rng.integers(0, len(pool)))
        return pool.members()[index]


@dataclass(frozen=True)
class LatestNOpponentDistribution:
    """Uniform over the latest N learner-checkpoint members.

    Eligibility: ``kind == "checkpoint"`` members sorted by ``round_index``
    descending (ties broken by stable insertion order), truncated to the newest
    ``n``. With ``include_scripted=True`` the fixed-opponent members
    (``kind in ("scripted", "archetype")``) are unioned in as a diversity floor,
    de-duplicated by ``member_id`` and kept in insertion order. An empty eligible
    set raises :class:`ValueError` at sample time.
    """

    n: int
    include_scripted: bool = True

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(
                f"LatestNOpponentDistribution: n must be >= 1, got {self.n}; "
                "the trailing window must include at least one checkpoint"
            )

    def sample(
        self,
        rng: np.random.Generator,
        pool: FrozenPolicyPool,
        *,
        win_rate_matrix: WinRateMatrix | None = None,
        learner_id: str | None = None,
    ) -> PoolMember:
        """Draw one member uniformly from the eligible set (win-rate kwargs accepted and ignored)."""
        # win_rate_matrix/learner_id are accepted for Protocol compatibility;
        # latest-N sampling ignores them.
        del win_rate_matrix, learner_id
        members = pool.members()
        # Ties on round_index break by stable insertion order, so the same
        # pool yields the same eligible set across processes.
        checkpoint_indexed = [
            (insertion_index, member)
            for insertion_index, member in enumerate(members)
            if member.kind == "checkpoint"
        ]
        checkpoint_indexed.sort(key=lambda pair: (-pair[1].round_index, pair[0]))
        latest_checkpoint_ids: set[str] = {
            member.member_id for _idx, member in checkpoint_indexed[: self.n]
        }

        eligible_ids: set[str] = set(latest_checkpoint_ids)
        if self.include_scripted:
            for member in members:
                if member.kind in ("scripted", "archetype"):
                    eligible_ids.add(member.member_id)

        # Preserve pool insertion order in the eligible tuple so determinism
        # survives de-duplication.
        eligible = tuple(member for member in members if member.member_id in eligible_ids)
        if not eligible:
            raise ValueError(
                "LatestNOpponentDistribution: no eligible members; pool has no "
                "checkpoint members and include_scripted=False"
            )
        index = int(rng.integers(0, len(eligible)))
        return eligible[index]


@dataclass(frozen=True)
class PFSPOpponentDistribution:
    """Win-rate-weighted opponent sampling (alpha curve, unseen bias, uniform mix)."""

    alpha: float = 2.0
    bias_for_unseen: float = 1.0
    mix_uniform: float = 0.0

    def __post_init__(self) -> None:
        # alpha < 0 inverts the weight curve; reject at construction so a
        # typo is caught immediately, not at sample time.
        if self.alpha < 0:
            raise ValueError(f"PFSPOpponentDistribution: alpha must be >= 0, got {self.alpha}")
        # bias_for_unseen <= 0 makes an all-unseen pool's total weight 0 and
        # the sampler ill-defined; reject the whole non-positive range.
        if self.bias_for_unseen <= 0:
            raise ValueError(
                f"PFSPOpponentDistribution: bias_for_unseen must be > 0, got {self.bias_for_unseen}"
            )
        # mix_uniform is a mixing fraction: outside [0, 1] the formula
        # misbehaves, so it is rejected at construction like the checks above.
        if not 0.0 <= self.mix_uniform <= 1.0:
            raise ValueError(
                f"PFSPOpponentDistribution: mix_uniform must be in [0.0, 1.0], "
                f"got {self.mix_uniform}"
            )

    def sample(
        self,
        rng: np.random.Generator,
        pool: FrozenPolicyPool,
        *,
        win_rate_matrix: WinRateMatrix | None = None,
        learner_id: str | None = None,
    ) -> PoolMember:
        """Sample opponent via PFSP weighting; excludes the learner from its own opponent set."""
        members = pool.members()
        if not members:
            raise ValueError("PFSPOpponentDistribution: pool is empty")

        # A learner never plays itself in PFSP; with learner_id=None the
        # whole pool is eligible.
        eligible = [m for m in members if m.member_id != learner_id]
        if not eligible:
            raise ValueError(
                f"PFSPOpponentDistribution: no eligible opponents in pool of "
                f"{len(members)} (learner_id={learner_id!r} excluded; pool may "
                f"be size 1)"
            )

        # Per-member weights: the matrix is consulted only when both it and
        # learner_id are provided; otherwise everyone gets bias_for_unseen.
        weights = np.empty(len(eligible), dtype=np.float64)
        for i, m in enumerate(eligible):
            wr: float | None = None
            if win_rate_matrix is not None and learner_id is not None:
                wr = win_rate_matrix.win_rate(learner_id, m.member_id)
            if wr is None:
                # No-history fallback.
                weights[i] = self.bias_for_unseen
            else:
                # The (1 - x)^alpha curve: a fully beaten opponent
                # (win_rate=1.0) gets weight 0 - stop drilling it.
                weights[i] = (1.0 - wr) ** self.alpha

        total = float(weights.sum())
        if total <= 0.0:
            weights = np.ones(len(eligible), dtype=np.float64)
            total = float(len(eligible))

        probs = weights / total
        if self.mix_uniform > 0.0:
            probs = (1.0 - self.mix_uniform) * probs + self.mix_uniform / len(eligible)
        idx = int(rng.choice(len(eligible), p=probs))
        return eligible[idx]
