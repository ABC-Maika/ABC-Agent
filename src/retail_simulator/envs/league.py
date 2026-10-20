"""Frozen-policy pool primitives + opponent distributions for multi-round self-play.

The pool (``PoolMember`` / :class:`FrozenPolicyPool` / ``make_seeded_pool``) is
the on-disk artifact the training loop grows: each round saves a learner
checkpoint and appends it as a member. Opponent distributions (uniform,
latest-N, PFSP) draw the next opponent each episode. The league wrapper wraps
over :class:`CurriculumEnvWrapper`: the outer wrapper draws an opponent, the
inner draws a scenario.

``stable_baselines3`` is imported lazily (only in the checkpoint-materialize
path), so importing this module never pulls SB3.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    Protocol,
    cast,
    runtime_checkable,
)

import numpy as np

from retail_simulator.envs.geo_archetypes import (
    GEO_ARCHETYPE_LABELS,
    geo_archetypes as _geo_archetypes,
)
from retail_simulator.envs.spike_archetypes import (
    AGGRESSOR,
    LEAN,
    PATIENT,
    calibrated_archetypes,
)

if TYPE_CHECKING:  # type-only; never pulled in at runtime import time
    from retail_simulator.core.config import CoreConfig, RegionConfig
    from retail_simulator.envs.curriculum import ScenarioDistribution

logger = logging.getLogger(__name__)

# Manifest schema version: bump only on a breaking on-disk format change.
MANIFEST_VERSION: int = 1

# Version stamped on a manifest that carries arena-seating provenance
# (the three seating keys are present); see FrozenPolicyPool.save/.load.
MANIFEST_VERSION_ARENA_SEATING: int = 2

# Version stamped on a manifest that carries a recorded reward_mode;
# see FrozenPolicyPool.record_reward_mode/.save/.load.
MANIFEST_VERSION_REWARD_MODE: int = 3

# Win-rate matrix on-disk format version: detected at load time so a
# future format change surfaces as a clear ValueError.
WIN_RATE_MATRIX_VERSION: int = 1

# The scripted archetype names a pool seed may carry: the world NPC
# allowlist minus "wandering" (an exploration substrate, not an opponent).
DEFAULT_POOL_SEED_ARCHETYPES: tuple[str, ...] = ("discounter", "premium", "balanced")

POOL_SPIKE_ARCHETYPE_LABELS: tuple[str, ...] = (LEAN, AGGRESSOR, PATIENT)

# The full archetype allowlist: the spike family above UNION the four
# geo archetypes (imported from geo_archetypes so the two cannot drift).
POOL_ARCHETYPE_LABELS: tuple[str, ...] = POOL_SPIKE_ARCHETYPE_LABELS + GEO_ARCHETYPE_LABELS

# The basic league name; the CLI's --league=<name> flag routes it onto build_league.
DEFAULT_LEAGUE_NAME: str = "phase_1_5_basic_league"

# Sentinel for scripted members: not a learner checkpoint (those carry round_index >= 0).
_SCRIPTED_ROUND_INDEX: int = -1

# PoolMember.kind allowlist: "checkpoint" is a learner snapshot; "scripted"
# names a core NPC archetype; "archetype" is a fixed calibrated posture.
PoolMemberKind = Literal["scripted", "checkpoint", "archetype"]


@dataclass(frozen=True)
class PoolMember:
    """ """

    member_id: str
    kind: PoolMemberKind
    archetype_name: str
    checkpoint_path: Path | None
    round_index: int
    policy_kind: Literal["mlp", "lstm"] = "mlp"

    def __post_init__(self) -> None:
        if not self.member_id:
            raise ValueError("PoolMember.member_id must be non-empty")
        # Validate policy_kind at the boundary: "mlp" (default) and "lstm"
        # are legal; anything else is a programming error.
        if self.policy_kind not in ("mlp", "lstm"):
            raise ValueError(
                f"PoolMember {self.member_id!r} policy_kind must be 'mlp' or 'lstm'; "
                f"got {self.policy_kind!r}"
            )
        if self.kind == "scripted":
            if self.checkpoint_path is not None:
                raise ValueError(
                    f"scripted PoolMember {self.member_id!r} must have checkpoint_path=None; "
                    f"got {self.checkpoint_path!r}"
                )
            if self.archetype_name not in DEFAULT_POOL_SEED_ARCHETYPES:
                raise ValueError(
                    f"scripted PoolMember {self.member_id!r} archetype_name "
                    f"{self.archetype_name!r} not in scripted allowlist "
                    f"{DEFAULT_POOL_SEED_ARCHETYPES}"
                )
            if self.round_index != _SCRIPTED_ROUND_INDEX:
                raise ValueError(
                    f"scripted PoolMember {self.member_id!r} must use the "
                    f"round_index={_SCRIPTED_ROUND_INDEX} sentinel; got {self.round_index}"
                )
        elif self.kind == "checkpoint":
            if self.checkpoint_path is None:
                raise ValueError(
                    f"checkpoint PoolMember {self.member_id!r} must carry a checkpoint_path"
                )
            if not self.archetype_name.startswith("frozen_policy:"):
                raise ValueError(
                    f"checkpoint PoolMember {self.member_id!r} archetype_name must start "
                    f"with 'frozen_policy:'; got {self.archetype_name!r}"
                )
            if self.round_index < 0:
                raise ValueError(
                    f"checkpoint PoolMember {self.member_id!r} round_index must be >= 0; "
                    f"got {self.round_index}"
                )
        elif self.kind == "archetype":
            if self.checkpoint_path is not None:
                raise ValueError(
                    f"archetype PoolMember {self.member_id!r} must have checkpoint_path=None; "
                    f"got {self.checkpoint_path!r}"
                )
            if self.archetype_name not in POOL_ARCHETYPE_LABELS:
                raise ValueError(
                    f"archetype PoolMember {self.member_id!r} archetype_name "
                    f"{self.archetype_name!r} not in archetype allowlist "
                    f"{POOL_ARCHETYPE_LABELS}"
                )
            if self.round_index != _SCRIPTED_ROUND_INDEX:
                raise ValueError(
                    f"archetype PoolMember {self.member_id!r} must use the "
                    f"round_index={_SCRIPTED_ROUND_INDEX} sentinel; got {self.round_index}"
                )
        else:
            raise ValueError(
                f"PoolMember {self.member_id!r} kind must be 'scripted', 'checkpoint', "
                f"or 'archetype'; got {self.kind!r}"
            )

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for the pool manifest (paths as POSIX strings)."""
        return {
            "member_id": self.member_id,
            "kind": self.kind,
            "archetype_name": self.archetype_name,
            "checkpoint_path": (
                self.checkpoint_path.as_posix() if self.checkpoint_path is not None else None
            ),
            "round_index": self.round_index,
            # policy_kind is emitted last so new fields append to the tail
            # of the serialized member dict.
            "policy_kind": self.policy_kind,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> PoolMember:
        """Deserialize from a manifest dict; unknown/missing keys raise ``ValueError``."""
        try:
            member_id = data["member_id"]
            kind = data["kind"]
            archetype_name = data["archetype_name"]
            raw_path = data["checkpoint_path"]
            round_index = data["round_index"]
        except KeyError as exc:
            raise ValueError(
                f"PoolMember manifest entry missing required key {exc.args[0]!r}; got {data!r}"
            ) from exc
        checkpoint_path = Path(raw_path) if raw_path is not None else None
        if kind not in ("scripted", "checkpoint", "archetype"):
            raise ValueError(
                f"PoolMember kind must be 'scripted', 'checkpoint', or 'archetype'; got {kind!r}"
            )
        # policy_kind is optional on read: older manifests lack it and load
        # as "mlp"; an invalid on-disk value fails in __post_init__.
        policy_kind_raw = data.get("policy_kind", "mlp")
        return cls(
            member_id=str(member_id),
            kind=kind,
            archetype_name=str(archetype_name),
            checkpoint_path=checkpoint_path,
            round_index=int(round_index),
            policy_kind=policy_kind_raw,
        )


class FrozenPolicyPool:
    """Ordered collection of :class:`PoolMember` values — the league's opponent set.

    The manifest is written atomically (tmp file + ``os.replace``) so a crash can
    never leave it partially written. ``members()`` order is insertion order.
    Optional provenance — the arena seating and reward mode the pool's checkpoints
    were trained under — is set via :meth:`record_arena_seating` /
    :meth:`record_reward_mode` and read back by :meth:`load`, so a ``--resume`` run
    sees the config its checkpoints were actually trained with. ``region_table``
    is in-memory only (never persisted): a restored pool reattaches it via
    :meth:`attach_region_table`, which verifies it against the recorded
    region-table hash.
    """

    # Predict-closure cache for materialize, keyed by member_id: holds a
    # loaded SB3 model (checkpoint) or a calibrated posture closure (archetype),
    # so a reused pool never rebuilds the same callable.
    _materialize_cache: dict[str, Any]

    # The full {label: closure} roster from ONE geo_archetypes() call, cached
    # separately from the per-member cache: built once per pool instance, so a
    # 4-posture pool does not rebuild the other postures 4 times.
    _geo_archetype_roster: dict[str, Any] | None

    def __init__(
        self,
        pool_dir: Path,
        *,
        members: tuple[PoolMember, ...] = (),
        home_regions: tuple[int, ...] | None = None,
        arena_region_table_hash: str | None = None,
        arena_lock_knobs: dict[str, float] | None = None,
        region_table: tuple[RegionConfig, ...] | None = None,
        reward_mode: str | None = None,
    ) -> None:
        """Construct an in-memory pool view rooted at ``pool_dir`` (directory not created here).

        Provenance fields (``home_regions``/``arena_region_table_hash``/
        ``arena_lock_knobs``, ``reward_mode``) default to ``None`` — a fresh pool does
        not yet know what it will train under; the training loop sets them later.
        ``region_table`` is in-memory only and never persisted.
        """
        self._pool_dir: Path = Path(pool_dir)
        # Duplicate member_id is a programming error, rejected before storing.
        seen: set[str] = set()
        for member in members:
            if member.member_id in seen:
                raise ValueError(
                    f"FrozenPolicyPool members carry duplicate member_id "
                    f"{member.member_id!r}; member_ids must be unique within a pool"
                )
            seen.add(member.member_id)
        self._members: list[PoolMember] = list(members)
        self._materialize_cache = {}
        self._geo_archetype_roster = None
        self._home_regions: tuple[int, ...] | None = home_regions
        self._arena_region_table_hash: str | None = arena_region_table_hash
        self._arena_lock_knobs: dict[str, float] | None = arena_lock_knobs
        self._region_table: tuple[RegionConfig, ...] | None = region_table
        self._reward_mode: str | None = reward_mode

    @property
    def pool_dir(self) -> Path:
        """The on-disk root for this pool's manifest + checkpoint zips."""
        return self._pool_dir

    @property
    def home_regions(self) -> tuple[int, ...] | None:
        """The home-region seating this pool's checkpoints were trained under (None if untrained)."""
        return self._home_regions

    @property
    def arena_region_table_hash(self) -> str | None:
        """The region-table fairness hash of the trained config; None exactly when :attr:`home_regions` is None."""
        return self._arena_region_table_hash

    @property
    def arena_lock_knobs(self) -> dict[str, float] | None:
        """ """
        return self._arena_lock_knobs

    def record_arena_seating(
        self,
        *,
        home_regions: tuple[int, ...] | None,
        arena_region_table_hash: str | None,
        arena_lock_knobs: dict[str, float] | None = None,
    ) -> None:
        """Record the arena seating provenance and persist immediately.

        Called once per run by the training loop — even when there is nothing to
        record — so the manifest always reflects the current seating: the three keys
        are omitted entirely for a pool with no seating and present together for one
        that has it.
        """
        self._home_regions = home_regions
        self._arena_region_table_hash = arena_region_table_hash
        self._arena_lock_knobs = arena_lock_knobs
        self.save()

    @property
    def reward_mode(self) -> str | None:
        """The reward mode this pool's checkpoints were trained under (None for the profit default or unrecorded pools)."""
        return self._reward_mode

    def record_reward_mode(self, reward_mode: str | None) -> None:
        """Record the reward-mode provenance and persist immediately.

        Called once per run — even when ``reward_mode`` is ``None`` — so a reused pool
        object never keeps a stale value. Whether a mode is "notable" (worth
        recording) is the caller's decision, not this method's.
        """
        self._reward_mode = reward_mode
        self.save()

    @property
    def region_table(self) -> tuple[RegionConfig, ...] | None:
        """The in-memory region table for materializing geo members; never persisted."""
        return self._region_table

    def attach_region_table(self, region_table: tuple[RegionConfig, ...]) -> None:
        """Attach an in-memory region table so a restored pool can materialize geo members.

        ``load`` cannot recover the table (it is not part of the manifest). When the
        pool has a recorded :attr:`arena_region_table_hash`, the attached table is
        hashed the same way and must match, or this raises ``ValueError`` — never
        materialize against a different region table than the members were trained
        under. Pools with no recorded hash skip the check.
        """
        if self._arena_region_table_hash is not None:
            from dataclasses import replace  # noqa: PLC0415

            from retail_simulator.core.config import CoreConfig  # noqa: PLC0415
            from retail_simulator.envs.arena import (  # noqa: PLC0415
                arena_region_table_hash as _arena_region_table_hash,
            )

            base = CoreConfig.default()
            probe_config = replace(base, demand=replace(base.demand, regions=region_table))
            actual_hash = _arena_region_table_hash(probe_config)
            if actual_hash != self._arena_region_table_hash:
                raise ValueError(
                    "FrozenPolicyPool.attach_region_table: the given region_table "
                    f"hashes to {actual_hash!r}, but this pool's manifest recorded "
                    f"{self._arena_region_table_hash!r} — refusing to materialize "
                    "geo archetypes against a different region table than the one "
                    "this pool's members were seeded/trained under"
                )
        self._region_table = region_table

    def members(self) -> tuple[PoolMember, ...]:
        """Immutable snapshot of the current member list (insertion order)."""
        return tuple(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def __iter__(self) -> Iterator[PoolMember]:
        return iter(self._members)

    def add(self, member: PoolMember) -> None:
        """Append ``member``; reject a duplicate ``member_id`` (programming error)."""
        for existing in self._members:
            if existing.member_id == member.member_id:
                raise ValueError(
                    f"FrozenPolicyPool already contains a member with member_id "
                    f"{member.member_id!r}; member_ids must be unique within a pool"
                )
        self._members.append(member)

    def add_checkpoint(
        self,
        model: Any,
        round_index: int,
        *,
        seed: int,
        policy_kind: Literal["mlp", "lstm"] = "mlp",
    ) -> PoolMember:
        """Save ``model`` to ``{pool_dir}/round_{round_index}.zip``, append a checkpoint member, persist the manifest atomically.

        The seed participates only in the member_id naming — a provenance handle, not
        an RNG draw. The checkpoint is saved BEFORE the manifest is rewritten, so a
        crash between leaves an orphan zip (recoverable) rather than a manifest
        pointing at a missing zip. ``policy_kind`` is the caller's word ("mlp" or
        "lstm"); it selects the closure shape at :meth:`materialize` time.
        """
        relative_path = Path(f"round_{round_index}.zip")
        # PPO.save accepts str or PathLike; pass the absolute path.
        absolute_path = self._pool_dir / relative_path
        model.save(absolute_path)
        member = PoolMember(
            member_id=f"checkpoint:round_{round_index}_seed_{seed}",
            kind="checkpoint",
            archetype_name=f"frozen_policy:round_{round_index}_seed_{seed}",
            checkpoint_path=relative_path,
            round_index=round_index,
            policy_kind=policy_kind,
        )
        self.add(member)
        self.save()
        return member

    def save(self, pool_dir: Path | None = None) -> None:
        """Write ``manifest.json`` atomically (tmp file in the same dir + ``os.replace``).

        A crash mid-write leaves the old manifest intact. Seating provenance
        (``home_regions``/``arena_region_table_hash``/``arena_lock_knobs``) is emitted
        only when there is a real seating to record, stamping
        :data:`MANIFEST_VERSION_ARENA_SEATING` exactly then; ``reward_mode`` is emitted
        independently the same way, stamping :data:`MANIFEST_VERSION_REWARD_MODE` when
        present (which takes precedence).
        """
        target_dir = Path(pool_dir) if pool_dir is not None else self._pool_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "manifest.json"
        payload: dict[str, Any] = {
            "version": MANIFEST_VERSION,
            "members": [m.to_json_dict() for m in self._members],
        }
        if self._home_regions is not None:
            # Tuples become lists to match PoolMember.to_json_dict (JSON has
            # no tuple type); json.dump's sort_keys=True keeps key order stable.
            payload["version"] = MANIFEST_VERSION_ARENA_SEATING
            payload["home_regions"] = list(self._home_regions)
            payload["arena_region_table_hash"] = self._arena_region_table_hash
            payload["arena_lock_knobs"] = self._arena_lock_knobs
        if self._reward_mode is not None:
            payload["version"] = MANIFEST_VERSION_REWARD_MODE
            payload["reward_mode"] = self._reward_mode
        # NamedTemporaryFile with delete=False so we keep the file handle until
        # close, then os.replace it onto the target. delete=True races on Windows.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target_dir),
            prefix=".manifest-",
            suffix=".json.tmp",
            delete=False,
        ) as tmp:
            json.dump(payload, tmp, indent=2, sort_keys=True)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_name = tmp.name
        os.replace(tmp_name, target)

    @classmethod
    def load(cls, pool_dir: Path) -> FrozenPolicyPool:
        """Restore a pool from ``pool_dir / "manifest.json"``.

        Raises ``FileNotFoundError`` if the manifest is missing and ``ValueError`` on
        an unknown version (forward-compat guard). Accepts all three manifest
        versions; the seating and reward-mode keys are optional at every version.
        """
        path = Path(pool_dir) / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(
                f"FrozenPolicyPool.load: no manifest at {path}; "
                f"call FrozenPolicyPool({pool_dir!r}) or make_seeded_pool({pool_dir!r}) first"
            )
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        version = payload.get("version")
        if version not in (
            MANIFEST_VERSION,
            MANIFEST_VERSION_ARENA_SEATING,
            MANIFEST_VERSION_REWARD_MODE,
        ):
            raise ValueError(
                f"FrozenPolicyPool.load: manifest at {path} has unknown version {version!r}; "
                f"this code supports version {MANIFEST_VERSION}, "
                f"{MANIFEST_VERSION_ARENA_SEATING}, or {MANIFEST_VERSION_REWARD_MODE}"
            )
        members = tuple(PoolMember.from_json_dict(item) for item in payload.get("members", []))
        raw_home_regions = payload.get("home_regions")
        home_regions = tuple(raw_home_regions) if raw_home_regions is not None else None
        return cls(
            Path(pool_dir),
            members=members,
            home_regions=home_regions,
            arena_region_table_hash=payload.get("arena_region_table_hash"),
            arena_lock_knobs=payload.get("arena_lock_knobs"),
            reward_mode=payload.get("reward_mode"),
        )

    def materialize(self, member: PoolMember, env: Any) -> Callable[..., Any]:
        """ """
        del env  # forward-compat; the current implementation needs no env reference

        if member.kind == "scripted":

            def _scripted_placeholder(_obs: Any, **_kwargs: Any) -> Any:
                # **_kwargs absorbs the optional sample() kwargs (episode_start)
                # for signature uniformity; this branch ignores them.
                raise NotImplementedError(
                    f"scripted PoolMember {member.member_id!r} is seated via the env's "
                    f"npc_archetypes (the core's _make_npc_policy seam), NOT via this "
                    f"callable; the league wrapper (Slice A T-A3) handles the seating"
                )

            return _scripted_placeholder

        if member.kind == "archetype":
            # Archetype branch: no SB3, no disk zip — the closure comes from
            # the calibrated or geo roster and is cached like a checkpoint.
            if member.member_id not in self._materialize_cache:
                if member.archetype_name in GEO_ARCHETYPE_LABELS:
                    if self._region_table is None:
                        raise ValueError(
                            f"archetype PoolMember {member.member_id!r} is a geo "
                            "archetype but this pool has no region_table — pass "
                            "one to make_seeded_pool(..., region_table=...) when "
                            "seeding it, or call attach_region_table(...) after "
                            "FrozenPolicyPool.load(...) (a restored pool never "
                            "recovers its region table from the manifest)"
                        )
                    # Build the full 4-posture roster once per pool instance
                    # (see _geo_archetype_roster), never once per member.
                    if self._geo_archetype_roster is None:
                        self._geo_archetype_roster = _geo_archetypes(
                            len(self._region_table), self._region_table
                        )
                    self._materialize_cache[member.member_id] = self._geo_archetype_roster[
                        member.archetype_name
                    ]
                else:
                    self._materialize_cache[member.member_id] = calibrated_archetypes()[
                        member.archetype_name
                    ]
            return self._materialize_cache[member.member_id]

        # Checkpoint branch — lazy SB3 imports keep module load cheap.
        if member.member_id not in self._materialize_cache:
            if member.checkpoint_path is None:
                # Defensive: __post_init__ guarantees a checkpoint member has
                # a path, but typing infers Path | None; fail loudly if not.
                raise RuntimeError(
                    f"checkpoint PoolMember {member.member_id!r} has no checkpoint_path; "
                    f"the dataclass invariant was bypassed"
                )
            if member.policy_kind == "lstm":
                # Lazy sb3-contrib import: importing this module without the
                # [recurrent] extra still succeeds; the ImportError names the
                # extra and surfaces only when an LSTM checkpoint is materialized.
                try:
                    from sb3_contrib import RecurrentPPO  # noqa: PLC0415 - lazy by design
                except ImportError as exc:  # pragma: no cover - exercised when extra missing
                    raise ImportError(
                        f"materializing PoolMember {member.member_id!r} with "
                        f"policy_kind='lstm' requires the [recurrent] extra; "
                        f"install with: pip install 'retail-simulator[recurrent]' "
                        f"(or: pip install sb3-contrib>=2.3,<3.0)"
                    ) from exc
                self._materialize_cache[member.member_id] = RecurrentPPO.load(
                    self._pool_dir / member.checkpoint_path
                )
            else:
                from stable_baselines3 import PPO  # noqa: PLC0415 - lazy by design

                self._materialize_cache[member.member_id] = PPO.load(
                    self._pool_dir / member.checkpoint_path
                )
        loaded_model = self._materialize_cache[member.member_id]

        if member.policy_kind == "mlp":

            def _checkpoint_predict_mlp(obs: Any, **_kwargs: Any) -> Any:
                # **_kwargs absorbs the episode_start kwarg; MLP has no
                # recurrent state, so it is accepted and ignored.
                action, _state = loaded_model.predict(obs, deterministic=True)
                return action

            return _checkpoint_predict_mlp

        # LSTM branch — the closure carries lstm_states across calls and
        # resets them when episode_start=True (closure-local 1-element list).
        # SB3 RecurrentPPO API:
        #
        #     predict(obs, state, episode_start, deterministic=True)
        #         -> (action, new_state)
        #
        # episode_start uses shape (1,): the seam evaluates one env at a time.
        state_cell: list[Any] = [None]

        def _checkpoint_predict_lstm(obs: Any, *, episode_start: bool = False) -> Any:
            if episode_start:
                # SB3: state=None + episode_starts=True means "fresh hidden
                # state". Resetting before the predict call keeps the carried
                # cell consistent even if SB3's reset semantics change.
                state_cell[0] = None
            episode_starts_arr = np.array([episode_start], dtype=bool)
            action, new_state = loaded_model.predict(
                obs,
                state=state_cell[0],
                episode_start=episode_starts_arr,
                deterministic=True,
            )
            state_cell[0] = new_state
            return action

        return _checkpoint_predict_lstm


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


def make_seeded_pool(
    pool_dir: Path,
    *,
    archetypes: tuple[str, ...] = DEFAULT_POOL_SEED_ARCHETYPES,
    spike_archetypes: tuple[str, ...] = (),
    geo_archetypes: tuple[str, ...] = (),
    region_table: tuple[RegionConfig, ...] | None = None,
) -> FrozenPolicyPool:
    """ """
    for name in archetypes:
        if name not in DEFAULT_POOL_SEED_ARCHETYPES:
            raise ValueError(
                f"make_seeded_pool: archetype {name!r} not in scripted allowlist "
                f"{DEFAULT_POOL_SEED_ARCHETYPES}"
            )
    for name in spike_archetypes:
        if name not in POOL_SPIKE_ARCHETYPE_LABELS:
            raise ValueError(
                f"make_seeded_pool: spike archetype {name!r} not in allowlist "
                f"{POOL_SPIKE_ARCHETYPE_LABELS}"
            )
    for name in geo_archetypes:
        if name not in GEO_ARCHETYPE_LABELS:
            raise ValueError(
                f"make_seeded_pool: geo archetype {name!r} not in allowlist "
                f"{GEO_ARCHETYPE_LABELS}"
            )
    if geo_archetypes and region_table is None:
        raise ValueError(
            f"make_seeded_pool: geo archetypes {geo_archetypes} were given but no "
            "region_table — a geo posture's expansion rule needs to know the world "
            "it plays in (geo_archetypes.geo_archetypes' own region_table argument); "
            "pass the effective config's demand.regions"
        )
    directory = Path(pool_dir)
    directory.mkdir(parents=True, exist_ok=True)
    scripted_members = tuple(
        PoolMember(
            member_id=f"scripted:{name}",
            kind="scripted",
            archetype_name=name,
            checkpoint_path=None,
            round_index=_SCRIPTED_ROUND_INDEX,
        )
        for name in archetypes
    )
    spike_archetype_members = tuple(
        PoolMember(
            member_id=f"archetype:{name}",
            kind="archetype",
            archetype_name=name,
            checkpoint_path=None,
            round_index=_SCRIPTED_ROUND_INDEX,
        )
        for name in spike_archetypes
    )
    geo_archetype_members = tuple(
        PoolMember(
            member_id=f"archetype:{name}",
            kind="archetype",
            archetype_name=name,
            checkpoint_path=None,
            round_index=_SCRIPTED_ROUND_INDEX,
        )
        for name in geo_archetypes
    )
    pool = FrozenPolicyPool(
        directory,
        members=scripted_members + spike_archetype_members + geo_archetype_members,
        region_table=region_table,
    )
    pool.save()
    return pool


# --- Opponent distribution protocol + concrete shapes ----------------------


@runtime_checkable
class OpponentDistribution(Protocol):
    """ """

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
    """ """

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


# --- Named-league registry ---------------------------------------------------

# Name -> builder registry: the CLI's --league=<name> flag looks up the
# builder and threads the returned pool into run_league_training. New named
# leagues register here.
KNOWN_LEAGUES: dict[str, Callable[[Path], FrozenPolicyPool]] = {
    DEFAULT_LEAGUE_NAME: lambda pool_dir: make_seeded_pool(pool_dir),
}


def known_leagues() -> tuple[str, ...]:
    """The names the CLI's ``--league=<name>`` flag accepts (sorted for stability)."""
    return tuple(sorted(KNOWN_LEAGUES))


def build_league(name: str, pool_dir: Path) -> FrozenPolicyPool:
    """Look up a named league builder and call it with ``pool_dir``.

    Unknown names raise :class:`ValueError` listing the known names — boundary
    validation at the CLI surface, never a silent fallback.
    """
    if name not in KNOWN_LEAGUES:
        raise ValueError(f"unknown league {name!r}; known leagues: {list(known_leagues())}")
    builder = KNOWN_LEAGUES[name]
    return builder(Path(pool_dir))


# --- LeagueEnvWrapper ------------------------------------------------------

# PettingZoo's ParallelEnv is the base class SuperSuit's bridge requires via
# an isinstance check (it does not duck-type); inheriting it makes the wrapper
# a first-class PettingZoo parallel env for the bridge path.
from pettingzoo import ParallelEnv  # noqa: E402 - kept near the wrapper for locality


class LeagueEnvWrapper(ParallelEnv):  # type: ignore[misc, unused-ignore]
    """ """

    metadata: dict[str, Any] = {"name": "retail_sim_league_parallel_v0", "render_modes": []}

    def __init__(
        self,
        wrapped_env_factory: Callable[[tuple[str, ...]], Any],
        pool: FrozenPolicyPool,
        distribution: OpponentDistribution,
        *,
        league_seed: int = 0,
        league_mode: Literal["self_play", "single_learner"] = "self_play",
        single_learner_env_factory: Callable[[int, tuple[str, ...]], Any] | None = None,
        max_episode_steps: int | None = None,
        episode_seed_base: int | None = None,
    ) -> None:
        """ """
        if league_mode == "single_learner" and single_learner_env_factory is None:
            raise ValueError(
                "single_learner mode requires single_learner_env_factory: "
                "Callable[[int, tuple[str, ...]], Any]; got None"
            )
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError(
                f"max_episode_steps must be None or an int >= 1, got {max_episode_steps}"
            )

        self._wrapped_env_factory = wrapped_env_factory
        self._single_learner_env_factory = single_learner_env_factory
        self._pool = pool
        self._distribution = distribution
        self._league_seed = league_seed
        self._league_mode: Literal["self_play", "single_learner"] = league_mode
        self._max_episode_steps = max_episode_steps
        self._ticks_since_reset: int = 0
        self._episode_seed_seq: np.random.SeedSequence | None = (
            np.random.SeedSequence(episode_seed_base) if episode_seed_base is not None else None
        )
        # The carried league Generator: built once from league_seed, never
        # re-seeded. Only reset() consumes it (the three-RNG isolation
        # contract); step() and the __init__ snapshot draw do not.
        self._league_rng: np.random.Generator = np.random.default_rng(league_seed)

        # Build an initial env from the first opponent so observation_space /
        # action_space are queryable before the first reset. The initial draw
        # uses a snapshot of league_rng, so the first reset() draws fresh.
        snapshot = self._league_rng.bit_generator.state
        initial_member = self._distribution.sample(self._league_rng, self._pool)
        self._league_rng.bit_generator.state = snapshot

        # State carried across step() under single_learner + checkpoint:
        # the frozen predict closure and the hidden seat-1 observation.
        # Both are None in self_play mode or with a scripted opponent (the
        # env's NPC mechanism seats that; no in-process injection needed).
        self._frozen_predict: Callable[..., Any] | None = None
        self._frozen_seat_index: int | None = None
        self._last_obs_seat_1: Any = None
        self._first_step_after_reset: bool = True
        # The hidden seat's PettingZoo id: the inner env builds with two
        # learners named retailer_0 / retailer_1.
        self._hidden_agent_id: str = "retailer_1"

        if self._league_mode == "self_play":
            if initial_member.kind != "scripted":
                raise NotImplementedError(
                    "self_play LeagueEnvWrapper only seats scripted-archetype pool members; "
                    "checkpoint seating requires league_mode='single_learner'. "
                    f"Got member kind={initial_member.kind!r}."
                )
            self._env: Any = self._wrapped_env_factory((initial_member.archetype_name,))
            # The PettingZoo agent interface delegates to the wrapped env
            # under self_play, mirroring CurriculumEnvWrapper.
            self.possible_agents: list[str] = list(self._env.possible_agents)
            self.agents: list[str] = list(self._env.agents)
        else:
            # single_learner: scripted member -> 1 learner + NPC seat;
            # checkpoint/archetype -> 2 learners, the wrapper hides seat 1 and
            # injects its action at step time. Either way the exposed agent is
            # retailer_0 only.
            assert self._single_learner_env_factory is not None  # checked above
            self._env = self._build_single_learner_env(initial_member)
            self.possible_agents = ["retailer_0"]
            self.agents = ["retailer_0"]
            if initial_member.kind in ("checkpoint", "archetype"):
                self._frozen_predict = self._pool.materialize(initial_member, self._env)
                self._frozen_seat_index = 1

        # SuperSuit's MarkovVectorEnv reads par_env.unwrapped.render_mode;
        # expose a (None) render_mode here so the bridge sees a defined
        # attribute.
        self.render_mode: Any = getattr(self._env, "render_mode", None)
        # Track the last sampled member (a DX hook for ``info`` / diagnostics).
        self._last_member: PoolMember = initial_member

    def _build_single_learner_env(self, member: PoolMember) -> Any:
        """Construct the inner env for ``single_learner`` mode for ``member``.

        Scripted members get one learner + an NPC seat; every other member gets two
        learners and the wrapper hides seat 1 (its action is injected at step time
        from :meth:`FrozenPolicyPool.materialize`). Both branches go through the same
        factory seam.
        """
        assert self._single_learner_env_factory is not None
        if member.kind == "scripted":
            return self._single_learner_env_factory(1, (member.archetype_name,))
        return self._single_learner_env_factory(2, ())

    @property
    def unwrapped(self) -> "LeagueEnvWrapper":
        """Return self — we are the outermost wrapper.

        SuperSuit reads ``par_env.unwrapped.render_mode``; returning the inner env
        would leak a reference that is rebuilt every ``reset()``, so
        :attr:`render_mode` is exposed on the wrapper instead.
        """
        return self

    @property
    def league_rng(self) -> np.random.Generator:
        """ """
        return self._league_rng

    @property
    def last_member(self) -> PoolMember:
        """The most recently sampled :class:`PoolMember` (set by the last ``reset()``)."""
        return self._last_member

    def observation_space(self, agent: str) -> Any:
        """The homogeneous observation Box (identical across opponents).

        Under ``single_learner`` mode only ``retailer_0``'s space is exposed (the
        hidden seat is not a PettingZoo agent).
        """
        return self._env.observation_space(agent)

    def action_space(self, agent: str) -> Any:
        """The homogeneous action Box (schema v10 — UNCHANGED across opponents)."""
        return self._env.action_space(agent)

    @property
    def observation_spaces(self) -> dict[str, Any]:
        """The per-agent observation Box dict (PettingZoo / SuperSuit attribute surface).

        Under ``single_learner`` mode returns ONLY ``retailer_0``'s entry — the
        hidden seat is NOT a PettingZoo agent.
        """
        inner_spaces = cast(dict[str, Any], self._env.observation_spaces)
        if self._league_mode == "single_learner":
            return {"retailer_0": inner_spaces["retailer_0"]}
        return inner_spaces

    @property
    def action_spaces(self) -> dict[str, Any]:
        """The per-agent action Box dict (PettingZoo / SuperSuit attribute surface).

        Under ``single_learner`` mode returns ONLY ``retailer_0``'s entry — the
        hidden seat is NOT a PettingZoo agent.
        """
        inner_spaces = cast(dict[str, Any], self._env.action_spaces)
        if self._league_mode == "single_learner":
            return {"retailer_0": inner_spaces["retailer_0"]}
        return inner_spaces

    def _resolve_episode_seed(self, seed: int | None) -> int | None:
        """ """
        if seed is not None or self._episode_seed_seq is None or self._max_episode_steps is None:
            return seed
        child = self._episode_seed_seq.spawn(1)[0]
        return int(child.generate_state(1, dtype=np.uint32)[0])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Draw the next opponent, rebuild the env, and return its ``(observations, infos)``.

        The league draw happens here via the carried ``league_rng`` (only ``reset``
        consumes it). The ``seed`` argument seeds the new env's core PCG64 only —
        ``league_rng`` is not advanced by it. With ``episode_seed_base`` set, an
        incoming ``seed=None`` (what the bridge's auto-reset passes) is replaced by a
        fresh per-episode seed via :meth:`_resolve_episode_seed` so episodes do not
        replay the same world; an explicit seed is never touched.

        Under ``single_learner`` mode the sampled member is materialized, the hidden
        seat's initial observation is captured, and only ``retailer_0``'s slice is
        returned.
        """
        member = self._distribution.sample(self._league_rng, self._pool)
        self._last_member = member
        # Close the prior env (frees its World resources); rebuild from the
        # factory. PettingZoo parallel envs are cheap to construct.
        self._env.close()
        self._first_step_after_reset = True
        self._ticks_since_reset = 0
        effective_seed = self._resolve_episode_seed(seed)

        if self._league_mode == "self_play":
            if member.kind != "scripted":
                raise NotImplementedError(
                    "self_play LeagueEnvWrapper only seats scripted-archetype pool members; "
                    "checkpoint seating requires league_mode='single_learner'. "
                    f"Got member kind={member.kind!r}."
                )
            self._env = self._wrapped_env_factory((member.archetype_name,))
            self.possible_agents = list(self._env.possible_agents)
            self.agents = list(self._env.agents)
            return cast(
                tuple[dict[str, Any], dict[str, Any]],
                self._env.reset(seed=effective_seed, options=options),
            )

        # single_learner: build the inner env for member.kind, then cache the
        # frozen predict closure so step() can inject seat-1's action.
        self._env = self._build_single_learner_env(member)
        if member.kind in ("checkpoint", "archetype"):
            self._frozen_predict = self._pool.materialize(member, self._env)
            self._frozen_seat_index = 1
        else:
            self._frozen_predict = None
            self._frozen_seat_index = None
        self.possible_agents = ["retailer_0"]
        self.agents = ["retailer_0"]

        inner_obs, inner_infos = cast(
            tuple[dict[str, Any], dict[str, Any]],
            self._env.reset(seed=effective_seed, options=options),
        )
        # Capture the hidden seat's initial observation for the next step's
        # predict call; the scripted branch leaves this None.
        self._last_obs_seat_1 = inner_obs.get(self._hidden_agent_id)
        observations = {"retailer_0": inner_obs["retailer_0"]}
        infos = {"retailer_0": inner_infos.get("retailer_0", {})}
        return observations, infos

    def _apply_episode_truncation(self, trunc: dict[str, Any], infos: dict[str, Any]) -> None:
        """ """
        if self._max_episode_steps is None:
            return
        if self._ticks_since_reset < self._max_episode_steps:
            return
        for agent in trunc:
            trunc[agent] = True
            infos.setdefault(agent, {})["TimeLimit.truncated"] = True

    def step(self, actions: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
        """ """
        self._ticks_since_reset += 1
        if self._league_mode == "self_play":
            observations, rewards, term, trunc, infos = self._env.step(actions)
            self.agents = list(self._env.agents)
            self._apply_episode_truncation(trunc, infos)
            return observations, rewards, term, trunc, infos

        # single_learner mode.
        if self._frozen_predict is None:
            # Scripted opponent: the NPC fills its own seat, so the joint
            # action is just retailer_0's.
            joint = {"retailer_0": actions["retailer_0"]}
        else:
            seat_1_action = self._frozen_predict(
                self._last_obs_seat_1, episode_start=self._first_step_after_reset
            )
            joint = {
                "retailer_0": actions["retailer_0"],
                self._hidden_agent_id: seat_1_action,
            }

        inner_obs, inner_rewards, inner_term, inner_trunc, inner_infos = self._env.step(joint)
        # Any successful step() clears the first-step flag so the next step
        # gets episode_start=False; reset() re-arms it. Flipped AFTER the
        # inner step() so a step() that raises does not turn off the LSTM
        # reset mid-episode.
        self._first_step_after_reset = False
        # Update the cached hidden-seat observation for the next step.
        # .get(): the scripted branch (n_learning_agents=1) has no retailer_1
        # key.
        self._last_obs_seat_1 = inner_obs.get(self._hidden_agent_id)
        # Hide retailer_1: agents is the inner env's agents intersected with
        # the exposed single-learner surface.
        self.agents = ["retailer_0"] if "retailer_0" in self._env.agents else []
        observations = {"retailer_0": inner_obs["retailer_0"]}
        rewards = {"retailer_0": inner_rewards["retailer_0"]}
        term = {"retailer_0": inner_term["retailer_0"]}
        trunc = {"retailer_0": inner_trunc["retailer_0"]}
        infos = {"retailer_0": inner_infos.get("retailer_0", {})}
        self._apply_episode_truncation(trunc, infos)
        return observations, rewards, term, trunc, infos

    def close(self) -> None:
        """Close the wrapped env (the league carries no resources of its own)."""
        self._env.close()


# --- Multi-round league training loop ---------------------------------------


@dataclass(frozen=True)
class LeagueTrainingResult:
    """ """

    pool_dir: Path
    rounds_completed: int
    checkpoints: tuple[PoolMember, ...]
    learning_curves: tuple[tuple[tuple[int, float], ...], ...]
    seeds: tuple[int, int, int]
    train_steps_per_round: int
    win_rate_matrix_snapshot: WinRateMatrix | None = None
    max_episode_steps: int | None = None


def _single_learner_wrapped_factory_guard(_archetypes: tuple[str, ...]) -> Any:
    """Raise if called; the factory slot under single_learner mode must stay unused.

    A stub that silently built an env would mask a refactor that accidentally
    routes a self_play call under single_learner. Module scope so mypy --strict
    has a well-typed reference.
    """
    raise AssertionError(
        "wrapped_env_factory must not be invoked under league_mode='single_learner'; "
        "single_learner_env_factory is the construction seam in that mode"
    )


@dataclass(frozen=True)
class _MatrixAwareDistribution:
    """Per-round proxy that injects the win-rate matrix + learner_id into ``sample``.

    The wrapper's ``sample(rng, pool)`` call site does not pass the matrix kwargs;
    the training loop wraps the caller's distribution with this proxy each round
    so they are forwarded transparently. Uniform/latest-N ignore them; PFSP
    consumes them. Not exported.
    """

    inner: OpponentDistribution
    win_rate_matrix: WinRateMatrix
    learner_id: str | None

    def sample(
        self,
        rng: np.random.Generator,
        pool: FrozenPolicyPool,
        *,
        win_rate_matrix: WinRateMatrix | None = None,
        learner_id: str | None = None,
    ) -> PoolMember:
        # The round loop is the authoritative source for the matrix kwargs:
        # override any caller-supplied values so they can't be elided.
        del win_rate_matrix, learner_id
        return self.inner.sample(
            rng,
            pool,
            win_rate_matrix=self.win_rate_matrix,
            learner_id=self.learner_id,
        )


def _load_warm_start_parameters(
    model: Any, *, source_checkpoint: Path, policy_kind: Literal["mlp", "lstm"]
) -> None:
    """ """
    if policy_kind == "lstm":
        # By the time warm_start gets here the model was already built as
        # LSTM (so sb3-contrib is present); this import is guaranteed to
        # succeed.
        from sb3_contrib import RecurrentPPO  # noqa: PLC0415 - lazy by design

        source_model: Any = RecurrentPPO.load(source_checkpoint, device=model.device)
    else:
        from stable_baselines3 import PPO  # noqa: PLC0415 - lazy by design

        source_model = PPO.load(source_checkpoint, device=model.device)
    model.policy.load_state_dict(source_model.policy.state_dict())


def run_league_training(
    initial_pool: FrozenPolicyPool,
    n_rounds: int,
    distribution_factory: Callable[[FrozenPolicyPool], OpponentDistribution],
    train_steps_per_round: int,
    seeds: tuple[int, int, int],
    *,
    scenario_distribution: ScenarioDistribution | None = None,
    config: CoreConfig | None = None,
    n_learning_agents: int = 1,
    pfsp_alpha: float = 2.0,
    round_end_eval_episodes: int = 5,
    distribution: OpponentDistribution | None = None,
    policy_kind: Literal["mlp", "lstm"] = "mlp",
    ent_coef: float = 0.0,
    gae_lambda: float = 0.95,
    home_regions: tuple[int, ...] | None = None,
    mask_expansion: bool = False,
    reward_mode: str | None = None,
    max_episode_steps: int | None = None,
    warm_start: bool = False,
) -> LeagueTrainingResult:
    """ """
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")

    # Lazy heavy imports keep module import cheap.
    import supersuit as ss

    from retail_simulator.envs.parallel_env import RetailParallelEnv
    from retail_simulator.envs.curriculum import CurriculumEnvWrapper
    from retail_simulator.envs.parallel_gate import (
        _make_curve_callback,
        _make_shared_ppo,
        competition_config,
    )

    if scenario_distribution is not None and home_regions is not None:
        raise ValueError(
            "run_league_training: scenario_distribution and home_regions cannot "
            "both be given — CurriculumEnvWrapper has no seating seam of its own"
        )
    if scenario_distribution is not None and mask_expansion:
        raise ValueError(
            "run_league_training: scenario_distribution and mask_expansion cannot "
            "both be given — CurriculumEnvWrapper has no masking seam of its own"
        )

    league_seed, curriculum_seed, core_seed = seeds
    # The effective config, computed once: reused by both the single-learner
    # env factory and the provenance hash recorded on the pool, so the two
    # can't silently diverge.
    effective_config = config if config is not None else competition_config()

    master_seq = np.random.SeedSequence(league_seed)
    round_seqs = master_seq.spawn(n_rounds)
    per_round_league_seeds: tuple[int, ...] = tuple(
        int(round_seqs[k].generate_state(1, dtype=np.uint32)[0]) for k in range(n_rounds)
    )

    # Load (or initialize) the persistent win-rate matrix: a fresh pool
    # yields an empty matrix, a resumed run restores the cumulative record
    # from <pool_dir>/win_rate_matrix.json.
    win_rate_matrix_path = initial_pool.pool_dir / "win_rate_matrix.json"
    try:
        win_rate_matrix = WinRateMatrix.load(win_rate_matrix_path)
    except FileNotFoundError:
        win_rate_matrix = WinRateMatrix()

    # Record the seating provenance on the pool once, before round 0,
    # always (even when nothing is to record) - see
    # FrozenPolicyPool.record_arena_seating. The region-table hash alone
    # cannot tell arena configs apart (same regions, different economics
    # knobs), so the distinguishing knobs are recorded alongside: the three
    # the lock fixes (overhead_reference_stores/overhead_exponent/
    # provided_capacity_per_store), plus credit_limit/overdraft_rate (which
    # distinguish "arena" from "arena-balanced" and are never touched by a
    # lock amendment), plus writeoff_frac_max for the same future-proofing
    # reason. They are read straight off the effective config.
    if home_regions is not None:
        from retail_simulator.envs.arena import (
            arena_region_table_hash as _arena_region_table_hash,
        )

        region_table_hash = _arena_region_table_hash(effective_config)
        arena_lock_knobs: dict[str, float] | None = {
            "overhead_reference_stores": effective_config.wage.overhead_reference_stores,
            "overhead_exponent": effective_config.wage.overhead_exponent,
            "provided_capacity_per_store": effective_config.warehouse.provided_capacity_per_store,
            "credit_limit": effective_config.wage.credit_limit,
            "overdraft_rate": effective_config.wage.overdraft_rate,
            "writeoff_frac_max": effective_config.wage.writeoff_frac_max,
        }
    else:
        region_table_hash = None
        arena_lock_knobs = None
    initial_pool.record_arena_seating(
        home_regions=home_regions,
        arena_region_table_hash=region_table_hash,
        arena_lock_knobs=arena_lock_knobs,
    )
    # Record the reward mode unconditionally (even when None) - see
    # FrozenPolicyPool.record_reward_mode for why "notable" is the caller's
    # decision.
    initial_pool.record_reward_mode(reward_mode)

    def _single_learner_factory(n_learn: int, archetypes: tuple[str, ...]) -> Any:
        """ """
        if scenario_distribution is not None:
            return CurriculumEnvWrapper(
                scenario_distribution,
                n_learning_agents=n_learn,
                npc_archetypes=archetypes,
                curriculum_seed=curriculum_seed,
                core_seed=core_seed,
            )
        return RetailParallelEnv(
            effective_config,
            n_learning_agents=n_learn,
            npc_archetypes=archetypes,
            seed=core_seed,
            home_regions=home_regions,
            # Seat 0 is always the trainee under single_learner mode (never
            # the frozen opponent).
            mask_expansion_seats=(0,) if mask_expansion else (),
        )

    def _build_bridged_vec(wrapper: LeagueEnvWrapper) -> Any:
        """Run the SuperSuit bridge over an existing wrapper (mirrors ``build_parallel_vec_env``).

        Bridges the already-constructed :class:`LeagueEnvWrapper` so its ``league_rng``
        state and the round-specific distribution are preserved. The call shape is
        verbatim from :func:`build_parallel_vec_env`.
        """
        if not hasattr(wrapper, "render_mode") or wrapper.render_mode is None:
            wrapper.render_mode = None
        vec_pz = ss.pettingzoo_env_to_vec_env_v1(wrapper)
        return ss.concat_vec_envs_v1(
            vec_pz, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3"
        )

    learning_curves: list[tuple[tuple[int, float], ...]] = []

    # ---- Round 0: build the random-init snapshot WITHOUT calling .learn
    #
    # The throw-away wrapper exists only to satisfy PPO's vec-env requirement;
    # it is closed right after PPO is constructed. Sampling the first scripted
    # member directly keeps round 0 deterministic and independent of the
    # caller's distribution.
    initial_members = initial_pool.members()
    if not initial_members:
        raise ValueError(
            "initial_pool must contain at least one member to host the round-0 "
            "PPO constructor; use make_seeded_pool(...) to seed the pool"
        )
    first_scripted: PoolMember | None = next(
        (m for m in initial_members if m.kind == "scripted"), None
    )
    if first_scripted is None:
        raise ValueError(
            "initial_pool must contain at least one scripted member to host the "
            "round-0 PPO constructor; the F5-A seeded pool provides this"
        )

    # A throw-away distribution that always returns the first scripted
    # member: the league_rng is never advanced beyond the wrapper __init__
    # snapshot draw, and the wrapper's reset() never fires (no .learn()).
    @dataclass(frozen=True)
    class _FixedMemberDistribution:
        member: PoolMember

        def sample(
            self,
            _rng: np.random.Generator,
            _pool: FrozenPolicyPool,
            *,
            win_rate_matrix: WinRateMatrix | None = None,
            learner_id: str | None = None,
        ) -> PoolMember:
            # Protocol-extension kwargs accepted for compatibility; the
            # fixed-member shim ignores them (round 0 has no win-rate data).
            del win_rate_matrix, learner_id
            return self.member

    round_zero_distribution: OpponentDistribution = _FixedMemberDistribution(first_scripted)
    round_zero_wrapper = LeagueEnvWrapper(
        wrapped_env_factory=_single_learner_wrapped_factory_guard,
        pool=initial_pool,
        distribution=round_zero_distribution,
        # Per-round spawn-derived seed (round 0).
        league_seed=per_round_league_seeds[0],
        league_mode="single_learner",
        single_learner_env_factory=_single_learner_factory,
        max_episode_steps=max_episode_steps,
        episode_seed_base=core_seed,
    )
    try:
        vec_throw = _build_bridged_vec(round_zero_wrapper)
        try:
            model_zero = _make_shared_ppo(
                vec_throw,
                seed=core_seed,
                policy_kind=policy_kind,
                ent_coef=ent_coef,
                gae_lambda=gae_lambda,
            )
            initial_pool.add_checkpoint(
                model_zero,
                round_index=0,
                seed=core_seed,
                policy_kind=policy_kind,
            )
            learning_curves.append(())
        finally:
            vec_throw.close()
    finally:
        round_zero_wrapper.close()

    # ---- Rounds 1..n_rounds-1: train then snapshot ------------------------
    # ``pfsp_alpha`` is informational for callers passing ``distribution``
    # directly (the PFSP instance carries its own alpha).
    del pfsp_alpha
    for round_index in range(1, n_rounds):
        # Distribution selection: an explicit caller distribution is used
        # as-is; otherwise rebuild via the factory at the top of each round so
        # the pool growth from the prior round is visible to this round's
        # opponent draws.
        base_distribution: OpponentDistribution = (
            distribution if distribution is not None else distribution_factory(initial_pool)
        )
        # Thread the win-rate matrix + prior-round learner id into
        # sample(): the wrapper's call site does not pass these kwargs, so
        # wrap with a per-round proxy that forwards them (uniform/latest-N
        # ignore them; PFSP consumes them). The learner_id is the most
        # recently appended checkpoint (round K-1's snapshot) - round K's
        # does not exist yet.
        prior_learner_members = [m for m in initial_pool.members() if m.kind == "checkpoint"]
        prior_learner_id = prior_learner_members[-1].member_id if prior_learner_members else None
        round_distribution: OpponentDistribution = _MatrixAwareDistribution(
            inner=base_distribution,
            win_rate_matrix=win_rate_matrix,
            learner_id=prior_learner_id,
        )
        wrapper = LeagueEnvWrapper(
            wrapped_env_factory=_single_learner_wrapped_factory_guard,
            pool=initial_pool,
            distribution=round_distribution,
            # Per-round spawn-derived seed.
            league_seed=per_round_league_seeds[round_index],
            league_mode="single_learner",
            single_learner_env_factory=_single_learner_factory,
            max_episode_steps=max_episode_steps,
            episode_seed_base=core_seed,
        )
        try:
            vec = _build_bridged_vec(wrapper)
            try:
                curve: list[tuple[int, float]] = []
                callback = _make_curve_callback(curve)
                model = _make_shared_ppo(
                    vec,
                    seed=core_seed,
                    policy_kind=policy_kind,
                    ent_coef=ent_coef,
                    gae_lambda=gae_lambda,
                )
                if warm_start and round_index >= 2:
                    warm_start_source = prior_learner_members[-1]
                    warm_start_checkpoint_path = warm_start_source.checkpoint_path
                    if warm_start_checkpoint_path is None:
                        # Defensive: a checkpoint member is guaranteed a path by
                        # __post_init__, but typing infers Path | None.
                        raise RuntimeError(
                            f"warm start: checkpoint PoolMember "
                            f"{warm_start_source.member_id!r} has no checkpoint_path; "
                            f"the dataclass invariant was bypassed"
                        )
                    _load_warm_start_parameters(
                        model,
                        source_checkpoint=initial_pool.pool_dir / warm_start_checkpoint_path,
                        policy_kind=policy_kind,
                    )
                    logger.info(
                        "warm start: round %d policy initialized from %s",
                        round_index,
                        warm_start_source.member_id,
                    )
                model.learn(
                    total_timesteps=train_steps_per_round,
                    callback=callback,
                    progress_bar=False,
                )
                round_k_member = initial_pool.add_checkpoint(
                    model,
                    round_index=round_index,
                    seed=core_seed,
                    policy_kind=policy_kind,
                )
                learning_curves.append(tuple(curve))
            finally:
                vec.close()
        finally:
            wrapper.close()

        # Round-end head-to-head evaluation: runs for all distributions
        # (uniform/latest-N ignore the matrix; PFSP consumes it next round).
        # The matrix is persisted atomically after each fold so a resumed run
        # sees the cumulative record. The per-round seed offset keeps the eval
        # reproducible and distinct from the evaluator's internal offsets.
        round_end_results = _run_round_end_evaluation(
            initial_pool,
            round_k_member,
            n_episodes=round_end_eval_episodes,
            config=config,
            seed=core_seed + round_index,
            home_regions=home_regions,
        )
        for opp_id, entry in round_end_results.items():
            # record(...) adds one outcome per call; fold the per-pair
            # bucketed wins/losses into the cumulative matrix.
            for _ in range(entry.wins):
                win_rate_matrix.record(
                    learner_id=round_k_member.member_id,
                    opponent_id=opp_id,
                    learner_won=True,
                )
            for _ in range(entry.total - entry.wins):
                win_rate_matrix.record(
                    learner_id=round_k_member.member_id,
                    opponent_id=opp_id,
                    learner_won=False,
                )
        win_rate_matrix.save(win_rate_matrix_path)

    checkpoints = tuple(m for m in initial_pool.members() if m.kind == "checkpoint")
    # n_learning_agents is reserved for forward compatibility (a 2-learner
    # shared-policy variant); the current loop is single-learner only.
    del n_learning_agents
    return LeagueTrainingResult(
        pool_dir=initial_pool.pool_dir,
        rounds_completed=n_rounds,
        checkpoints=checkpoints,
        learning_curves=tuple(learning_curves),
        seeds=seeds,
        train_steps_per_round=train_steps_per_round,
        win_rate_matrix_snapshot=win_rate_matrix,
        max_episode_steps=max_episode_steps,
    )


@dataclass(frozen=True)
class PoolSnapshotImprovementReport:
    """ """

    round_n_mean: float
    round_1_mean: float
    round_n_std: float
    round_1_std: float
    margin: float
    threshold: float
    beats_round_1: bool
    n_episodes: int
    per_round_means: tuple[tuple[int, float], ...]


# The four head-to-head-scored reward components, in RewardComponents
# field order; the scoreboard sink-filling code below iterates this tuple.
HEAD_TO_HEAD_COMPONENTS: tuple[str, ...] = ("profit", "revenue", "market_share", "loyalty")

# The subset of the above whose per-episode reduction is a per-tick MEAN
# (share/EMA are per-tick quantities; summing them across ticks is not
# meaningful). A named constant so a new component can't inherit the wrong
# reduction by accident.
PER_TICK_MEAN_COMPONENTS: tuple[str, ...] = ("market_share", "loyalty")


def _head_to_head_episode_scores(
    env: Any,
    cfg: Any,
    *,
    seed: int,
    predict_for_agent: dict[str, Callable[..., Any]],
    component_sink: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """ """
    from retail_simulator.envs.learnability import DEFAULT_MAX_EPISODE_STEPS
    from retail_simulator.envs.parallel_gate import (
        _components_from_info,
        weighted_objective,
    )

    observations, _infos = env.reset(seed=seed)
    totals: dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
    if component_sink is not None:
        for aid in env.possible_agents:
            component_sink[aid] = {
                "profit": 0.0,
                "revenue": 0.0,
                "market_share": 0.0,
                "loyalty": 0.0,
            }
    is_first_step = True
    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
        actions = {
            aid: predict_for_agent[aid](observations[aid], episode_start=is_first_step)
            for aid in env.agents
        }
        observations, _rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            totals[aid] += weighted_objective(_components_from_info(infos[aid]), cfg)
            if component_sink is not None:
                # A second, separate reconstruction from the same info, kept
                # apart so the score-computation line above stays untouched.
                components = _components_from_info(infos[aid])
                component_sink[aid]["profit"] += components.profit
                component_sink[aid]["revenue"] += components.revenue
                component_sink[aid]["market_share"] += components.market_share
                component_sink[aid]["loyalty"] += components.loyalty
        is_first_step = False
    if component_sink is not None:
        for aid in env.possible_agents:
            component_sink[aid]["n_ticks"] = float(DEFAULT_MAX_EPISODE_STEPS)
    return totals


def _position_controlled_head_to_head(
    env: Any,
    cfg: Any,
    *,
    predict_a: Callable[..., Any],
    predict_b: Callable[..., Any],
    n_episodes: int,
    seed: int,
    env_factory: Callable[[], Any] | None = None,
    component_sink: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[list[float], list[float]]:
    """ """
    scores_a: list[float] = []
    scores_b: list[float] = []
    # Agent IDs are stable across env constructions (seats 0..N-1), so the
    # passed env's agent list is a valid stand-in. When env is None (the
    # reuse_env=False path), peek at one freshly-built env and close it just
    # to learn the set.
    if env is not None:
        possible_agents = list(env.possible_agents)
    else:
        if env_factory is None:  # pragma: no cover - defensive; not reachable from callers
            raise ValueError(
                "_position_controlled_head_to_head: either env or env_factory must be provided"
            )
        peek_env = env_factory()
        try:
            possible_agents = list(peek_env.possible_agents)
        finally:
            peek_env.close()
    for episode in range(n_episodes):
        episode_seed = seed + episode
        a_seat_vals: list[float] = []
        b_seat_vals: list[float] = []
        # One (a_seat, component sink) pair per seat rotation this episode,
        # collected so the per-component means mirror scores_a/scores_b.
        rotation_component_sinks: list[tuple[str, dict[str, dict[str, float]]]] = []
        for a_seat in possible_agents:
            predict_for_agent: dict[str, Callable[..., Any]] = {
                aid: (predict_a if aid == a_seat else predict_b) for aid in possible_agents
            }
            inner_component_sink: dict[str, dict[str, float]] | None = (
                {} if component_sink is not None else None
            )
            if env_factory is None:
                totals = _head_to_head_episode_scores(
                    env,
                    cfg,
                    seed=episode_seed,
                    predict_for_agent=predict_for_agent,
                    component_sink=inner_component_sink,
                )
            else:
                # Fresh-env path: build, reset+step, close.
                episode_env = env_factory()
                try:
                    totals = _head_to_head_episode_scores(
                        episode_env,
                        cfg,
                        seed=episode_seed,
                        predict_for_agent=predict_for_agent,
                        component_sink=inner_component_sink,
                    )
                finally:
                    episode_env.close()
            a_seat_vals.append(totals[a_seat])
            b_seat_vals.extend(v for aid, v in totals.items() if aid != a_seat)
            if inner_component_sink is not None:
                rotation_component_sinks.append((a_seat, inner_component_sink))
        scores_a.append(float(np.mean(a_seat_vals)))
        scores_b.append(float(np.mean(b_seat_vals)))
        if component_sink is not None:
            for name in HEAD_TO_HEAD_COMPONENTS:
                component_a_vals: list[float] = []
                component_b_vals: list[float] = []
                for rotation_a_seat, inner_sink in rotation_component_sinks:
                    for aid, agent_totals in inner_sink.items():
                        value = (
                            agent_totals[name] / agent_totals["n_ticks"]
                            if name in PER_TICK_MEAN_COMPONENTS
                            else agent_totals[name]
                        )
                        if aid == rotation_a_seat:
                            component_a_vals.append(value)
                        else:
                            component_b_vals.append(value)
                a_side_mean = float(np.mean(component_a_vals))
                b_side_mean = float(np.mean(component_b_vals))
                component_sink.setdefault(name, []).append((a_side_mean, b_side_mean))
    return scores_a, scores_b


_PER_ROUND_CURVE_EPISODES: int = 3


def evaluate_pool_snapshot_improvement(
    pool: FrozenPolicyPool,
    training_result: LeagueTrainingResult,
    *,
    n_eval_episodes: int = 20,
    config: CoreConfig | None = None,
    seed: int = 0,
    reuse_env: bool = True,
    home_regions: tuple[int, ...] | None = None,
) -> PoolSnapshotImprovementReport:
    """ """
    # Lazy import: RetailParallelEnv pulls pettingzoo, which this module
    # otherwise avoids until call time.
    from retail_simulator.envs.parallel_env import RetailParallelEnv
    from retail_simulator.envs.learnability import GATE_SIGMA_MULTIPLE
    from retail_simulator.envs.parallel_gate import competition_config

    checkpoints = training_result.checkpoints
    if len(checkpoints) < 2:
        raise ValueError(
            "evaluate_pool_snapshot_improvement requires at least 2 checkpoints "
            "(round-1 and round-N) to compare; "
            f"got {len(checkpoints)} checkpoint(s) in training_result"
        )

    # Round 1 is the first trained checkpoint (trained against the round-0
    # random-init pool); round N (max round_index) is the latest snapshot.
    by_round = {member.round_index: member for member in checkpoints}
    if 1 not in by_round:
        raise ValueError(
            "evaluate_pool_snapshot_improvement requires a checkpoint at round_index=1 "
            f"(the first trained snapshot); got round_indexes {sorted(by_round)}"
        )
    round_1_member = by_round[1]
    round_n_member = checkpoints[-1]  # checkpoints are appended in round order

    effective_config = config if config is not None else competition_config()
    cfg = effective_config.reward

    predict_round_n = pool.materialize(round_n_member, env=None)
    predict_round_1 = pool.materialize(round_1_member, env=None)

    # Either build the eval env once and reuse it across all episodes
    # (default), or build a fresh env per inner episode (the reuse_env=False
    # path). The two paths produce field-equal reports.
    def _build_eval_env() -> Any:
        return RetailParallelEnv(
            effective_config, n_learning_agents=2, seed=seed, home_regions=home_regions
        )

    if reuse_env:
        shared_env: Any | None = _build_eval_env()
        primitive_env_factory: Callable[[], Any] | None = None
    else:
        shared_env = None
        primitive_env_factory = _build_eval_env
    try:
        round_n_scores, round_1_scores = _position_controlled_head_to_head(
            shared_env,
            cfg,
            predict_a=predict_round_n,
            predict_b=predict_round_1,
            n_episodes=n_eval_episodes,
            seed=seed,
            env_factory=primitive_env_factory,
        )

        # Per-round means curve: for each checkpoint k (round-index order),
        # a cheap mini-eval of round-k vs round-1 (round-1 vs round-0 when
        # k == 1). Each entry is round-k's mean over _PER_ROUND_CURVE_EPISODES.
        # The pool's load cache reads each checkpoint zip at most once.
        per_round: list[tuple[int, float]] = []
        for member in checkpoints:
            predict_k = pool.materialize(member, env=None)
            if member.round_index == 1:
                # Round 1's peer is round 0 (the random-init snapshot).
                round_0 = by_round.get(0)
                if round_0 is None:
                    # A 1-round result may lack round 0 (a hand-built pool);
                    # compare round 1 against itself so the curve is defined.
                    peer_predict = predict_k
                else:
                    peer_predict = pool.materialize(round_0, env=None)
            else:
                peer_predict = predict_round_1
            k_scores, _peer_scores = _position_controlled_head_to_head(
                shared_env,
                cfg,
                predict_a=predict_k,
                predict_b=peer_predict,
                n_episodes=_PER_ROUND_CURVE_EPISODES,
                seed=seed + 10_000 + member.round_index,
                env_factory=primitive_env_factory,
            )
            per_round.append((member.round_index, float(np.mean(k_scores))))
    finally:
        if shared_env is not None:
            shared_env.close()

    round_n_mean = float(np.mean(round_n_scores))
    round_1_mean = float(np.mean(round_1_scores))
    round_n_std = float(np.std(round_n_scores))
    round_1_std = float(np.std(round_1_scores))
    margin = round_n_mean - round_1_mean
    threshold = float(GATE_SIGMA_MULTIPLE) * round_1_std
    beats_round_1 = margin >= threshold

    return PoolSnapshotImprovementReport(
        round_n_mean=round_n_mean,
        round_1_mean=round_1_mean,
        round_n_std=round_n_std,
        round_1_std=round_1_std,
        margin=margin,
        threshold=threshold,
        beats_round_1=beats_round_1,
        n_episodes=n_eval_episodes,
        per_round_means=tuple(per_round),
    )


# --- Round-end head-to-head evaluation ----------------------------------------


def _run_round_end_evaluation(
    pool: FrozenPolicyPool,
    learner_member: PoolMember,
    *,
    n_episodes: int = 5,
    config: "CoreConfig | None" = None,
    seed: int = 0,
    home_regions: tuple[int, ...] | None = None,
) -> dict[str, WinRateEntry]:
    """ """
    # Lazy imports: RetailParallelEnv pulls pettingzoo, kept out of module load.
    from retail_simulator.envs.parallel_env import RetailParallelEnv
    from retail_simulator.envs.learnability import DEFAULT_MAX_EPISODE_STEPS
    from retail_simulator.envs.parallel_gate import (
        _components_from_info,
        competition_config,
        weighted_objective,
    )

    if learner_member.kind != "checkpoint":
        raise ValueError(
            f"_run_round_end_evaluation: learner_member must be a checkpoint, "
            f"got kind={learner_member.kind!r} (the learner is the current round's "
            f"trained policy; scripted members can only be opponents)"
        )

    opponents = tuple(m for m in pool.members() if m.member_id != learner_member.member_id)
    if not opponents:
        return {}

    effective_config = config if config is not None else competition_config()
    reward_cfg = effective_config.reward

    learner_predict = pool.materialize(learner_member, env=None)
    rng = np.random.default_rng(seed)
    results: dict[str, WinRateEntry] = {}

    for opp_idx, opp in enumerate(opponents):
        wins = 0
        if opp.kind in ("checkpoint", "archetype"):
            # Checkpoint-vs-checkpoint (or vs an archetype): position-
            # controlled head-to-head, reusing the shared primitive so the
            # scoring substrate is identical to evaluate_pool_snapshot_improvement.
            opp_predict = pool.materialize(opp, env=None)
            env = RetailParallelEnv(
                effective_config, n_learning_agents=2, seed=seed, home_regions=home_regions
            )
            try:
                # Per-opponent base seed offset so concurrent eval loops
                # don't collide on the same episode_seed.
                pair_seed = seed + opp_idx * 10_000
                learner_scores, opp_scores = _position_controlled_head_to_head(
                    env,
                    reward_cfg,
                    predict_a=learner_predict,
                    predict_b=opp_predict,
                    n_episodes=n_episodes,
                    seed=pair_seed,
                )
            finally:
                env.close()
            # One episode -> one win/loss outcome (mean per-seat scores;
            # ties via the seeded rng coin flip).
            for learner_score, opp_score in zip(learner_scores, opp_scores, strict=True):
                if learner_score > opp_score:
                    wins += 1
                elif learner_score == opp_score:
                    if rng.random() < 0.5:
                        wins += 1
        else:
            # Scripted opponent: a FRESH single-learner env per episode,
            # because each episode's seed differs and the env is bound to its
            # constructor seed.
            for ep_idx in range(n_episodes):
                ep_seed = seed + opp_idx * 10_000 + ep_idx * 1_000
                env_sl = RetailParallelEnv(
                    effective_config,
                    n_learning_agents=1,
                    npc_archetypes=(opp.archetype_name,),
                    seed=ep_seed,
                    home_regions=home_regions,
                )
                try:
                    observations, _infos = env_sl.reset(seed=ep_seed)
                    learner_total = 0.0
                    is_first_step = True
                    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
                        learner_obs = observations["retailer_0"]
                        learner_action = learner_predict(learner_obs, episode_start=is_first_step)
                        joint = {"retailer_0": learner_action}
                        observations, _rewards, _term, _trunc, infos = env_sl.step(joint)
                        learner_total += weighted_objective(
                            _components_from_info(infos["retailer_0"]), reward_cfg
                        )
                        is_first_step = False
                finally:
                    env_sl.close()
                # Fallback win signal: the env does not surface the NPC's
                # per-seat objective, so treat the scripted opponent as a
                # zero-objective baseline - learner wins iff its score > 0.
                # Non-degenerate enough for PFSP's weight curve; exact-zero
                # ties resolve via the seeded coin flip.
                if learner_total > 0.0:
                    wins += 1
                elif learner_total == 0.0:
                    if rng.random() < 0.5:
                        wins += 1
        results[opp.member_id] = WinRateEntry(wins=wins, total=n_episodes)

    return results


@dataclass(frozen=True)
class RealPpoGeneralizationDiagnostic:
    """ """

    mean_curriculum_score: float
    mean_fixed_score: float
    per_scenario_curriculum_means: tuple[float, ...]
    per_scenario_fixed_means: tuple[float, ...]
    margin: float
    n_scenarios: int
    n_episodes_per_scenario: int
    train_steps: int
    per_scenario_curriculum_std_rewards: tuple[float, ...] = ()
    per_scenario_fixed_std_rewards: tuple[float, ...] = ()


def _default_holdout_scenarios() -> tuple[CoreConfig, ...]:
    """ """
    from retail_simulator.envs.curriculum import (
        _VARIANT_B_REGION_0_MIX,
        _VARIANT_B_REGION_1_MIX,
        _VARIANT_C_REGION_0_MIX,
        _VARIANT_C_REGION_1_MIX,
        phase_1_4_six_segment_config,
    )

    # Mirror variant-B (region-1 mix in region 0; region-0 mix in region 1).
    mirror_b = phase_1_4_six_segment_config(
        region_0_mix=_VARIANT_B_REGION_1_MIX,
        region_1_mix=_VARIANT_B_REGION_0_MIX,
    )
    # Mirror variant-C (the symmetric mirroring of the bargain-leaning regime).
    mirror_c = phase_1_4_six_segment_config(
        region_0_mix=_VARIANT_C_REGION_1_MIX,
        region_1_mix=_VARIANT_C_REGION_0_MIX,
    )
    return (mirror_b, mirror_c)


# Holdout scenarios the CLI's league generalization-diagnostic command
# defaults to; built lazily on first access to keep imports cheap.
_DEFAULT_HOLDOUT_SCENARIOS: tuple[CoreConfig, ...] | None = None


def default_holdout_scenarios() -> tuple[CoreConfig, ...]:
    """Cached accessor for the seed held-out scenarios.

    Populated on first call so the scenario helpers (and pettingzoo) are not
    pulled at module import; subsequent calls return the SAME tuple instance.
    """
    global _DEFAULT_HOLDOUT_SCENARIOS
    if _DEFAULT_HOLDOUT_SCENARIOS is None:
        _DEFAULT_HOLDOUT_SCENARIOS = _default_holdout_scenarios()
    return _DEFAULT_HOLDOUT_SCENARIOS


def run_real_ppo_generalization_diagnostic(
    curriculum: ScenarioDistribution,
    holdout_scenarios: tuple[CoreConfig, ...],
    train_steps: int,
    seeds: tuple[int, int],
    *,
    n_eval_episodes_per_scenario: int = 5,
    fixed_config: CoreConfig | None = None,
    n_learning_agents: int = 2,
) -> RealPpoGeneralizationDiagnostic:
    """ """
    if not holdout_scenarios:
        raise ValueError(
            "run_real_ppo_generalization_diagnostic requires at least one held-out "
            "scenario; the diagnostic measures per-scenario means"
        )
    if train_steps < 1:
        raise ValueError(
            f"train_steps must be >= 1, got {train_steps}; "
            "at least one timestep is needed to train either model"
        )
    if n_eval_episodes_per_scenario < 1:
        raise ValueError(
            f"n_eval_episodes_per_scenario must be >= 1, got {n_eval_episodes_per_scenario}; "
            "at least one evaluation episode per scenario is needed for a mean"
        )

    # Lazy heavy imports keep module import cheap.
    from retail_simulator.envs.curriculum import evaluate_generalization
    from retail_simulator.envs.parallel_gate import (
        _make_shared_ppo,
        build_parallel_vec_env,
        competition_config,
    )

    curriculum_seed, core_seed = seeds

    # --- Train the curriculum-PPO -------------------------------------------
    # build_parallel_vec_env wraps the parallel env in the curriculum wrapper
    # (training-time only; the schema is unchanged, so the bridged Box is
    # identical to the fixed run).
    vec_curric = build_parallel_vec_env(
        n_learning_agents=n_learning_agents,
        seed=core_seed,
        curriculum=curriculum,
        curriculum_seed=curriculum_seed,
    )
    try:
        curriculum_model = _make_shared_ppo(vec_curric, seed=core_seed)
        curriculum_model.learn(total_timesteps=train_steps, progress_bar=False)
    finally:
        vec_curric.close()

    # --- Train the fixed-PPO -------------------------------------------------
    fixed_config_effective = fixed_config if fixed_config is not None else competition_config()
    vec_fixed = build_parallel_vec_env(
        config=fixed_config_effective,
        n_learning_agents=n_learning_agents,
        seed=core_seed,
    )
    try:
        fixed_model = _make_shared_ppo(vec_fixed, seed=core_seed)
        fixed_model.learn(total_timesteps=train_steps, progress_bar=False)
    finally:
        vec_fixed.close()

    holdout_tuple = tuple(holdout_scenarios)
    report = evaluate_generalization(
        curriculum_model,
        fixed_model,
        holdout_tuple,
        episodes_per_scenario=n_eval_episodes_per_scenario,
        n_learning_agents=n_learning_agents,
        seed=core_seed,
    )

    return RealPpoGeneralizationDiagnostic(
        mean_curriculum_score=report.mean_curriculum_score,
        mean_fixed_score=report.mean_fixed_score,
        per_scenario_curriculum_means=report.per_scenario_curriculum_means,
        per_scenario_fixed_means=report.per_scenario_fixed_means,
        margin=report.margin,
        n_scenarios=report.n_scenarios,
        n_episodes_per_scenario=report.n_episodes_per_scenario,
        train_steps=int(train_steps),
    )


__all__ = [
    "DEFAULT_LEAGUE_NAME",
    "DEFAULT_POOL_SEED_ARCHETYPES",
    "HEAD_TO_HEAD_COMPONENTS",
    "KNOWN_LEAGUES",
    "MANIFEST_VERSION",
    "PER_TICK_MEAN_COMPONENTS",
    "WIN_RATE_MATRIX_VERSION",
    "FrozenPolicyPool",
    "LatestNOpponentDistribution",
    "LeagueEnvWrapper",
    "LeagueTrainingResult",
    "OpponentDistribution",
    "PFSPOpponentDistribution",
    "PoolMember",
    "PoolSnapshotImprovementReport",
    "RealPpoGeneralizationDiagnostic",
    "UniformOpponentDistribution",
    "WinRateEntry",
    "WinRateMatrix",
    "build_league",
    "default_holdout_scenarios",
    "evaluate_pool_snapshot_improvement",
    "known_leagues",
    "make_seeded_pool",
    "run_league_training",
    "run_real_ppo_generalization_diagnostic",
]
