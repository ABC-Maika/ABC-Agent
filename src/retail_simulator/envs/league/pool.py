"""League pool primitives: members, the frozen policy pool, and named leagues.

The low-level layer of the ``envs.league`` subpackage. Holds the dataclasses
and classes that describe *what a league is* (a ``PoolMember``, a
:class:`FrozenPolicyPool`) and the named-league registry that the CLI and
training drivers resolve by name. No head-to-head scoring, no training loops,
no evaluation reporting — those live in the sibling modules and import from
here.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np

from retail_simulator.envs.geo_archetypes import GEO_ARCHETYPE_LABELS
from retail_simulator.envs.geo_archetypes import geo_archetypes as _geo_archetypes
from retail_simulator.envs.spike_archetypes import AGGRESSOR, LEAN, PATIENT, calibrated_archetypes

if TYPE_CHECKING:
    from retail_simulator.core.config import RegionConfig

logger = logging.getLogger(__name__)

# Manifest schema version: bump only on a breaking on-disk format change.
MANIFEST_VERSION: int = 1

# Version stamped on a manifest that carries arena-seating provenance
# (the three seating keys are present); see FrozenPolicyPool.save/.load.
MANIFEST_VERSION_ARENA_SEATING: int = 2

# Version stamped on a manifest that carries a recorded reward_mode;
# see FrozenPolicyPool.record_reward_mode/.save/.load.
MANIFEST_VERSION_REWARD_MODE: int = 3

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

# SB3 class a checkpoint zip is loaded with. A2C loads as "mlp" (PPO) and DDPG as "td3":
# the saved policy class and the deterministic inference are identical.
PolicyKind = Literal["mlp", "lstm", "sac", "td3", "tqc"]
_POLICY_KINDS: tuple[str, ...] = get_args(PolicyKind)


@dataclass(frozen=True)
class PoolMember:
    """One league participant: scripted NPC, trained checkpoint, or fixed archetype."""

    member_id: str
    kind: PoolMemberKind
    archetype_name: str
    checkpoint_path: Path | None
    round_index: int
    policy_kind: PolicyKind = "mlp"

    def __post_init__(self) -> None:
        if not self.member_id:
            raise ValueError("PoolMember.member_id must be non-empty")
        # Validate policy_kind at the boundary: anything outside PolicyKind is a programming error.
        if self.policy_kind not in _POLICY_KINDS:
            raise ValueError(
                f"PoolMember {self.member_id!r} policy_kind must be one of "
                f"{_POLICY_KINDS}; got {self.policy_kind!r}"
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


def _load_checkpoint_model(member_id: str, policy_kind: PolicyKind, path: Path) -> Any:
    """Load the checkpoint zip at ``path`` with the SB3 class that ``policy_kind`` names."""
    if policy_kind in ("lstm", "tqc"):
        # Lazy sb3-contrib import: importing this module without the [recurrent]
        # extra still succeeds; the ImportError surfaces only when such a
        # checkpoint is materialized.
        try:
            from sb3_contrib import TQC, RecurrentPPO  # lazy by design
        except ImportError as exc:  # pragma: no cover - exercised when extra missing
            raise ImportError(
                f"materializing PoolMember {member_id!r} with policy_kind={policy_kind!r} "
                f"requires sb3-contrib; install with: pip install 'retail-simulator[recurrent]' "
                f"(or: pip install 'sb3-contrib>=2.3,<3.0')"
            ) from exc
        return (RecurrentPPO if policy_kind == "lstm" else TQC).load(path)
    if policy_kind == "sac":
        from stable_baselines3 import SAC  # lazy by design

        return SAC.load(path)
    if policy_kind == "td3":
        from stable_baselines3 import TD3  # lazy by design

        return TD3.load(path)
    from stable_baselines3 import PPO  # lazy by design

    return PPO.load(path)


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
        """Locked fairness knobs the member was trained with; None when the pool has none."""
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
            from dataclasses import replace

            from retail_simulator.core.config import CoreConfig
            from retail_simulator.envs.arena import (
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
        policy_kind: PolicyKind = "mlp",
    ) -> PoolMember:
        """Save ``model`` to ``{pool_dir}/round_{round_index}.zip``, append a checkpoint member, persist the manifest atomically.

        The seed participates only in the member_id naming — a provenance handle, not
        an RNG draw. The checkpoint is saved BEFORE the manifest is rewritten, so a
        crash between leaves an orphan zip (recoverable) rather than a manifest
        pointing at a missing zip. ``policy_kind`` is the caller's word (a
        :data:`PolicyKind`); it selects the loader and closure shape at :meth:`materialize`
        time.
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
        """Load (and cache) the callable ``predict`` implementation for ``member``."""
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
            self._materialize_cache[member.member_id] = _load_checkpoint_model(
                member.member_id, member.policy_kind, self._pool_dir / member.checkpoint_path
            )
        loaded_model = self._materialize_cache[member.member_id]

        if member.policy_kind != "lstm":

            def _checkpoint_predict_feedforward(obs: Any, **_kwargs: Any) -> Any:
                # **_kwargs absorbs the episode_start kwarg; feedforward policies
                # carry no recurrent state, so it is accepted and ignored.
                action, _state = loaded_model.predict(obs, deterministic=True)
                return action

            return _checkpoint_predict_feedforward

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
def make_seeded_pool(
    pool_dir: Path,
    *,
    archetypes: tuple[str, ...] = DEFAULT_POOL_SEED_ARCHETYPES,
    spike_archetypes: tuple[str, ...] = (),
    geo_archetypes: tuple[str, ...] = (),
    region_table: tuple[RegionConfig, ...] | None = None,
) -> FrozenPolicyPool:
    """Build a pool dir seeded with the given scripted/calibrated members."""
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

