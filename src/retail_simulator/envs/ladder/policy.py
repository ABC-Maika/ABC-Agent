"""Policy detection and predict-closure materialization for ladder participants."""

from __future__ import annotations

import base64
import copy
import json
import logging
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from retail_simulator.core.config import CoreConfig
from retail_simulator.envs.geo_archetypes import GEO_ARCHETYPE_LABELS, geo_archetypes
from retail_simulator.envs.spike_archetypes import ArchetypePredict, calibrated_archetypes

from .model import parse_participant_spec

if TYPE_CHECKING:
    from retail_simulator.envs.league.pool import PolicyKind

logger = logging.getLogger(__name__)

# First match wins. PPO and A2C (ActorCriticPolicy) match none and fall through to "mlp";
# DDPG saves TD3Policy, so it is detected as "td3".
_POLICY_KIND_MARKERS: tuple[tuple[tuple[bytes, ...], PolicyKind], ...] = (
    ((b"Recurrent", b"recurrent"), "lstm"),
    ((b"TQCPolicy", b".tqc."), "tqc"),
    ((b"SACPolicy", b".sac."), "sac"),
    ((b"TD3Policy", b".td3."), "td3"),
)


def _detect_policy_kind(checkpoint_path: str | Path) -> PolicyKind:
    """Peek a checkpoint zip and return the loader kind from its serialized policy class."""
    try:
        with zipfile.ZipFile(checkpoint_path) as zf, zf.open("data") as data_member:
            data = json.load(data_member)
        serialized = data["policy_class"][":serialized:"]
        decoded = base64.b64decode(serialized)
    except (
        zipfile.BadZipFile,
        KeyError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        # Peek failed: default to "mlp" and warn so a demoted non-PPO policy stays visible.
        logger.warning(
            "policy-kind auto-detect peek failed for %s (%s: %s); defaulting to 'mlp'",
            checkpoint_path,
            type(exc).__name__,
            exc,
        )
        return "mlp"
    for markers, kind in _POLICY_KIND_MARKERS:
        if any(marker in decoded for marker in markers):
            return kind
    return "mlp"


def _make_random_predict(seed: int, action_space: Any) -> Callable[..., Any]:
    """Build a predict closure that re-seeds a deep-copied action space each episode."""
    space = copy.deepcopy(action_space)
    episode_index = 0
    space.seed(seed)

    def predict(_obs: object, *, episode_start: bool = False) -> Any:
        nonlocal episode_index
        if episode_start:
            space.seed(seed + 1_000_003 * episode_index)
            episode_index += 1
        return np.asarray(space.sample(), dtype=np.float32)

    return predict


def _materialize_predicts(
    spec_by_label: dict[str, str],
    *,
    action_space: Any | None = None,
    config: CoreConfig | None = None,
) -> dict[str, Callable[..., Any]]:
    """Build each participant's predict closure once, routed by its spec's kind.

    Checkpoints wrap a one-member FrozenPolicyPool (lazy SB3 import). Spike
    archetype names share one calibrated_archetypes() build; geo names share one
    geo_archetypes() build and need ``config``; random baselines need
    ``action_space``. The map is reused across every pair and seed, so each
    checkpoint loads once rather than per (pair, seed).
    """
    # Lazy import: keep module load cheap and sb3-free.
    from retail_simulator.envs.league import (
        FrozenPolicyPool,
        PoolMember,
    )

    predicts: dict[str, Callable[..., Any]] = {}
    spike_archetype_map: dict[str, ArchetypePredict] | None = None
    geo_archetype_map: dict[str, Callable[..., Any]] | None = None
    for label, spec in spec_by_label.items():
        parsed = parse_participant_spec(spec)
        if parsed.kind == "archetype":
            if parsed.default_label in GEO_ARCHETYPE_LABELS:
                if geo_archetype_map is None:
                    if config is None:
                        raise ValueError(
                            f"participant {label!r} ({spec!r}) is a geo archetype but "
                            "no config was provided to materialize it (no region table "
                            "to build geo_archetypes() from)"
                        )
                    # One geo_archetypes() call per invocation, shared across geo
                    # participants; config.demand.regions is read only here, so a
                    # field with no geo participant never needs it.
                    region_table = config.demand.regions
                    geo_archetype_map = geo_archetypes(len(region_table), region_table)
                predicts[label] = geo_archetype_map[parsed.default_label]
            else:
                if spike_archetype_map is None:
                    # One calibrated_archetypes() call per invocation, shared
                    # across spike participants.
                    spike_archetype_map = calibrated_archetypes()
                predicts[label] = spike_archetype_map[parsed.default_label]
        elif parsed.kind == "random":
            if action_space is None:
                raise ValueError(
                    f"participant {label!r} ({spec!r}) is a random baseline but no "
                    "action_space was provided to materialize it"
                )
            # parse_participant_spec sets seed for "random" and only "random".
            assert parsed.seed is not None
            predicts[label] = _make_random_predict(parsed.seed, action_space)
        elif parsed.kind == "checkpoint":
            # materialize resolves checkpoint_path against pool_dir, so seat
            # pool_dir at the zip's parent and pass the bare filename.
            path = Path(spec)
            policy_kind = _detect_policy_kind(path)
            member = PoolMember(
                member_id=f"checkpoint:ladder_{label}",
                kind="checkpoint",
                archetype_name=f"frozen_policy:ladder_{label}",
                checkpoint_path=Path(path.name),
                round_index=0,
                policy_kind=policy_kind,
            )
            pool = FrozenPolicyPool(path.parent, members=(member,))
            predicts[label] = pool.materialize(member, env=None)
        else:
            # Unreachable; an explicit raise so a future kind can't fall through
            # into the checkpoint branch above.
            raise ValueError(f"participant {label!r} has unhandled kind {parsed.kind!r}")
    return predicts
