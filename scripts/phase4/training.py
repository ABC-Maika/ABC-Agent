"""League training: PFSP self-play rounds with per-round policy snapshots.

Also provides checkpoint deduplication (by policy-weight hash) and checkpoint
materialization (predict closure loading).
"""

from __future__ import annotations

import hashlib
import logging
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from retail_simulator.core.config import CoreConfig
from retail_simulator.envs.spike import calibrated_spike_config

logger = logging.getLogger("phase4.training")

# PPO entropy-bonus coefficient — SB3's PPO default; the default value reproduces every prior run.
DEFAULT_ENT_COEF: float = 0.0
# PPO GAE lambda — SB3's PPO default; the default value reproduces every prior run.
DEFAULT_GAE_LAMBDA: float = 0.95
DEFAULT_PFSP_MIX_UNIFORM: float = 0.0


def train_league(
    *,
    pool_dir: Path,
    n_rounds: int,
    train_steps_per_round: int,
    pfsp_alpha: float,
    seeds: tuple[int, int, int],
    ent_coef: float = DEFAULT_ENT_COEF,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
    seed_archetypes: tuple[str, ...] = (),
    config: CoreConfig | None = None,
    home_regions: tuple[int, ...] | None = None,
    mask_expansion: bool = False,
    max_episode_steps: int | None = None,
    pfsp_mix_uniform: float = DEFAULT_PFSP_MIX_UNIFORM,
    warm_start: bool = False,
) -> Any:
    """Train the PFSP self-play league, snapshotting one PPO checkpoint per round into ``pool_dir``."""
    from retail_simulator.envs.league import (
        POOL_SPIKE_ARCHETYPE_LABELS,
        FrozenPolicyPool,
        OpponentDistribution,
        PFSPOpponentDistribution,
        make_seeded_pool,
        run_league_training,
    )

    effective_config = config if config is not None else calibrated_spike_config()
    # Record only a non-default reward mode on the manifest: "profit" is the universal
    # default, so recording it would grow every pool's manifest for no reader.
    reward_mode_to_record = (
        effective_config.reward.mode if effective_config.reward.mode != "profit" else None
    )
    is_calibrated_shaped = (
        effective_config.demand.regions == calibrated_spike_config().demand.regions
    )
    if is_calibrated_shaped:
        pool = make_seeded_pool(pool_dir, spike_archetypes=seed_archetypes)
    else:
        offending = tuple(name for name in seed_archetypes if name in POOL_SPIKE_ARCHETYPE_LABELS)
        if offending:
            raise ValueError(
                f"train_league: seed_archetypes {offending} are calibrated spike "
                "posture(s) with a FIXED expansion_choice — outside the calibrated "
                "lock's own region table (e.g. under the arena's, it only ever "
                "opens region 1), that fixed choice can no longer be assumed to "
                "target a meaningful region, which would silently corrupt the "
                "measurement; pass geo archetypes "
                "(harness.geo_archetypes.GEO_ARCHETYPE_LABELS) instead"
            )
        pool = make_seeded_pool(
            pool_dir, geo_archetypes=seed_archetypes, region_table=effective_config.demand.regions
        )
    dist = PFSPOpponentDistribution(alpha=pfsp_alpha, mix_uniform=pfsp_mix_uniform)

    def _distribution_factory(_pool: FrozenPolicyPool) -> OpponentDistribution:
        return dist

    logger.info(
        "training PFSP league: rounds=%d, steps/round=%d, pfsp_alpha=%s, "
        "pfsp_mix_uniform=%s, seeds=%s, "
        "ent_coef=%s, gae_lambda=%s, seed_archetypes=%s, home_regions=%s, "
        "mask_expansion=%s, max_episode_steps=%s, warm_start=%s, pool_dir=%s",
        n_rounds,
        train_steps_per_round,
        pfsp_alpha,
        pfsp_mix_uniform,
        seeds,
        ent_coef,
        gae_lambda,
        seed_archetypes,
        home_regions,
        mask_expansion,
        max_episode_steps,
        warm_start,
        pool_dir,
    )
    return run_league_training(
        initial_pool=pool,
        n_rounds=n_rounds,
        distribution_factory=_distribution_factory,
        train_steps_per_round=train_steps_per_round,
        seeds=seeds,
        config=effective_config,
        distribution=dist,
        pfsp_alpha=pfsp_alpha,
        ent_coef=ent_coef,
        gae_lambda=gae_lambda,
        home_regions=home_regions,
        mask_expansion=mask_expansion,
        reward_mode=reward_mode_to_record,
        max_episode_steps=max_episode_steps,
        warm_start=warm_start,
    )


def _policy_weight_hash(zip_path: Path) -> str:
    """Hash an SB3 checkpoint's policy weights (stdlib zipfile — never imports SB3).

    Two league rounds can converge to byte-identical weights; deduping on the SHA-256 of the zip's policy-weight bytes (first 16 hex chars) drops duplicates before the tournament.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        member = "policy.pth" if "policy.pth" in names else next(iter(sorted(names)))
        payload = zf.read(member)
    return hashlib.sha256(payload).hexdigest()[:16]


def distinct_checkpoints(pool_dir: Path, checkpoints: tuple[Any, ...]) -> dict[str, Path]:
    """Map ``label -> checkpoint zip path`` for the DISTINCT trained checkpoints (earliest round wins per weight hash).

    Labels are ``round_<n>``; the round-0 random-init snapshot is included as a legitimate distinct policy.
    """
    by_hash: dict[str, Path] = {}
    labels: dict[str, Path] = {}
    for member in checkpoints:
        if member.checkpoint_path is None:
            continue
        zip_path = pool_dir / member.checkpoint_path
        weight_hash = _policy_weight_hash(zip_path)
        if weight_hash in by_hash:
            logger.info(
                "dedup: round_%d checkpoint is weight-identical to an earlier round; skipping",
                member.round_index,
            )
            continue
        by_hash[weight_hash] = zip_path
        labels[f"round_{member.round_index}"] = zip_path
    return labels


def _materialize(label_to_zip: dict[str, Path]) -> dict[str, Callable[..., Any]]:
    """Load each distinct checkpoint's predict closure once (reuses the ladder loader; lazy import keeps module load cheap and SB3-free)."""
    from retail_simulator.envs.ladder import _materialize_predicts

    return _materialize_predicts({label: str(path) for label, path in label_to_zip.items()})
