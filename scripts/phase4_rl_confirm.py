#!/usr/bin/env python
"""Phase 4 RL-confirmation driver — do TRAINED self-play agents form the non-transitive cycle?

Three steps: (1) train a PFSP self-play league under :func:`calibrated_spike_config`, snapshotting a PPO checkpoint per round into the pool; (2) tournament the DISTINCT checkpoints (deduped by policy-weight hash) head-to-head, position-controlled over a fixed shared seed set, collecting continuous per-seed margins; (3) measure non-transitivity on the trained tournament (dominance graph, 3-cycles, Condorcet winner, decisive cycles) and write a summary JSON.

``rl_confirms = True`` means >=1 DECISIVE cycle (every edge's bootstrap CI excludes 0) AND no Condorcet winner. ``False`` (transitive / Condorcet winner) is a legitimate finding — the game has counters but RL collapsed to a corner — and is reported as-is, never forced.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import re
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import combinations
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from retail_simulator.core.config import CoreConfig
from retail_simulator.envs.ladder import _bootstrap_margin_ci
from retail_simulator.envs.spike import PairMargin, _measure_tournament, calibrated_spike_config

REPO = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO / "runs" / "phase4_rl_confirm"
DOCS_DATA = REPO / "docs" / "data"

logger = logging.getLogger("phase4_rl_confirm")

# Defaults (full run). PFSP grows a diverse pool over R rounds of S steps each.
DEFAULT_ROUNDS = 3
DEFAULT_STEPS_PER_ROUND = 300_000
DEFAULT_PFSP_ALPHA = 2.0
DEFAULT_PFSP_MIX_UNIFORM: float = 0.0
# The (league, curriculum, core) seed triple threaded into run_league_training.
DEFAULT_SEEDS: tuple[int, int, int] = (0, 0, 42)
# PPO entropy-bonus coefficient — SB3's PPO default; the default value reproduces every prior run.
DEFAULT_ENT_COEF: float = 0.0
# PPO GAE lambda — SB3's PPO default; the default value reproduces every prior run.
DEFAULT_GAE_LAMBDA: float = 0.95
# Tournament: fixed shared seed set 0..K-1 (the spike/ladder fairness contract).
DEFAULT_TOURNEY_SEEDS = 24
DEFAULT_TOURNEY_EPISODES = 1
DEFAULT_BOOTSTRAP_RESAMPLES = 1000

# Smoke budgets — exercise EVERY code path (train -> snapshot -> dedup -> tournament ->
# cycle measurement) fast enough to validate the driver end-to-end in ~1-2 min.
SMOKE_ROUNDS = 2
SMOKE_STEPS_PER_ROUND = 2_000
SMOKE_TOURNEY_SEEDS = 3
SMOKE_BOOTSTRAP_RESAMPLES = 200

# Fixed fact about the mechanism, always present in the summary (never a per-run value):
# --mask-expansion scopes to the learner's TRAINING rollouts only and never reaches any
# evaluation surface (the tournament below, the ladder, the snapshot-improvement and
# round-end evals, the arena), which all roll out unmasked regardless of the flag.
MASK_EXPANSION_SCOPE_NOTE = (
    "mask_expansion, when true, applies only to the learner's TRAINING rollouts; "
    "the post-training tournament (this run), harness/ladder.py's ladder, "
    "harness/league.py's snapshot-improvement + round-end evaluations, and "
    "harness/arena.py's arena all roll out UNMASKED regardless of this flag"
)


# --------------------------------------------------------------------------- #
# Step 1 — train a PFSP self-play league under the calibrated config           #
# --------------------------------------------------------------------------- #


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
    from retail_simulator.envs.league import (  # noqa: PLC0415
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


# --------------------------------------------------------------------------- #
# Step 2 — dedupe distinct checkpoints by policy-weight hash                    #
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Step 3 — tournament the distinct checkpoints under the calibrated config      #
# --------------------------------------------------------------------------- #


def _materialize(label_to_zip: dict[str, Path]) -> dict[str, Callable[..., Any]]:
    """Load each distinct checkpoint's predict closure once (reuses the ladder loader; lazy import keeps module load cheap and SB3-free)."""
    from retail_simulator.envs.ladder import _materialize_predicts  # noqa: PLC0415

    return _materialize_predicts({label: str(path) for label, path in label_to_zip.items()})


def tournament(
    label_to_zip: dict[str, Path],
    *,
    n_seeds: int,
    n_episodes: int,
    bootstrap_resamples: int,
    config: CoreConfig | None = None,
    home_regions: tuple[int, ...] | None = None,
) -> tuple[PairMargin, ...]:
    """Position-controlled head-to-head over every distinct-checkpoint pair; return PairMargins.

    One shared ``RetailParallelEnv(config, n_learning_agents=2)`` reused across every (pair, seed); per-seed continuous margin ``mean(scores_a)-mean(scores_b)``; the ladder's seed-axis bootstrap CI applied CONSISTENTLY across pairs (one resample draw, reused). ``config`` defaults to :func:`calibrated_spike_config`; pass the config the checkpoints were actually trained on so the tournament rates them on that world.
    """
    from retail_simulator.envs.parallel_env import RetailParallelEnv  # noqa: PLC0415
    from retail_simulator.envs.league import (  # noqa: PLC0415
        _position_controlled_head_to_head,
    )

    effective_config = config if config is not None else calibrated_spike_config()
    labels = tuple(sorted(label_to_zip))
    predicts = _materialize(label_to_zip)
    seed_set = tuple(range(n_seeds))

    env = RetailParallelEnv(
        effective_config, n_learning_agents=2, seed=seed_set[0], home_regions=home_regions
    )
    per_seed_by_pair: dict[tuple[str, str], list[float]] = {}
    try:
        for label_a, label_b in combinations(labels, 2):
            per_seed: list[float] = []
            for seed in seed_set:
                scores_a, scores_b = _position_controlled_head_to_head(
                    env,
                    effective_config.reward,
                    predict_a=predicts[label_a],
                    predict_b=predicts[label_b],
                    n_episodes=n_episodes,
                    seed=seed,
                )
                per_seed.append(float(np.mean(scores_a)) - float(np.mean(scores_b)))
            per_seed_by_pair[(label_a, label_b)] = per_seed
    finally:
        env.close()

    # Seed-axis bootstrap, drawn ONCE and applied consistently (the ladder coherence
    # contract — reuses _bootstrap_margin_ci exactly as run_spike does).
    rng = np.random.default_rng(0)
    resampled_seed_indices = rng.integers(0, n_seeds, size=(bootstrap_resamples, n_seeds))
    pair_per_seed: dict[tuple[str, str], np.ndarray] = {
        pair: np.asarray(vals, dtype=np.float64) for pair, vals in per_seed_by_pair.items()
    }

    pairwise: list[PairMargin] = []
    for label_a, label_b in combinations(labels, 2):
        arr = pair_per_seed[(label_a, label_b)]
        ci_low, ci_high = _bootstrap_margin_ci(
            label_a,
            [label_b],
            pair_per_seed,
            n_seeds=n_seeds,
            resampled_seed_indices=resampled_seed_indices,
        )
        pairwise.append(
            PairMargin(
                label_a=label_a,
                label_b=label_b,
                per_seed_margins=tuple(float(v) for v in arr),
                mean_margin=float(arr.mean()),
                margin_se=float(np.std(arr, ddof=0)) / sqrt(n_seeds),
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
    return tuple(pairwise)


# --------------------------------------------------------------------------- #
# Summary assembly + reporting                                                  #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConfirmResult:
    """The trained-agent non-transitivity measurement (the experiment's verdict)."""

    labels: tuple[str, ...]
    pairwise: tuple[PairMargin, ...]
    dominance: dict[str, tuple[str, ...]]
    n_cyclic_triples: int
    total_triples: int
    has_condorcet_winner: bool
    condorcet_winner: str | None
    decisive_cycle: bool
    rl_confirms: bool


def measure(pairwise: tuple[PairMargin, ...], labels: tuple[str, ...]) -> ConfirmResult:
    """Wrap :func:`_measure_tournament` in a :class:`ConfirmResult`, adding the rl_confirms verdict (>=1 decisive cycle AND no Condorcet winner)."""
    (
        dominance,
        n_cyclic_triples,
        has_condorcet_winner,
        condorcet_winner,
        _cycle_direction,
        decisive,
    ) = _measure_tournament(pairwise, labels)
    total_triples = len(list(combinations(labels, 3)))
    rl_confirms = n_cyclic_triples >= 1 and not has_condorcet_winner and decisive
    return ConfirmResult(
        labels=labels,
        pairwise=pairwise,
        dominance=dominance,
        n_cyclic_triples=n_cyclic_triples,
        total_triples=total_triples,
        has_condorcet_winner=has_condorcet_winner,
        condorcet_winner=condorcet_winner,
        decisive_cycle=decisive,
        rl_confirms=rl_confirms,
    )


def _summary_payload(
    result: ConfirmResult,
    *,
    rounds: int | None,
    train_steps_per_round: int | None,
    pfsp_alpha: float | None,
    pfsp_mix_uniform: float | None,
    seeds: tuple[int, int, int] | None,
    ent_coef: float | None,
    gae_lambda: float | None,
    mask_expansion: bool | None,
    max_episode_steps: int | None,
    warm_start: bool | None,
    reward_mode: str,
    seed_archetypes: tuple[str, ...],
    env: str,
    home_regions: tuple[int, ...] | None,
    arena_region_table_hash: str | None,
    arena_lock_knobs: dict[str, float] | None,
    label: str | None,
    tourney_seeds: int,
    tourney_episodes: int,
    smoke: bool,
    resumed: bool,
) -> dict[str, Any]:
    """Assemble the full summary JSON payload (training recipe, tournament setup, measurement, verdict)."""
    return {
        "experiment": "phase4_rl_confirm",
        "question": "Do TRAINED self-play agents form the non-transitive cycle, "
        "or collapse to a corner?",
        "smoke": smoke,
        "label": label,
        "config": (
            "calibrated_spike_config"
            if env == "calibrated"
            else (
                "arena_lock_config (candidate lock, RLB-8a)"
                if env == "arena"
                else "arena_balanced_config (credit-line-and-rebalance: working-"
                "capital credit line + smaller warehouse footprint)"
            )
        ),
        "training": {
            # `resumed` tells a reader WHY the other recipe fields below are null under --resume.
            "resumed": resumed,
            "n_rounds": rounds,
            "train_steps_per_round": train_steps_per_round,
            "max_episode_steps": max_episode_steps,
            "distribution": "pfsp",
            "pfsp_alpha": pfsp_alpha,
            "pfsp_mix_uniform": pfsp_mix_uniform,
            "warm_start": warm_start,
            "seeds": list(seeds) if seeds is not None else None,
            "ent_coef": ent_coef,
            "gae_lambda": gae_lambda,
            # null under --resume (train_league skipped); otherwise the as-typed CLI value.
            "mask_expansion": mask_expansion,
            # Fixed mechanism fact (see MASK_EXPANSION_SCOPE_NOTE), never nulled under --resume.
            "mask_expansion_scope": MASK_EXPANSION_SCOPE_NOTE,
            "reward_mode": reward_mode,
            # Always recorded (empty when the pool has no archetype members), so a missing
            # key never means "unknown"; under --resume this reflects the resumed pool's
            # actual manifest.
            "seed_archetypes": list(seed_archetypes),
            # Which world this run trained/tournamented on; under --resume derived from the
            # resumed pool's manifest (never nulled).
            "env": env,
            "home_regions": list(home_regions) if home_regions is not None else None,
            "arena_region_table_hash": arena_region_table_hash,
            "arena_lock_knobs": arena_lock_knobs,
        },
        "tournament": {
            "n_distinct_checkpoints": len(result.labels),
            "labels": list(result.labels),
            "n_seeds": tourney_seeds,
            "n_episodes": tourney_episodes,
        },
        "measurement": {
            "n_cyclic_triples": result.n_cyclic_triples,
            "total_triples": result.total_triples,
            "has_condorcet_winner": result.has_condorcet_winner,
            "condorcet_winner": result.condorcet_winner,
            "decisive_cycle": result.decisive_cycle,
            "dominance": {k: list(v) for k, v in result.dominance.items()},
            "pairwise": [
                {
                    "label_a": pm.label_a,
                    "label_b": pm.label_b,
                    "mean_margin": pm.mean_margin,
                    "ci_low": pm.ci_low,
                    "ci_high": pm.ci_high,
                    "ci_excludes_zero": pm.ci_low > 0.0 or pm.ci_high < 0.0,
                }
                for pm in result.pairwise
            ],
        },
        "rl_confirms": result.rl_confirms,
        "interpretation": (
            "RL CONFIRMS the non-transitivity: trained self-play policies form the "
            "rock-paper-scissors (>=1 decisive cycle, no Condorcet winner)."
            if result.rl_confirms
            else "RL does NOT confirm: the trained tournament is transitive / has a "
            "Condorcet winner. The GAME has counters (proven by the scripted spike) but "
            "RL collapsed to a corner — a real, publishable finding, reported honestly."
        ),
    }


def _print_table(result: ConfirmResult) -> None:
    """Print a compact human-readable table of the trained-agent tournament."""
    print("\n=== Phase 4 RL-confirmation — trained-agent tournament ===")
    print(f"distinct checkpoints: {len(result.labels)}  ({', '.join(result.labels)})")
    print(f"{'pair':<28}{'mean_margin':>14}{'ci_low':>12}{'ci_high':>12}{'decisive':>10}")
    for pm in result.pairwise:
        decisive_edge = pm.ci_low > 0.0 or pm.ci_high < 0.0
        print(
            f"{pm.label_a + ' vs ' + pm.label_b:<28}"
            f"{pm.mean_margin:>14.2f}{pm.ci_low:>12.2f}{pm.ci_high:>12.2f}"
            f"{('yes' if decisive_edge else 'no'):>10}"
        )
    print(f"\ndominance: {dict(result.dominance)}")
    print(
        f"n_cyclic_triples={result.n_cyclic_triples}/{result.total_triples}  "
        f"has_condorcet_winner={result.has_condorcet_winner} "
        f"(winner={result.condorcet_winner})  decisive_cycle={result.decisive_cycle}"
    )
    verdict = "RL CONFIRMS non-transitivity" if result.rl_confirms else "RL does NOT confirm"
    print(f"\nVERDICT: rl_confirms={result.rl_confirms}  ({verdict})\n")


# --------------------------------------------------------------------------- #
# Orchestration                                                                 #
# --------------------------------------------------------------------------- #


def _load_existing_pool_state(
    pool_dir: Path,
) -> tuple[
    tuple[Any, ...],
    tuple[str, ...],
    tuple[int, ...] | None,
    str | None,
    dict[str, float] | None,
    str | None,
]:
    """Reload the trained checkpoints + training provenance from an existing pool dir (--resume).

    Every value is read off the pool's on-disk manifest, NOT from this invocation's CLI flags: under --resume, train_league is skipped, so --seed-archetypes/--env/--reward reflect nothing about what the resumed pool actually contains or was trained under.

    Returns ``(checkpoints, seed_archetypes, home_regions, arena_region_table_hash, arena_lock_knobs, reward_mode)``; ``reward_mode`` is ``None`` for pools that never trained under an explicit "weighted" mode — the caller treats that like the "profit" default.
    """
    from retail_simulator.envs.league import FrozenPolicyPool  # noqa: PLC0415

    pool = FrozenPolicyPool.load(pool_dir)
    members = pool.members()
    checkpoints = tuple(m for m in members if m.kind == "checkpoint")
    seed_archetypes = tuple(m.archetype_name for m in members if m.kind == "archetype")
    return (
        checkpoints,
        seed_archetypes,
        pool.home_regions,
        pool.arena_region_table_hash,
        pool.arena_lock_knobs,
        pool.reward_mode,
    )


def _resolve_env(
    env: str,
) -> tuple[CoreConfig | None, tuple[int, ...] | None, str | None, dict[str, float] | None]:
    """Resolve ``--env`` into ``(config, home_regions, arena_region_table_hash, arena_lock_knobs)``.

    ``"calibrated"`` (default) -> all None (train_league's own :func:`calibrated_spike_config` default, no explicit seating). ``"arena"`` -> the 25-region lock (:func:`~retail_simulator.envs.arena.arena_lock_config`); ``"arena-balanced"`` -> the same table with a working-capital credit line + smaller warehouse footprint (:func:`~retail_simulator.envs.arena.arena_balanced_config`). Both arena variants seat both seats and record the region-table hash plus the economics knobs the hash alone cannot discriminate — ``credit_limit``/``overdraft_rate`` distinguish the two variants, and they are read off the resolved config so config/homes/hash/knobs are computed once and can never diverge.
    """
    if env in ("arena", "arena-balanced"):
        from retail_simulator.envs.arena import (  # noqa: PLC0415
            arena_balanced_config,
            arena_home_regions,
            arena_lock_config,
            arena_region_table_hash,
        )

        config = arena_balanced_config() if env == "arena-balanced" else arena_lock_config()
        home_regions = arena_home_regions(2)
        arena_lock_knobs = {
            "overhead_reference_stores": config.wage.overhead_reference_stores,
            "overhead_exponent": config.wage.overhead_exponent,
            "provided_capacity_per_store": config.warehouse.provided_capacity_per_store,
            "credit_limit": config.wage.credit_limit,
            "overdraft_rate": config.wage.overdraft_rate,
            "writeoff_frac_max": config.wage.writeoff_frac_max,
        }
        return config, home_regions, arena_region_table_hash(config), arena_lock_knobs
    return None, None, None, None


def _apply_reward_override(config: CoreConfig | None, reward: str | None) -> CoreConfig | None:
    """Return ``config`` with its reward mode replaced by ``reward`` (unchanged when ``reward is None``); a ``None`` config falls back to :func:`calibrated_spike_config` first."""
    if reward is None:
        return config
    effective = config if config is not None else calibrated_spike_config()
    return replace(effective, reward=replace(effective.reward, mode=reward))


def _effective_reward_mode(config: CoreConfig | None) -> str:
    """The reward mode ``config`` will actually train/tournament under.

    Mirrors train_league's/tournament's own ``calibrated_spike_config`` fallback, so the summary always names the SAME mode those functions resolve to. Always a concrete "profit"/"weighted" (never nulled), including under --resume, because the tournament step DOES run under --resume.
    """
    effective = config if config is not None else calibrated_spike_config()
    return effective.reward.mode


def run(args: argparse.Namespace) -> int:
    rounds = SMOKE_ROUNDS if args.smoke else args.rounds
    steps = SMOKE_STEPS_PER_ROUND if args.smoke else args.train_steps_per_round
    tourney_seeds = SMOKE_TOURNEY_SEEDS if args.smoke else args.tourney_seeds
    resamples = SMOKE_BOOTSTRAP_RESAMPLES if args.smoke else args.bootstrap_resamples
    seeds = args.seeds
    ent_coef = args.ent_coef
    gae_lambda = args.gae_lambda
    mask_expansion = args.mask_expansion
    max_episode_steps = args.league_max_episode_steps
    warm_start = args.warm_start
    run_label = args.label
    resumed = args.resume

    seed_archetypes: tuple[str, ...]
    if args.seed_archetypes:
        if args.env in ("arena", "arena-balanced"):
            from retail_simulator.envs.geo_archetypes import (  # noqa: PLC0415
                GEO_ARCHETYPE_LABELS,
            )

            seed_archetypes = GEO_ARCHETYPE_LABELS
        else:
            from retail_simulator.envs.league import (  # noqa: PLC0415
                POOL_SPIKE_ARCHETYPE_LABELS,
            )

            seed_archetypes = POOL_SPIKE_ARCHETYPE_LABELS
    else:
        seed_archetypes = ()

    run_name = "smoke" if args.smoke else args.run_name
    pool_dir = Path(args.pool_dir) if args.pool_dir else RUNS_ROOT / run_name
    pool_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        # --resume never calls train_league, so this invocation's CLI flags reflect nothing
        # about this pool: the resumed pool's own manifest is authoritative for seeding,
        # seating, and reward mode.
        (
            checkpoints,
            seed_archetypes,
            home_regions,
            region_table_hash,
            arena_lock_knobs,
            stored_reward_mode,
        ) = _load_existing_pool_state(pool_dir)
        if home_regions is None:
            env_choice = "calibrated"
        else:
            # "arena" and "arena-balanced" share the exact region table, so disambiguate on
            # `credit_limit` (0 vs ARENA_BALANCED_CREDIT_LIMIT), NOT provided_capacity_per_store:
            # the lock's own knobs may move (a future lock amendment could set its cap to 300,
            # colliding with the balanced variant), but credit_limit/overdraft_rate exist only
            # on the balanced variant.
            from retail_simulator.envs.arena import (  # noqa: PLC0415
                ARENA_BALANCED_CREDIT_LIMIT,
            )

            is_balanced = (
                arena_lock_knobs is not None
                and arena_lock_knobs.get("credit_limit") == ARENA_BALANCED_CREDIT_LIMIT
            )
            env_choice = "arena-balanced" if is_balanced else "arena"
        # The manifest-wins precedence above is deliberate; log when --env disagrees so it is
        # not silent.
        if args.env != env_choice:
            logger.warning(
                "--resume: --env=%s was given but the resumed pool's manifest "
                "says %s; using %s (the pool's recorded seating is authoritative, "
                "not the --env flag)",
                args.env,
                env_choice,
                env_choice,
            )
        # Only the config half is needed here (homes/hash/knobs came from the manifest);
        # fresh_region_table_hash guards against a moved lock.
        league_config, _, fresh_region_table_hash, _ = _resolve_env(env_choice)
        # --reward under --resume: train_league never runs, so the flag only affects the
        # tournament scoring. When OMITTED, the pool's manifest-recorded mode wins (a
        # weighted-trained pool must not be silently re-scored on profit). An EXPLICIT
        # --reward still wins over the manifest (re-scoring without retraining is a
        # deliberate, documented use), with a warning on mismatch.
        effective_reward = args.reward if args.reward is not None else stored_reward_mode
        if args.reward is not None:
            historical_reward_mode = (
                stored_reward_mode if stored_reward_mode is not None else "profit"
            )
            if args.reward != historical_reward_mode:
                logger.warning(
                    "--resume: --reward=%s was given but the resumed pool's manifest "
                    "records training under reward_mode=%s; the tournament will score "
                    "these checkpoints under %s, which is NOT what they were trained under",
                    args.reward,
                    historical_reward_mode,
                    args.reward,
                )
        league_config = _apply_reward_override(league_config, effective_reward)
        # Guard against a moved lock: the stored hash came from the manifest, but
        # league_config was rebuilt from today's arena config — if the lock moved since
        # training, the tournament would roll out on the new world while the summary
        # reports the old hash.
        if region_table_hash is not None and region_table_hash != fresh_region_table_hash:
            logger.error(
                "--resume: pool was trained under a different arena region table; refusing"
            )
            return 1
        if not checkpoints:
            logger.error(
                "--resume: no trained checkpoints found in %s; run without --resume", pool_dir
            )
            return 1
        logger.info("--resume: reusing %d trained checkpoint(s) in %s", len(checkpoints), pool_dir)
    else:
        env_choice = args.env
        league_config, home_regions, region_table_hash, arena_lock_knobs = _resolve_env(env_choice)
        # Apply --reward AFTER _resolve_env, so a resolver that already set mode="weighted"
        # is never silently overridden back to "profit".
        league_config = _apply_reward_override(league_config, args.reward)
        training_result = train_league(
            pool_dir=pool_dir,
            n_rounds=rounds,
            train_steps_per_round=steps,
            pfsp_alpha=args.pfsp_alpha,
            seeds=seeds,
            ent_coef=ent_coef,
            gae_lambda=gae_lambda,
            seed_archetypes=seed_archetypes,
            config=league_config,
            home_regions=home_regions,
            mask_expansion=mask_expansion,
            max_episode_steps=max_episode_steps,
            pfsp_mix_uniform=args.pfsp_mix_uniform,
            warm_start=warm_start,
        )
        checkpoints = training_result.checkpoints

    # The reward mode league_config ends up carrying (post-override, both branches) — always
    # concrete, never null.
    reward_mode = _effective_reward_mode(league_config)

    label_to_zip = distinct_checkpoints(pool_dir, checkpoints)
    if len(label_to_zip) < 3:
        logger.warning(
            "only %d distinct trained checkpoint(s) — a tournament needs >=3 for a 3-cycle; "
            "the dominance graph is reported but cannot exhibit non-transitivity. "
            "(Expected with --smoke's tiny round count; increase --rounds for a real run.)",
            len(label_to_zip),
        )

    # The tournament rolls out on the SAME (config, home_regions) the checkpoints were
    # trained under — but always UNMASKED, even for masked-trained checkpoints (see
    # MASK_EXPANSION_SCOPE_NOTE).
    pairwise = tournament(
        label_to_zip,
        n_seeds=tourney_seeds,
        n_episodes=args.tourney_episodes,
        bootstrap_resamples=resamples,
        config=league_config,
        home_regions=home_regions,
    )
    result = measure(pairwise, tuple(sorted(label_to_zip)))
    _print_table(result)

    # Under --resume the recipe knobs (rounds, steps, seeds, ...) describe nothing about how
    # this pool was trained — the manifest carries no recipe — so null them out; `resumed`
    # records why.
    payload = _summary_payload(
        result,
        rounds=None if resumed else rounds,
        train_steps_per_round=None if resumed else steps,
        pfsp_alpha=None if resumed else args.pfsp_alpha,
        pfsp_mix_uniform=None if resumed else args.pfsp_mix_uniform,
        seeds=None if resumed else seeds,
        ent_coef=None if resumed else ent_coef,
        gae_lambda=None if resumed else gae_lambda,
        mask_expansion=None if resumed else mask_expansion,
        max_episode_steps=None if resumed else max_episode_steps,
        warm_start=None if resumed else warm_start,
        reward_mode=reward_mode,
        seed_archetypes=seed_archetypes,
        env=env_choice,
        home_regions=home_regions,
        arena_region_table_hash=region_table_hash,
        arena_lock_knobs=arena_lock_knobs,
        label=run_label,
        tourney_seeds=tourney_seeds,
        tourney_episodes=args.tourney_episodes,
        smoke=args.smoke,
        resumed=resumed,
    )
    if args.summary:
        summary_path = Path(args.summary)
    else:
        today = _dt.date.today().isoformat()
        smoke_suffix = "_smoke" if args.smoke else ""
        label_suffix = f"_{run_label}" if run_label else ""
        summary_path = DOCS_DATA / f"phase4_rl_confirm_{today}{smoke_suffix}{label_suffix}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"summary written to {summary_path}")
    return 0


def _parse_seeds(value: str) -> tuple[int, int, int]:
    """Parse ``--seeds league,curriculum,core`` into the 3-tuple ``run_league_training`` wants.

    Boundary validation (argparse ``type=``): exactly 3 comma-separated ints, else a clean argparse usage error.
    """
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--seeds must be 3 comma-separated ints 'league,curriculum,core'; got {value!r}"
        )
    try:
        parsed_seeds = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--seeds must be 3 comma-separated ints 'league,curriculum,core'; got {value!r}"
        ) from exc
    league_seed, curriculum_seed, core_seed = parsed_seeds
    return (league_seed, curriculum_seed, core_seed)


def _parse_label(value: str) -> str:
    """Validate ``--label`` as a safe filename component (``[A-Za-z0-9_-]+``)."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError(
            f"--label must match [A-Za-z0-9_-]+ (a safe filename component); got {value!r}"
        )
    return value


def _parse_gae_lambda(value: str) -> float:
    """Validate ``--gae-lambda`` as a float in ``(0, 1]`` (argparse ``type=``).

    0 or below, or above 1, degrades PPO's advantage estimator silently (SB3 does not clamp it); rejecting it at the CLI boundary turns a typo into a clean usage error.
    """
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--gae-lambda must be a float in (0, 1]; got {value!r}"
        ) from exc
    if not (0.0 < parsed <= 1.0):
        raise argparse.ArgumentTypeError(f"--gae-lambda must be a float in (0, 1]; got {value!r}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny train+tournament to validate the driver end-to-end",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip training; tournament the existing pool's checkpoints",
    )
    parser.add_argument(
        "--run-name", default="run", help="sub-dir under runs/phase4_rl_confirm/ (full runs)"
    )
    parser.add_argument(
        "--pool-dir",
        default=None,
        help="override the pool dir (default: runs/phase4_rl_confirm/<run-name>)",
    )
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_ROUNDS, help="league rounds (full run)"
    )
    parser.add_argument(
        "--train-steps-per-round",
        type=int,
        default=DEFAULT_STEPS_PER_ROUND,
        help="PPO steps per round (full run)",
    )
    parser.add_argument(
        "--pfsp-alpha", type=float, default=DEFAULT_PFSP_ALPHA, help="PFSP weighting exponent"
    )
    parser.add_argument(
        "--pfsp-mix-uniform",
        type=float,
        default=DEFAULT_PFSP_MIX_UNIFORM,
        help=(
            f"""decision 10 (taken 2026-09-19): uniform-mixing floor passed to PFSPOpponentDistribution(mix_uniform=...) -- guarantees every eligible pool member keeps at least mix_uniform/n draw probability each round, preventing the single-opponent collapse measured (default:{DEFAULT_PFSP_MIX_UNIFORM}, no floor; must be in [0.0, 1.0])"""
        ),
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=DEFAULT_SEEDS,
        help=(
            "'league,curriculum,core' seed triple threaded into run_league_training "
            f"(comma-separated ints; default: {','.join(str(s) for s in DEFAULT_SEEDS)})"
        ),
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=DEFAULT_ENT_COEF,
        help=(
            "PPO/RecurrentPPO entropy-bonus coefficient, passed straight through to "
            f"_make_shared_ppo (SB3 default: {DEFAULT_ENT_COEF})"
        ),
    )
    parser.add_argument(
        "--gae-lambda",
        type=_parse_gae_lambda,
        default=DEFAULT_GAE_LAMBDA,
        help=(
            "PPO/RecurrentPPO GAE lambda, passed straight through to "
            "_make_shared_ppo; raising it lengthens the advantage estimator's "
            "horizon 1/(1-gamma*lambda) so a learner can see slower-payoff "
            "levers like expansion (the pre-registered arena recipe uses 0.98) "
            f"(SB3 default: {DEFAULT_GAE_LAMBDA}; must be in (0, 1])"
        ),
    )
    parser.add_argument(
        "--mask-expansion",
        action="store_true",
        help=(
            "S8 (/ row S8): mask the learner's illegal expansion logits (already-open/unaffordable regions, and slots beyond n_regions) to -inf before its argmax decode, so it can only ever make a legal choice (hold is always legal but is not itself a region); the frozen opponent is never masked. Applies to the learner's TRAINING rollouts only -- the post-training tournament below and every evaluation/scoring surface (the ladder, league snapshot/round-end evaluations, the arena) always roll out UNMASKED; see the summary JSON's training.mask_expansion_scope field. Recorded in training.mask_expansion. Default (off) is byte-identical to every prior run"
        ),
    )
    parser.add_argument(
        "--league-max-episode-steps",
        type=int,
        default=None,
        help=(
            "decision 8 (taken 2026-09-18): the league training-episode horizon, passed straight through to run_league_training(max_episode_steps=...) -> every LeagueEnvWrapper it builds. Default (unset -> None) never truncates -- one opponent per round, byte-identical to every prior run. A positive int truncates each round's game at that many ticks so the SuperSuit bridge auto-resets mid-round and draws a NEW opponent per EPISODE instead (the arena recipe uses 1000, the arena match horizon). Recorded in training.max_episode_steps; null under --resume (train_league skipped, mirrors --mask-expansion/--ent-coef/--gae-lambda)"
        ),
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help=(
            "owner decision 12a (taken 2026-09-19): round k>=2 continues from round "
            "(k-1)'s just-trained checkpoint's policy PARAMETERS instead of an "
            "independent from-scratch init, passed straight through to "
            "run_league_training(warm_start=...) -- see that function's own "
            "docstring for the mechanism (only the policy weights transfer; the "
            "optimizer stays fresh). Round 1 is unaffected either way -- it always "
            "starts from round 0's random init. Default (off) is byte-identical to "
            "every prior run. Recorded in training.warm_start; null under --resume "
            "(train_league skipped, mirrors --mask-expansion/--ent-coef/--gae-lambda)"
        ),
    )
    parser.add_argument(
        "--seed-archetypes",
        action="store_true",
        help=(
            "seed the pool with a fixed-opponent archetype roster alongside the default scripted seed, following --env (H4, addendum; RLB-8b): under --env calibrated (default), the three calibrated Phase 4 spike postures (lean, aggressor, patient); under --env arena, the four closed-loop geo postures (sprawler, fortress, cherry-picker, fast-follower) instead — the spike postures' fixed expansion_choice can only ever open region 1 there. Default behavior (off) is byte-identical to every prior run"
        ),
    )
    parser.add_argument(
        "--env",
        choices=["calibrated", "arena", "arena-balanced"],
        default="calibrated",
        help=(
            "which world train_league trains on and the tournament rates on "
            "(RLB-8a; credit-line-and-rebalance): 'calibrated' (default) is "
            "calibrated_spike_config, byte-identical to every prior run; 'arena' is "
            "the 25-region candidate-lock arena (harness.arena.arena_lock_config); "
            "'arena-balanced' is the same 25-region arena with a working-capital "
            "credit line + a smaller warehouse footprint (harness.arena."
            "arena_balanced_config) — both arena envs seat both seats via "
            "arena_home_regions(2). Ignored under --resume (the resumed pool's own "
            "recorded seating is authoritative, like --seed-archetypes already is)."
        ),
    )
    parser.add_argument(
        "--reward",
        choices=["profit", "weighted"],
        default=None,
        help=(
            "override the resolved --env's reward mode (core/reward.py; H5) for BOTH training and the tournament: 'profit' is raw profit, 'weighted' is the DESIGN-locked per-component z-normalized blend (RewardConfig's own weights). Default (unset) leaves the resolved env's own reward mode untouched -- calibrated/arena/arena-balanced all default to 'profit', byte-identical to every prior run, and a caller that already resolved a 'weighted' config (arena_train_player.py's monkey-patched env resolver) is never silently reset to 'profit' by this flag's own default. Applied AFTER --env resolves, so it always wins when given explicitly. Recorded in training.reward_mode (never null, even under --resume, since it also governs the tournament step)."
        ),
    )
    parser.add_argument(
        "--label",
        type=_parse_label,
        default=None,
        help=(
            "run label recorded in the summary JSON and appended to the default "
            "summary filename (phase4_rl_confirm_<date>_<label>.json); must match "
            "[A-Za-z0-9_-]+"
        ),
    )
    parser.add_argument(
        "--tourney-seeds",
        type=int,
        default=DEFAULT_TOURNEY_SEEDS,
        help="shared seed-set size K for the tournament",
    )
    parser.add_argument(
        "--tourney-episodes",
        type=int,
        default=DEFAULT_TOURNEY_EPISODES,
        help="episodes per head-to-head seed",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="bootstrap CI resamples",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="summary JSON path (default:)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
