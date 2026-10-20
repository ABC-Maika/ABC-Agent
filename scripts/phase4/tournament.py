"""Tournament + measurement: position-controlled head-to-head over distinct checkpoints,
non-transitivity metrics, and summary/reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np

from phase4.training import _materialize
from retail_simulator.core.config import CoreConfig
from retail_simulator.envs.ladder import _bootstrap_margin_ci
from retail_simulator.envs.spike import PairMargin, _measure_tournament, calibrated_spike_config

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
    from retail_simulator.envs.league import (
        _position_controlled_head_to_head,
    )
    from retail_simulator.envs.parallel_env import RetailParallelEnv

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
