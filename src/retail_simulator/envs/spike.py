"""Archetype spike: measure the intended LEAN → AGGRESSOR → PATIENT cycle.

Pairwise head-to-head rollouts between scripted archetypes, with the
dominance/cycle read and bootstrap CIs (reused from ``ladder``). Heavy
env/league imports stay lazy; stats are hand-rolled numpy (no scipy).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import combinations
from math import sqrt
from typing import Any

import numpy as np

from retail_simulator.core.config import CoreConfig, SupplierConfig, WageConfig, WarehouseConfig
from retail_simulator.envs.ladder import _bootstrap_margin_ci
from retail_simulator.envs.spike_archetypes import (
    AGGRESSOR,
    LEAN,
    PATIENT,
    ArchetypePredict,
    default_archetypes,
)

logger = logging.getLogger(__name__)

__all__ = [
    "INTENDED_CYCLE_DIRECTION",
    "PairMargin",
    "SpikeResult",
    "calibrated_spike_config",
    "run_spike",
    "spike_config",
]

INTENDED_CYCLE_DIRECTION: tuple[str, str, str] = (LEAN, AGGRESSOR, PATIENT)

# --- spike_config: the spike's CoreConfig builder (seed values) ---

_SEED_WAGE_BASE: float = 50.0
_SEED_WAGE_OWN_SHARE_COEFF: float = 200.0
_SEED_WAGE_COUPLING_COEFF: float = 0.5
_SEED_WRITEOFF_SEVERITY_SCALE: float = 1.0
_SEED_WRITEOFF_FRAC_MIN: float = 0.0
_SEED_WRITEOFF_FRAC_MAX: float = 0.05  # bounded %-of-turnover write-off
_SEED_LEADER_WEIGHT_EXPONENT: float = 1.5  # > 1 taxes the biggest more (leader-weighted)
_SEED_HAPPINESS_DECAY: float = 0.1
_SEED_OVERHEAD_THRESHOLD_STORES: float = 1.0
_SEED_OVERHEAD_SUPERLINEAR_COEFF: float = 100.0
_SEED_OVERHEAD_EXPONENT: float = 1.5  # super-linear


def spike_config(base: CoreConfig | None = None, **overrides: object) -> CoreConfig:
    """Build the spike's feature-ON CoreConfig from the seed values.

    ``overrides`` replace individual wage fields on top.
    """
    resolved_base = base if base is not None else CoreConfig.default()
    wage_kwargs: dict[str, object] = {
        "cash_budget_enabled": True,
        "wage_base": _SEED_WAGE_BASE,
        "wage_own_share_coeff": _SEED_WAGE_OWN_SHARE_COEFF,
        "wage_coupling_coeff": _SEED_WAGE_COUPLING_COEFF,
        "writeoff_severity_scale": _SEED_WRITEOFF_SEVERITY_SCALE,
        "writeoff_frac_min": _SEED_WRITEOFF_FRAC_MIN,
        "writeoff_frac_max": _SEED_WRITEOFF_FRAC_MAX,
        "leader_weight_exponent": _SEED_LEADER_WEIGHT_EXPONENT,
        "happiness_decay": _SEED_HAPPINESS_DECAY,
        "overhead_threshold_stores": _SEED_OVERHEAD_THRESHOLD_STORES,
        "overhead_superlinear_coeff": _SEED_OVERHEAD_SUPERLINEAR_COEFF,
        "overhead_exponent": _SEED_OVERHEAD_EXPONENT,
    }
    # Overrides win over the seed values; an unknown key raises TypeError from the
    # WageConfig constructor.
    wage_kwargs.update(overrides)
    return replace(resolved_base, wage=WageConfig(**wage_kwargs))  # type: ignore[arg-type]


# Calibrated operating-point overrides (distinct from the seed values above).
_CALIBRATED_WAGE_OWN_SHARE_COEFF: float = 150.0
_CALIBRATED_OVERHEAD_SUPERLINEAR_COEFF: float = 380.0
_CALIBRATED_COGS_PREMIUM: float = 0.10
_CALIBRATED_FILL_RATE_FLOOR: float = 0.5
_CALIBRATED_SUPPLIER_CAPACITY: float = 1200.0
_CALIBRATED_PREFERENCE_SHARPNESS: float = 3.0
_CALIBRATED_WAREHOUSE_CAPACITY: float = 1450.0
_CALIBRATED_WAGE_SPEND_SCALE: float = 200.0
_CALIBRATED_WAREHOUSE_CAPACITY_PER_UNIT_INVEST: float = 100.0
_CALIBRATED_WAREHOUSE_COST_PER_UNIT_INVEST: float = 200.0


def calibrated_spike_config() -> CoreConfig:
    """Build the calibrated spike CoreConfig on top of :func:`spike_config`."""
    base = spike_config(
        wage_own_share_coeff=_CALIBRATED_WAGE_OWN_SHARE_COEFF,
        overhead_superlinear_coeff=_CALIBRATED_OVERHEAD_SUPERLINEAR_COEFF,
        wage_spend_scale=_CALIBRATED_WAGE_SPEND_SCALE,
    )
    return replace(
        base,
        scm=replace(
            base.scm,
            cogs_premium=_CALIBRATED_COGS_PREMIUM,
            fill_rate_floor=_CALIBRATED_FILL_RATE_FLOOR,
        ),
        supplier=SupplierConfig(
            supplier_contention_enabled=True,
            supplier_capacity=_CALIBRATED_SUPPLIER_CAPACITY,
            preference_sharpness=_CALIBRATED_PREFERENCE_SHARPNESS,
        ),
        warehouse=WarehouseConfig(
            throughput_enabled=True,
            default_warehouse_capacity=_CALIBRATED_WAREHOUSE_CAPACITY,
            capacity_per_unit_invest=_CALIBRATED_WAREHOUSE_CAPACITY_PER_UNIT_INVEST,
            cost_per_unit_invest=_CALIBRATED_WAREHOUSE_COST_PER_UNIT_INVEST,
        ),
    )


# --- SpikeResult and the per-pair margin record ---


@dataclass(frozen=True)
class PairMargin:
    """One pair's margin (A over B) per seed, with mean, standard error, and bootstrap CI.

    ``label_a < label_b``; ``per_seed_margins`` is ``mean(scores_a) − mean(scores_b)``
    per seed; ``ci_low``/``ci_high`` are the bootstrap 95% CI on the mean margin
    (seed-axis resample); ``margin_se`` uses the population sd (``ddof=0``).
    """

    label_a: str
    label_b: str
    per_seed_margins: tuple[float, ...]
    mean_margin: float
    margin_se: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class SpikeResult:
    """Outcome of one spike run: pairwise margins plus the dominance/cycle read."""

    n_seeds: int
    seed_set: tuple[int, ...]
    n_episodes: int
    archetype_labels: tuple[str, ...]
    pairwise: tuple[PairMargin, ...]
    dominance: dict[str, tuple[str, ...]]
    n_cyclic_triples: int
    has_condorcet_winner: bool
    condorcet_winner: str | None
    cycle_direction: tuple[str, ...] | None
    decisive: bool
    matches_intended_direction: bool
    success: bool
    bootstrap_resamples: int
    bootstrap_seed: int


# --- cycle measurement (pure: operates on collected per-seed margins) ---


def _triple_cycles(
    triple: tuple[str, str, str],
    beats: dict[tuple[str, str], bool],
) -> tuple[str, str, str] | None:
    """Return the cyclic order ``(a, b, c)`` of a 3-node sub-tournament, or ``None`` if transitive.

    ``beats[(x, y)]`` is True iff x beats y.
    """
    a, b, c = triple
    # out-degree of each node within the triple
    out = {node: 0 for node in triple}
    for x, y in ((a, b), (a, c), (b, c)):
        if beats[(x, y)]:
            out[x] += 1
        else:
            out[y] += 1
    # Transitive iff some node beats both others; otherwise it is a 3-cycle.
    if any(o == 2 for o in out.values()):
        return None

    # In a 3-cycle every node has exactly one out-edge; follow it to recover the order.
    def _beaten_by(node: str) -> str:
        others = [n for n in triple if n != node]
        x, y = others
        return x if beats[(node, x)] else y

    start = a
    second = _beaten_by(start)
    third = _beaten_by(second)
    return (start, second, third)


def _same_cycle_up_to_rotation(realized: tuple[str, ...], intended: tuple[str, str, str]) -> bool:
    """Whether two length-3 cyclic orders are the same directed cycle (rotation, not reflection)."""
    if len(realized) != 3:
        return False
    rotations = {
        (intended[0], intended[1], intended[2]),
        (intended[1], intended[2], intended[0]),
        (intended[2], intended[0], intended[1]),
    }
    return tuple(realized) in rotations


def _measure_tournament(
    pairwise: tuple[PairMargin, ...],
    labels: tuple[str, ...],
) -> tuple[
    dict[str, tuple[str, ...]],
    int,
    bool,
    str | None,
    tuple[str, ...] | None,
    bool,
]:
    """Build the dominance graph and cycle stats from pairwise margins.

    Returns ``(dominance, n_cyclic_triples, has_condorcet_winner, condorcet_winner,
    cycle_direction, decisive)``; ``decisive`` is True iff every edge of the realized
    cycle has a bootstrap CI excluding 0.
    """
    # beats[(x, y)]: x beats y iff the mean margin is positive; build both orientations.
    beats: dict[tuple[str, str], bool] = {}
    ci_excludes_zero: dict[frozenset[str], bool] = {}
    margin_by_pair: dict[tuple[str, str], float] = {}
    for pm in pairwise:
        beats[(pm.label_a, pm.label_b)] = pm.mean_margin > 0.0
        beats[(pm.label_b, pm.label_a)] = pm.mean_margin < 0.0
        margin_by_pair[(pm.label_a, pm.label_b)] = pm.mean_margin
        # A CI strictly above or below 0 is a decisive edge.
        ci_excludes_zero[frozenset((pm.label_a, pm.label_b))] = pm.ci_low > 0.0 or pm.ci_high < 0.0

    dominance: dict[str, tuple[str, ...]] = {
        node: tuple(other for other in labels if other != node and beats[(node, other)])
        for node in labels
    }

    # A Condorcet winner beats EVERY other node.
    condorcet_winner: str | None = None
    for node in labels:
        if len(dominance[node]) == len(labels) - 1:
            condorcet_winner = node
            break
    has_condorcet_winner = condorcet_winner is not None

    # Count cyclic triples; capture the first realized direction and its CI-decisiveness.
    n_cyclic_triples = 0
    cycle_direction: tuple[str, ...] | None = None
    decisive = False
    for triple in combinations(labels, 3):
        order = _triple_cycles(triple, beats)
        if order is None:
            continue
        n_cyclic_triples += 1
        if cycle_direction is None:
            cycle_direction = order
            edges = (
                frozenset((order[0], order[1])),
                frozenset((order[1], order[2])),
                frozenset((order[2], order[0])),
            )
            decisive = all(ci_excludes_zero[edge] for edge in edges)

    return (
        dominance,
        n_cyclic_triples,
        has_condorcet_winner,
        condorcet_winner,
        cycle_direction,
        decisive,
    )


# --- default real rollout seam ---


def _default_head_to_head(
    config: CoreConfig,
    *,
    env_factory: Callable[[], Any] | None = None,
) -> Callable[..., tuple[list[float], list[float]]]:
    """Build the default head-to-head seam over a ``RetailParallelEnv``.

    Returns a ``head_to_head(predict_a, predict_b, *, n_episodes, seed)`` closure over
    the league's position-controlled primitive, on one shared env; the primitive re-seeds
    the world per call, so a single env is deterministic across all pairs and seeds.
    With ``env_factory``, a fresh env is built (and closed) per call instead, letting the
    caller pass extras like ``home_regions``. The closure's ``close`` attribute releases
    the shared env; it is ``None`` when ``env_factory`` is given. Heavy imports stay lazy.
    """
    from retail_simulator.envs.league import (  # noqa: PLC0415
        _position_controlled_head_to_head,
    )

    if env_factory is not None:

        def head_to_head_via_factory(
            predict_a: ArchetypePredict,
            predict_b: ArchetypePredict,
            *,
            n_episodes: int,
            seed: int,
        ) -> tuple[list[float], list[float]]:
            return _position_controlled_head_to_head(
                None,
                config.reward,
                predict_a=predict_a,
                predict_b=predict_b,
                n_episodes=n_episodes,
                seed=seed,
                env_factory=env_factory,
            )

        return head_to_head_via_factory

    from retail_simulator.envs.parallel_env import RetailParallelEnv  # noqa: PLC0415

    env = RetailParallelEnv(config, n_learning_agents=2, seed=0)

    def head_to_head(
        predict_a: ArchetypePredict,
        predict_b: ArchetypePredict,
        *,
        n_episodes: int,
        seed: int,
    ) -> tuple[list[float], list[float]]:
        return _position_controlled_head_to_head(
            env,
            config.reward,
            predict_a=predict_a,
            predict_b=predict_b,
            n_episodes=n_episodes,
            seed=seed,
        )

    # Expose the env so run_spike can close it after the rollouts.
    head_to_head.close = env.close  # type: ignore[attr-defined]
    return head_to_head


def run_spike(
    *,
    archetypes: dict[str, ArchetypePredict] | None = None,
    config: CoreConfig | None = None,
    n_seeds: int = 30,
    seed_offset: int = 0,
    n_episodes: int = 1,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 0,
    head_to_head: Callable[..., tuple[list[float], list[float]]] | None = None,
    env_factory: Callable[[], Any] | None = None,
) -> SpikeResult:
    """Roll out every archetype pair over the seed set and report the cycle verdict."""
    roster = archetypes if archetypes is not None else default_archetypes()
    if len(roster) < 3:
        raise ValueError(
            f"run_spike requires at least 3 archetypes (a tournament needs a triple); "
            f"got {len(roster)}"
        )
    if n_seeds < 1:
        raise ValueError(f"run_spike requires n_seeds >= 1; got {n_seeds}")
    if n_episodes < 1:
        raise ValueError(f"run_spike requires n_episodes >= 1; got {n_episodes}")
    if seed_offset < 0:
        raise ValueError(f"run_spike requires seed_offset >= 0; got {seed_offset}")

    effective_config = config if config is not None else spike_config()
    seed_set = tuple(range(seed_offset, seed_offset + n_seeds))
    labels = tuple(sorted(roster))

    # Select the rollout seam; the default seam owns an env to close after the rollouts.
    owns_env_close: Callable[[], None] | None = None
    if head_to_head is None:
        real_seam = _default_head_to_head(effective_config, env_factory=env_factory)
        owns_env_close = getattr(real_seam, "close", None)
        seam = real_seam
    else:
        seam = head_to_head

    try:
        # Collect per-seed margins for every unordered pair (label_a < label_b).
        per_seed_by_pair: dict[tuple[str, str], list[float]] = {}
        for label_a, label_b in combinations(labels, 2):
            per_seed: list[float] = []
            for seed in seed_set:
                scores_a, scores_b = seam(
                    roster[label_a],
                    roster[label_b],
                    n_episodes=n_episodes,
                    seed=seed,
                )
                per_seed.append(float(np.mean(scores_a)) - float(np.mean(scores_b)))
            per_seed_by_pair[(label_a, label_b)] = per_seed
    finally:
        if owns_env_close is not None:
            owns_env_close()

    # Draw the seed-index resamples once (independent of rollout seeds) and apply them
    # consistently to every pair; each pair is its own 2-node field, so the ladder
    # bootstrap yields the CI on that pair's mean margin.
    rng = np.random.default_rng(bootstrap_seed)
    resampled_seed_indices = rng.integers(0, n_seeds, size=(bootstrap_resamples, n_seeds))
    pair_per_seed: dict[tuple[str, str], np.ndarray] = {
        pair: np.asarray(per_seed, dtype=np.float64) for pair, per_seed in per_seed_by_pair.items()
    }

    pairwise: list[PairMargin] = []
    for label_a, label_b in combinations(labels, 2):
        per_seed_arr = pair_per_seed[(label_a, label_b)]
        mean_margin = float(per_seed_arr.mean())
        margin_se = float(np.std(per_seed_arr, ddof=0)) / sqrt(n_seeds)
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
                per_seed_margins=tuple(float(v) for v in per_seed_arr),
                mean_margin=mean_margin,
                margin_se=margin_se,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )

    pairwise_tuple = tuple(pairwise)
    (
        dominance,
        n_cyclic_triples,
        has_condorcet_winner,
        condorcet_winner,
        cycle_direction,
        decisive,
    ) = _measure_tournament(pairwise_tuple, labels)

    matches_intended_direction = cycle_direction is not None and _same_cycle_up_to_rotation(
        cycle_direction, INTENDED_CYCLE_DIRECTION
    )
    success = (
        decisive
        and not has_condorcet_winner
        and n_cyclic_triples >= 1
        and matches_intended_direction
    )

    return SpikeResult(
        n_seeds=n_seeds,
        seed_set=seed_set,
        n_episodes=n_episodes,
        archetype_labels=labels,
        pairwise=pairwise_tuple,
        dominance=dominance,
        n_cyclic_triples=n_cyclic_triples,
        has_condorcet_winner=has_condorcet_winner,
        condorcet_winner=condorcet_winner,
        cycle_direction=cycle_direction,
        decisive=decisive,
        matches_intended_direction=matches_intended_direction,
        success=success,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
