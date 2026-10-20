"""The named 25-region arena ``CoreConfig``, composed on the calibrated 2-region point.

Public names are prefixed ``arena_`` (not ``competition_*``) to avoid colliding with the established :func:`~retail_simulator.harness.parallel_gate.competition_config`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

from retail_simulator.core.config import CoreConfig, RegionConfig
from retail_simulator.core.schema import N_REGIONS_MAX
from retail_simulator.core.state import (
    CONVENIENCE,
    PRICE_SENSITIVE,
    QUALITY_SEEKING,
)
from retail_simulator.harness.spike import calibrated_spike_config

if TYPE_CHECKING:  # type-only; never pulled in at runtime import time
    from retail_simulator.envs.parallel_env import RetailParallelEnv

ARENA_N_REGIONS: int = 25


@dataclass(frozen=True)
class ArenaRegionClass:
    """One region class: name, base demand, segment mix."""

    name: str
    base_regional_demand: float
    segment_mix: dict[str, float]


def _swap(mix: dict[str, float], key_a: str, key_b: str) -> dict[str, float]:
    """Return a copy of ``mix`` with ``key_a``'s and ``key_b``'s shares exchanged.

    Swapping two shares never changes the sum, so a valid (sums-to-1.0) input mix always produces a valid output mix — no re-normalization needed.
    """
    swapped = dict(mix)
    swapped[key_a], swapped[key_b] = mix[key_b], mix[key_a]
    return swapped


def _build_arena_region_classes() -> tuple[ArenaRegionClass, ...]:
    """Build the four region classes in fixed cycle order (METRO/AFFLUENT/LOYAL/FRONTIER).

    METRO and LOYAL use the calibrated config's region-0 / region-1 ``segment_mix`` VERBATIM (never hardcoded, so they track the lock); AFFLUENT and FRONTIER are derived from the METRO mix by promoting a different segment to the heaviest share (:func:`_swap`). Order matters: index 0 (METRO) is the class home regions are drawn from; index -1 (FRONTIER, the smallest) is the class stranded at region 0.
    """
    calibrated_regions = calibrated_spike_config().demand.regions
    # Defensive copies: never share a mutable dict with the calibrated config's
    # (frozen-dataclass, but dict-valued) ``segment_mix``.
    metro_mix = dict(calibrated_regions[0].segment_mix)
    loyal_mix = dict(calibrated_regions[1].segment_mix)
    return (
        ArenaRegionClass(name="METRO", base_regional_demand=1400.0, segment_mix=metro_mix),
        ArenaRegionClass(
            name="AFFLUENT",
            base_regional_demand=900.0,
            segment_mix=_swap(metro_mix, PRICE_SENSITIVE, QUALITY_SEEKING),
        ),
        ArenaRegionClass(name="LOYAL", base_regional_demand=600.0, segment_mix=loyal_mix),
        ArenaRegionClass(
            name="FRONTIER",
            base_regional_demand=350.0,
            segment_mix=_swap(metro_mix, PRICE_SENSITIVE, CONVENIENCE),
        ),
    )


# Computed once at import time (pure; never mutates the calibrated lock); every
# ``arena_regions`` call reuses this instead of rebuilding it.
ARENA_REGION_CLASSES: tuple[ArenaRegionClass, ...] = _build_arena_region_classes()


def arena_regions(n_regions: int = ARENA_N_REGIONS) -> tuple[RegionConfig, ...]:
    """Build the ``n_regions``-tuple of deterministic per-region demand configs.

    Region 0 is deliberately STRANDED: present and observable but never enterable (``choice=0`` is the expansion lever's permanent hold/no-op slot), using the smallest class (FRONTIER). Regions 1..n_regions-1 cycle :data:`ARENA_REGION_CLASSES` as ``(r - 1) mod 4``; at the default 25 regions that is six enterable instances of each class plus the stranded region. Purely a function of config data: no RNG draw.
    """
    if not (1 <= n_regions <= N_REGIONS_MAX):
        raise ValueError(
            f"n_regions={n_regions} must be in [1, {N_REGIONS_MAX}] (the expansion "
            f"lever's fixed capacity, N_REGIONS_MAX={N_REGIONS_MAX})"
        )
    n_classes = len(ARENA_REGION_CLASSES)
    stranded = ARENA_REGION_CLASSES[-1]
    class_sequence = [stranded] + [
        ARENA_REGION_CLASSES[(r - 1) % n_classes] for r in range(1, n_regions)
    ]
    return tuple(
        RegionConfig(
            base_regional_demand=region_class.base_regional_demand,
            # A fresh dict per region: several regions share a CLASS, never a mutable dict.
            segment_mix=dict(region_class.segment_mix),
        )
        for region_class in class_sequence
    )


def arena_config(
    n_regions: int = ARENA_N_REGIONS,
    *,
    overhead_reference_stores: float = 3.8,
    overhead_exponent: float | None = None,
    provided_capacity_per_store: float = 700.0,
) -> CoreConfig:
    """The named arena config: the calibrated 2-region economics + the arena region table + wage/warehouse knobs."""
    base = calibrated_spike_config()
    wage = replace(base.wage, overhead_reference_stores=overhead_reference_stores)
    if overhead_exponent is not None:
        wage = replace(wage, overhead_exponent=overhead_exponent)
    return replace(
        base,
        demand=replace(base.demand, regions=arena_regions(n_regions)),
        wage=wage,
        warehouse=replace(base.warehouse, provided_capacity_per_store=provided_capacity_per_store),
    )


def arena_home_regions(n_seats: int, *, n_regions: int = ARENA_N_REGIONS) -> tuple[int, ...]:
    """The ``n_seats`` distinct METRO home regions (region ``1 + 4*i`` per seat).

    Raises if ``n_regions`` cannot supply one METRO instance per seat.
    """
    if n_seats < 1:
        raise ValueError(f"n_seats={n_seats} must be >= 1")
    if not (1 <= n_regions <= N_REGIONS_MAX):
        raise ValueError(
            f"n_regions={n_regions} must be in [1, {N_REGIONS_MAX}] (the expansion "
            f"lever's fixed capacity, N_REGIONS_MAX={N_REGIONS_MAX})"
        )
    n_classes = len(ARENA_REGION_CLASSES)
    last_home = 1 + n_classes * (n_seats - 1)
    if last_home >= n_regions:
        # The exact count of METRO instances at this n_regions — the same cycle
        # arena_regions builds, so this can never drift from reality.
        n_metro_instances = len(range(1, n_regions, n_classes))
        raise ValueError(
            f"n_seats={n_seats} needs {n_seats} distinct METRO-class home regions "
            f"(region 1 + {n_classes}*i per seat — region 0 is the stranded no-op "
            f"slot) but n_regions={n_regions} only provides {n_metro_instances} "
            f"METRO instances"
        )
    return tuple(1 + n_classes * i for i in range(n_seats))


def arena_env(
    n_seats: int = 2,
    *,
    n_regions: int = ARENA_N_REGIONS,
    seed: int | None = None,
    **knobs: float,
) -> RetailParallelEnv:
    """Build a :class:`RetailParallelEnv` on the arena config with ``n_seats`` learning agents."""
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    return RetailParallelEnv(
        arena_config(n_regions, **knobs),
        n_learning_agents=n_seats,
        home_regions=arena_home_regions(n_seats, n_regions=n_regions),
        seed=seed,
    )


def arena_region_table_hash(config: CoreConfig) -> str:
    """Stable sha256 of the region table (demand + mix only) — the env's identity for reports."""
    payload = [
        {"base_regional_demand": region.base_regional_demand, "segment_mix": region.segment_mix}
        for region in config.demand.regions
    ]
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


ARENA_LOCK_OVERHEAD_REFERENCE_STORES: Final[float] = 7.15
ARENA_LOCK_OVERHEAD_EXPONENT: Final[float] = 1.5
ARENA_LOCK_PROVIDED_CAPACITY_PER_STORE: Final[float] = 700.0


def arena_lock_config() -> CoreConfig:
    """The named arena at the Phase-6 lock — the three ``ARENA_LOCK_*`` constants.

    Every other knob is inherited from :func:`arena_config`'s defaults. This is the config the league driver and the ladder train/rate on (see :data:`ARENA_ENV_LABEL`).
    """
    return arena_config(
        overhead_reference_stores=ARENA_LOCK_OVERHEAD_REFERENCE_STORES,
        overhead_exponent=ARENA_LOCK_OVERHEAD_EXPONENT,
        provided_capacity_per_store=ARENA_LOCK_PROVIDED_CAPACITY_PER_STORE,
    )


def _arena_env_label(
    *,
    overhead_reference_stores: float,
    overhead_exponent: float,
    provided_capacity_per_store: float,
) -> str:
    """The env label string for the arena candidate lock."""
    return (
        f"arena ({ARENA_N_REGIONS}-region candidate lock "
        f"s={overhead_reference_stores:g} e={overhead_exponent:g} "
        f"cap={provided_capacity_per_store:g})"
    )


ARENA_ENV_LABEL: str = _arena_env_label(
    overhead_reference_stores=ARENA_LOCK_OVERHEAD_REFERENCE_STORES,
    overhead_exponent=ARENA_LOCK_OVERHEAD_EXPONENT,
    provided_capacity_per_store=ARENA_LOCK_PROVIDED_CAPACITY_PER_STORE,
)


ARENA_BALANCED_PROVIDED_CAPACITY_PER_STORE: Final[float] = 300.0
ARENA_BALANCED_CREDIT_LIMIT: Final[float] = 5000.0
ARENA_BALANCED_OVERDRAFT_RATE: Final[float] = 0.03
#
# Explicit, named override of the lock-inherited writeoff ceiling: an amendment to the
# lock is caught here instead of silently drifting the balanced world, and keeps
# ARENA_BALANCED_ENV_LABEL (and therefore the default ladder_id) moving whenever this
# knob does. Every other balanced knob stays as the lock has it.
ARENA_BALANCED_WRITEOFF_FRAC_MAX: Final[float] = 0.05

#
ARENA_BALANCED_OVERHEAD_REFERENCE_STORES: Final[float] = 4.5

ARENA_BALANCED_RESTRUCTURE_AFTER_TICKS: Final[int] = 50
ARENA_BALANCED_RESTRUCTURE_KEEP_STORES: Final[int] = 3
ARENA_BALANCED_RESTRUCTURE_DEBT_THRESHOLD: Final[float] = 5000.0

ARENA_BALANCED_MAX_SHARE_PER_REGION: Final[float] = 0.6

ARENA_BALANCED_PROMOTION_CONTEST_WEIGHT: Final[float] = 1.0
# Per-unit promotion discount cost, dropped to 0.0 (class default is 0.009).
ARENA_BALANCED_PROMO_COST_PER_UNIT: Final[float] = 0.0
# Stockpile build rate, dropped to 0.0 (class default is 0.17); with build 0.0 the seated
# stockpile is always 0, so ``stockpile_decay`` stays at its class default (moot).
ARENA_BALANCED_STOCKPILE_BUILD: Final[float] = 0.0


def arena_balanced_config() -> CoreConfig:
    """The arena-balanced config: the lock with credit/writeoff/restructure, share-cap, and promotion rebalance knobs applied."""
    base = arena_lock_config()
    return replace(
        base,
        warehouse=replace(
            base.warehouse,
            provided_capacity_per_store=ARENA_BALANCED_PROVIDED_CAPACITY_PER_STORE,
        ),
        wage=replace(
            base.wage,
            credit_limit=ARENA_BALANCED_CREDIT_LIMIT,
            overdraft_rate=ARENA_BALANCED_OVERDRAFT_RATE,
            writeoff_frac_max=ARENA_BALANCED_WRITEOFF_FRAC_MAX,
            overhead_reference_stores=ARENA_BALANCED_OVERHEAD_REFERENCE_STORES,
            restructure_after_ticks=ARENA_BALANCED_RESTRUCTURE_AFTER_TICKS,
            restructure_keep_stores=ARENA_BALANCED_RESTRUCTURE_KEEP_STORES,
            restructure_debt_threshold=ARENA_BALANCED_RESTRUCTURE_DEBT_THRESHOLD,
        ),
        demand=replace(
            base.demand,
            max_share_per_region=ARENA_BALANCED_MAX_SHARE_PER_REGION,
        ),
        promotion=replace(
            base.promotion,
            promotion_contest_weight=ARENA_BALANCED_PROMOTION_CONTEST_WEIGHT,
            promo_cost_per_unit=ARENA_BALANCED_PROMO_COST_PER_UNIT,
            stockpile_build=ARENA_BALANCED_STOCKPILE_BUILD,
        ),
    )


def _arena_balanced_env_label(
    *,
    provided_capacity_per_store: float,
    credit_limit: float,
    overdraft_rate: float,
    writeoff_frac_max: float,
    overhead_reference_stores: float,
    restructure_after_ticks: int,
    restructure_keep_stores: int,
    restructure_debt_threshold: float,
    max_share_per_region: float,
    promotion_contest_weight: float,
    promo_cost_per_unit: float,
    stockpile_build: float,
) -> str:
    """The env label string for the arena-balanced config."""
    label = (
        f"arena-balanced ({ARENA_N_REGIONS}-region rebalance "
        f"cap={provided_capacity_per_store:g} credit={credit_limit:g} "
        f"overdraft={overdraft_rate:g} writeoff_max={writeoff_frac_max:g} "
        f"s={overhead_reference_stores:g} "
        f"restructure={restructure_after_ticks:g}/{restructure_keep_stores:g}/"
        f"{restructure_debt_threshold:g} "
        f"promo_contest={promotion_contest_weight:g} "
        f"promo_cost={promo_cost_per_unit:g} "
        f"stockpile_build={stockpile_build:g})"
    )
    if max_share_per_region < 1.0:
        label = f"{label[:-1]} share_cap={max_share_per_region:g})"
    return label


ARENA_BALANCED_ENV_LABEL: str = _arena_balanced_env_label(
    provided_capacity_per_store=ARENA_BALANCED_PROVIDED_CAPACITY_PER_STORE,
    credit_limit=ARENA_BALANCED_CREDIT_LIMIT,
    overdraft_rate=ARENA_BALANCED_OVERDRAFT_RATE,
    writeoff_frac_max=ARENA_BALANCED_WRITEOFF_FRAC_MAX,
    overhead_reference_stores=ARENA_BALANCED_OVERHEAD_REFERENCE_STORES,
    restructure_after_ticks=ARENA_BALANCED_RESTRUCTURE_AFTER_TICKS,
    restructure_keep_stores=ARENA_BALANCED_RESTRUCTURE_KEEP_STORES,
    restructure_debt_threshold=ARENA_BALANCED_RESTRUCTURE_DEBT_THRESHOLD,
    max_share_per_region=ARENA_BALANCED_MAX_SHARE_PER_REGION,
    promotion_contest_weight=ARENA_BALANCED_PROMOTION_CONTEST_WEIGHT,
    promo_cost_per_unit=ARENA_BALANCED_PROMO_COST_PER_UNIT,
    stockpile_build=ARENA_BALANCED_STOCKPILE_BUILD,
)
