"""Sweep data types: constants, row dataclasses, and shared scoring helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Obs index of the perceived competitor price (read from the schema, never hardcoded): the
# reactive research heuristic best-responds to this noised read.
_COMPETITOR_PRICE_OBS_NAME: str = "competitor_price"

# Default sweep resolution / horizon: a modest grid keeps the 2-D cell count affordable;
# ticks_per_point ticks per (cell, seed) average out the demand noise.
DEFAULT_GRID_POINTS: int = 11
DEFAULT_TICKS_PER_POINT: int = 1000
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)

# Marketing (spend_fraction) sweep range — the action bounds for the lever.
SPEND_LOW: float = 0.0
SPEND_HIGH: float = 1.0

# Assortment (breadth fraction) sweep range — the action bounds for the lever.
ASSORT_LOW: float = 0.0
ASSORT_HIGH: float = 1.0

# Promotion (intensity fraction) sweep range — the action bounds for the lever.
PROMO_LOW: float = 0.0
PROMO_HIGH: float = 1.0

PINNED_ASSORTMENT: float = 0.5

PINNED_PROMOTION: float = 0.3

# A clear optimum must beat the comparison cell by more than this multiple of the per-point
# standard error to count as a real (non-noise) win. ~3 sigma.
MARGIN_SIGMA_MULTIPLE: float = 3.0

# Floating-point tolerance for classifying a grid point as a boundary value.
_BOUNDARY_ATOL: float = 1e-9

SCORING_WINDOW_TICKS: int = 104

DEFAULT_EXPANSION_TIMINGS: tuple[int, ...] = (0, 10, 26, 52)

# Sentinel ``T`` for the never-expand baseline row (a policy that never opens).
NEVER_EXPAND_T: int = -1

# A timing curve is NON-flat only if the best T beats the WORST expand-at-T by > this multiple
# of their combined std-error (reuses the shared sigma multiple).
TIMING_FLAT_SIGMA_MULTIPLE: float = MARGIN_SIGMA_MULTIPLE

# Research (fidelity) sweep range — the research lever's action bounds. The axis maps a
# research SPEND in [0, 1] through the config's saturating fidelity mapping; the verdict
# reasons about the SPEND axis, reporting the resulting fidelity.
RESEARCH_LOW: float = 0.0
RESEARCH_HIGH: float = 1.0

DEFAULT_RESEARCH_SPENDS: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)

RESEARCH_OPPONENT_ARCHETYPE: str = "wandering"

RESEARCH_SECONDARY_OPPONENT_ARCHETYPE: str = "balanced"

RESEARCH_GATE_BETA_REFERENCE: float = 1.5

SERVICE_LOW: float = 0.0
SERVICE_HIGH: float = 1.0

# Default SCM-axis grid: 11 points over [0, 1]; the endpoints are load-bearing (the
# 'pinned at extreme' classification requires 0.0 and 1.0 in the grid).
DEFAULT_SERVICE_LEVELS: tuple[float, ...] = tuple(round(0.1 * i, 2) for i in range(11))

SCM_GATE_COGS_PREMIUM: float = 0.10
SCM_GATE_LOST_SALES_SHARE_PENALTY: float = 1.0
SCM_GATE_CONVENIENCE_BETA_SERVICE: float = 5.0


AUTOMATION_GATE_CAPEX_PER_TIER: tuple[float, float, float] = (0.0, 1500.0, 5000.0)
AUTOMATION_GATE_SAVINGS_PER_TIER: tuple[float, float, float] = (0.0, 0.10, 0.16)

DEFAULT_AUTOMATION_TIMINGS: tuple[int, ...] = (0, 5, 10, 20)

# Sentinel ``T`` for the never-upgrade baseline row (a policy that never upgrades).
NEVER_UPGRADE_T: int = -1

# Sentinel ``T`` for the ramp_0_to_1_to_2 row (encoded so the row carries a single
# integer policy field). Reserved value below NEVER_UPGRADE_T so it never collides.
RAMP_POLICY_T: int = -2

LOYALTY_LOW: float = 0.0
LOYALTY_HIGH: float = 1.0

# Default loyalty-spend grid: 11 points over [0, 1]; the endpoints are load-bearing (the
# 'pinned at extreme' classification requires 0.0 and 1.0).
DEFAULT_LOYALTY_SPENDS: tuple[float, ...] = tuple(round(0.1 * i, 2) for i in range(11))

LOYALTY_GATE_COST_PER_UNIT_SPEND: float = 25.0
LOYALTY_GATE_BRAND_LOYAL_BETA_PROGRAM: float = 6.0


@dataclass(frozen=True)
class SweepRow2D:
    """One grid cell: a fixed (price, spend_fraction) held for the whole horizon.

    ``mean_profit`` is the per-tick profit over post-burn-in ticks and seeds; ``std_error`` is its standard error (so a win must beat the noise floor); ``mean_market_share`` is a diagnostic; ``n_samples`` counts the per-tick samples.
    """

    price_index: float
    spend_fraction: float
    mean_profit: float
    std_error: float
    mean_market_share: float
    n_samples: int


@dataclass(frozen=True)
class SweepRow1D:
    """One assortment-axis cell: a fixed assortment at the pinned ``(p*, m*)`` (carried for provenance). Other fields as in :class:`SweepRow2D`."""

    assortment: float
    fixed_price: float
    fixed_spend: float
    mean_profit: float
    std_error: float
    mean_market_share: float
    n_samples: int


@dataclass(frozen=True)
class SweepRow1DPromotion:
    """One promotion-axis cell: a fixed promotion at the pinned ``(p*, m*, a*)``, measured at steady state (burn-in discards the transient). Other fields as in :class:`SweepRow2D`."""

    promotion: float
    fixed_price: float
    fixed_spend: float
    fixed_assortment: float
    mean_profit: float
    std_error: float
    mean_market_share: float
    n_samples: int


@dataclass(frozen=True)
class ExpansionTimingRow:
    """One expansion-timing policy: a single open-region-1 tick and its full-trajectory profit."""

    open_tick: int
    mean_profit: float
    std_error: float
    n_seeds: int

    @property
    def is_never_expand(self) -> bool:
        """True for the never-expand baseline row (open_tick == NEVER_EXPAND_T)."""
        return self.open_tick == NEVER_EXPAND_T


@dataclass(frozen=True)
class ResearchFidelityRow:
    """One research-spend cell: the spend, the resulting fidelity, and the mean score (profit net of research opex)."""

    research_spend: float
    fidelity: float
    mean_score: float
    std_error: float
    mean_market_share: float
    n_samples: int


@dataclass(frozen=True)
class SweepRow1DService:
    """One SCM service-level cell: a fixed service at the pinned ``(p*, m*, a*, promo*)``."""

    service_level: float
    fixed_price: float
    fixed_spend: float
    fixed_assortment: float
    fixed_promotion: float
    mean_profit: float
    std_error: float
    mean_market_share: float
    n_samples: int


@dataclass(frozen=True)
class SweepRow1DAutomation:
    """One automation upgrade-timing policy: label, target tier, tick(s), and its full-trajectory profit."""

    policy_name: str
    target_tier: int
    upgrade_tick: int
    ramp_tick: int
    fixed_price: float
    fixed_spend: float
    fixed_assortment: float
    fixed_promotion: float
    mean_profit: float
    std_error: float
    n_seeds: int

    @property
    def is_never_upgrade(self) -> bool:
        """True for the never-upgrade baseline row (``upgrade_tick == NEVER_UPGRADE_T``)."""
        return self.upgrade_tick == NEVER_UPGRADE_T


@dataclass(frozen=True)
class SweepRow1DLoyalty:
    """One loyalty-spend cell: a fixed loyalty_spend at the pinned ``(p*, m*, a*, promo*)``."""

    loyalty_spend: float
    fixed_price: float
    fixed_spend: float
    fixed_assortment: float
    fixed_promotion: float
    mean_profit: float
    std_error: float
    mean_market_share: float
    n_samples: int


@dataclass(frozen=True)
class LoyaltyVsPromotionOrthogonality:
    """The loyalty-vs-promotion substitution probe: base point, swap cells, and the pass verdict."""

    base_loyalty: float
    base_promotion: float
    base_profit: float
    swap_profits: tuple[tuple[float, float, float, float], ...]
    max_swap_delta: float
    threshold: float
    passed: bool
    reason: str


@dataclass(frozen=True)
class SweepResult:
    """All sweep legs in one: grid, axes, optima, and per-leg degeneracy verdicts."""

    rows: tuple[SweepRow2D, ...]
    assortment_rows: tuple[SweepRow1D, ...]
    promotion_rows: tuple[SweepRow1DPromotion, ...]
    profit_max_cell: tuple[float, float]
    profit_max_value: float
    assortment_max_value: float
    assortment_max_cell: float
    promotion_max_value: float
    is_degenerate: bool
    reason: str
    assortment_reason: str
    promotion_reason: str
    pinned_assortment: float
    pinned_promotion: float
    ticks_per_point: int
    burn_in_ticks: int
    seeds: tuple[int, ...]
    expansion_timing_rows: tuple[ExpansionTimingRow, ...] = ()
    expansion_timing_degenerate: bool | None = None
    expansion_timing_reason: str = ""
    start_present_both: bool = False
    research_rows: tuple[ResearchFidelityRow, ...] = ()
    research_degenerate: bool | None = None
    research_reason: str = ""
    service_rows: tuple[SweepRow1DService, ...] = ()
    service_variable_mean: float | None = None
    service_variable_std_error: float | None = None
    service_degenerate: bool | None = None
    service_reason: str = ""
    automation_rows: tuple[SweepRow1DAutomation, ...] = ()
    automation_degenerate: bool | None = None
    automation_reason: str = ""
    loyalty_rows: tuple[SweepRow1DLoyalty, ...] = ()
    loyalty_degenerate: bool | None = None
    loyalty_reason: str = ""
    loyalty_orthogonality: LoyaltyVsPromotionOrthogonality | None = None

    @property
    def passed(self) -> bool:
        """True when no leg is degenerate."""
        return not self.is_degenerate


def _is_boundary(value: float, low: float, high: float) -> bool:
    """True if ``value`` coincides with the range low or high boundary."""
    return math.isclose(value, low, abs_tol=_BOUNDARY_ATOL) or math.isclose(
        value, high, abs_tol=_BOUNDARY_ATOL
    )


def _beats_profit(
    best_profit: float,
    best_se: float,
    other_profit: float,
    other_se: float,
    margin_sigma_multiple: float,
) -> bool:
    """True if ``best`` beats ``other`` by > margin_sigma_multiple * combined SE (added in quadrature, so a noisy cell cannot fake or hide a win)."""
    combined_se = math.hypot(best_se, other_se)
    margin = best_profit - other_profit
    return margin > margin_sigma_multiple * combined_se


def _beats(best: SweepRow2D, other: SweepRow2D, margin_sigma_multiple: float) -> bool:
    """True if 2-D cell ``best`` beats ``other`` by > margin * combined SE."""
    return _beats_profit(
        best.mean_profit, best.std_error, other.mean_profit, other.std_error, margin_sigma_multiple
    )


def _mean_and_std_error(profits: list[float]) -> tuple[float, float, int]:
    """Compute (mean, std_error, n) from a list of per-tick profit samples."""
    arr = np.asarray(profits, dtype=np.float64)
    n = len(arr)
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / (n ** 0.5)) if n > 1 else 0.0
    return mean, se, n
