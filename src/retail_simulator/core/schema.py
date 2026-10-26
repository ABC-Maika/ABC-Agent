"""Versioned, typed action/observation contracts for the sim core.

This module is the single source of truth for *what* an action and an
observation contain; ``encoding.py`` owns *how* they are laid out in a flat
vector. New dimensions are appended at the tail so existing offsets never
move.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cache, lru_cache

# Bump when the action layout changes in a way agents must be aware of (even pure tail appends).
#
ACTION_SCHEMA_VERSION: int = 12

N_REGIONS: int = 2

N_REGIONS_MAX: int = 32

# Number of automation tiers; the automation lever's width must equal this
# (enforced by validate_automation_width).
N_AUTOMATION_TIERS: int = 3


class LeverKind(Enum):
    """Whether a lever is a bounded scalar or a finite choice."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


@dataclass(frozen=True)
class LeverSpec:
    """One controllable lever in the action space.

    ``low``/``high`` apply to CONTINUOUS levers, ``n`` to DISCRETE levers.
    ``enabled`` gates whether the lever contributes dimensions to the flat layout.
    """

    name: str
    kind: LeverKind
    default: float
    introduced_phase: str
    enabled: bool
    low: float | None = None
    high: float | None = None
    n: int | None = None

    @property
    def width(self) -> int:
        """Number of flat dimensions this lever occupies when enabled."""
        if self.kind is LeverKind.CONTINUOUS:
            return 1
        if self.n is None:
            raise ValueError(f"discrete lever {self.name!r} must define n")
        # Discrete levers are encoded as a masked logits block (argmax-decoded).
        return self.n


# Ordered registry; order is the canonical flat-layout order. Append new levers
# at the tail — never reorder existing entries.
LEVER_REGISTRY: tuple[LeverSpec, ...] = (
    LeverSpec(
        name="price_index",
        kind=LeverKind.CONTINUOUS,
        low=0.5,
        high=2.0,
        default=1.0,
        introduced_phase="0.0",
        enabled=True,
    ),
    # Marketing spend fraction in [0, 1] (lagged investment).
    LeverSpec(
        name="marketing",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=0.0,
        introduced_phase="0.1",
        enabled=True,
    ),
    # Assortment breadth fraction in [0, 1] (0.0 = narrowest).
    LeverSpec(
        name="assortment",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=0.0,
        introduced_phase="0.2",
        enabled=True,
    ),
    # Promo intensity in [0, 1] (0.0 = none); demand lift + margin cost +
    # seated consumer stockpile.
    LeverSpec(
        name="promotion",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=0.0,
        introduced_phase="0.3",
        enabled=True,
    ),
    # DISCRETE: index 0 = no-op, index r = open region r (open-only, cash-gated, masked).
    LeverSpec(
        name="expansion",
        kind=LeverKind.DISCRETE,
        n=N_REGIONS_MAX,
        default=0.0,
        introduced_phase="0.4",
        enabled=True,
    ),
    LeverSpec(
        name="research",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=0.0,
        introduced_phase="1.0",
        enabled=True,
    ),
    LeverSpec(
        name="service_level",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=1.0,
        introduced_phase="1.1",
        enabled=True,
    ),
    # DISCRETE: target automation tier (monotone-irreversible, cash-gated, masked).
    LeverSpec(
        name="automation",
        kind=LeverKind.DISCRETE,
        n=N_AUTOMATION_TIERS,
        default=0.0,
        introduced_phase="1.2",
        enabled=True,
    ),
    LeverSpec(
        name="loyalty_spend",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=0.0,
        introduced_phase="1.3",
        enabled=True,
    ),
    # wage_spend: per-tick fraction that adds to the paid wage.
    # warehouse_invest: per-tick fraction that adds warehouse capacity
    # (non-depreciating stock: spend now, capacity from next tick).
    LeverSpec(
        name="wage_spend",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=0.0,
        introduced_phase="5.0",
        enabled=True,
    ),
    LeverSpec(
        name="warehouse_invest",
        kind=LeverKind.CONTINUOUS,
        low=0.0,
        high=1.0,
        default=0.0,
        introduced_phase="5.0",
        enabled=True,
    ),
)


def enabled_levers() -> tuple[LeverSpec, ...]:
    """Levers active in the current phase, in canonical flat-layout order."""
    return tuple(lever for lever in LEVER_REGISTRY if lever.enabled)


def validate_expansion_capacity(n_regions: int) -> None:
    """Raise ValueError if n_regions exceeds the expansion lever's capacity."""
    expansion = next(lever for lever in LEVER_REGISTRY if lever.name == "expansion")
    assert expansion.n is not None  # discrete levers always define n (LeverSpec.width)
    if not (1 <= n_regions <= expansion.n):
        raise ValueError(
            f"n_regions={n_regions} must be in [1, {expansion.n}] (the expansion "
            f"lever's fixed capacity, N_REGIONS_MAX={expansion.n})"
        )


def validate_automation_width(n_tiers: int) -> None:
    """Raise ValueError if the automation lever's width does not equal n_tiers."""
    automation = next(lever for lever in LEVER_REGISTRY if lever.name == "automation")
    if automation.n != n_tiers:
        raise ValueError(
            f"automation lever n={automation.n} must equal n_tiers={n_tiers} "
            "(one discrete choice per tier: tier_0 + ... + tier_n-1)"
        )


class ObservationGroup(Enum):
    """Which conceptual block an observation field belongs to."""

    SELF = "self"
    MARKET = "market"
    COMPETITOR = "competitor"


@dataclass(frozen=True)
class ObservationField:
    """One named scalar dimension of the flat observation vector."""

    name: str
    group: ObservationGroup
    description: str


# Ordered observation layout; encoding fills these in this exact order.
# Fields are appended at the tail — earlier offsets never move.
OBSERVATION_SCHEMA: tuple[ObservationField, ...] = (
    # --- Self block ---
    ObservationField("cash", ObservationGroup.SELF, "agent cash on hand"),
    ObservationField("last_profit", ObservationGroup.SELF, "profit from previous tick"),
    ObservationField("last_revenue", ObservationGroup.SELF, "revenue from previous tick"),
    ObservationField("market_share", ObservationGroup.SELF, "agent regional market share"),
    ObservationField("price_level", ObservationGroup.SELF, "agent current price_index"),
    ObservationField("loyalty", ObservationGroup.SELF, "agent loyalty stock"),
    ObservationField("stores", ObservationGroup.SELF, "agent store count"),
    # --- Market aggregates ---
    ObservationField(
        "total_regional_demand", ObservationGroup.MARKET, "total units demanded in the region"
    ),
    ObservationField(
        "mean_competitor_price", ObservationGroup.MARKET, "mean price_index across competitors"
    ),
        # --- Competitor block (perceived, noised) ---
    ObservationField(
        "competitor_price",
        ObservationGroup.COMPETITOR,
        "perceived discounter price_index (noised; sharpened by research fidelity)",
    ),
    ObservationField(
        "competitor_market_share",
        ObservationGroup.COMPETITOR,
        "perceived discounter market share (noised; sharpened by research fidelity)",
    ),
    # Seated awareness (read by demand this tick; built from prior-tick spend).
    ObservationField(
        "awareness",
        ObservationGroup.SELF,
        "agent seated awareness stock (read by demand this tick)",
    ),
    ObservationField(
        "competitor_awareness",
        ObservationGroup.COMPETITOR,
        "perceived discounter seated awareness stock (noised; sharpened by research fidelity)",
    ),
    # Assortment acted last tick (reporting-only; not a demand input).
    ObservationField(
        "assortment",
        ObservationGroup.SELF,
        "agent assortment breadth acted last tick (contemporaneous setting)",
    ),
    ObservationField(
        "competitor_assortment",
        ObservationGroup.COMPETITOR,
        "perceived discounter assortment breadth last tick (noised; sharpened by research fidelity)",
    ),
    # Promotion acted last tick (reporting-only). stockpile is the real seated
    # state that suppresses demand.
    # suppressing demand and learn to let it decay before promoting again. ---
    ObservationField(
        "promotion",
        ObservationGroup.SELF,
        "agent promotion intensity acted last tick (contemporaneous lift)",
    ),
    ObservationField(
        "competitor_promotion",
        ObservationGroup.COMPETITOR,
        "perceived discounter promotion intensity last tick (noised; sharpened by research fidelity)",
    ),
    ObservationField(
        "stockpile",
        ObservationGroup.MARKET,
        "seated regional consumer stockpile (suppresses demand this tick)",
    ),
    ObservationField(
        "expansion",
        ObservationGroup.SELF,
        "agent expansion choice acted last tick (0 = no-op, r = opened region r)",
    ),
    # Region-0 loyalty pair (makes region 0's per-region loyalty explicit).
    ObservationField(
        "loyalty_region_0",
        ObservationGroup.SELF,
        "agent seated loyalty stock in region 0 (per-region moat, read by demand)",
    ),
    ObservationField(
        "competitor_loyalty_region_0",
        ObservationGroup.COMPETITOR,
        "perceived discounter seated loyalty stock in region 0 (noised; sharpened by research fidelity)",
    ),
    # (3) Region-1 SELF block: per-region awareness/loyalty/presence/share.
    ObservationField(
        "awareness_region_1",
        ObservationGroup.SELF,
        "agent seated awareness stock in region 1 (read by demand this tick)",
    ),
    ObservationField(
        "loyalty_region_1",
        ObservationGroup.SELF,
        "agent seated loyalty stock in region 1 (per-region moat)",
    ),
    ObservationField(
        "stores_region_1",
        ObservationGroup.SELF,
        "agent presence in region 1 (0 = absent, 1 = present)",
    ),
    ObservationField(
        "market_share_region_1",
        ObservationGroup.SELF,
        "agent realized market share in region 1 last tick",
    ),
    # (4) Region-1 MARKET block: per-region realized demand + stockpile.
    ObservationField(
        "total_regional_demand_region_1",
        ObservationGroup.MARKET,
        "total units demanded in region 1 last tick (realized pie)",
    ),
    ObservationField(
        "stockpile_region_1",
        ObservationGroup.MARKET,
        "seated region-1 consumer stockpile (suppresses region-1 demand this tick)",
    ),
    # (5) Region-1 COMPETITOR block: the incumbent's per-region awareness/loyalty/share.
    ObservationField(
        "competitor_awareness_region_1",
        ObservationGroup.COMPETITOR,
        "perceived discounter seated awareness stock in region 1 (noised; sharpened by research fidelity)",
    ),
    ObservationField(
        "competitor_loyalty_region_1",
        ObservationGroup.COMPETITOR,
        "perceived discounter seated loyalty stock in region 1 (noised; sharpened by research fidelity)",
    ),
    ObservationField(
        "competitor_market_share_region_1",
        ObservationGroup.COMPETITOR,
        "perceived discounter realized market share in region 1 last tick (noised; sharpened by research fidelity)",
    ),
    ObservationField(
        "research_fidelity",
        ObservationGroup.SELF,
        "agent current perception fidelity in [0, 1) (how much to trust the perceived competitor block)",
    ),
    ObservationField(
        "competitor_mean_price_region_0",
        ObservationGroup.COMPETITOR,
        "perceived (noised) per-region mean price of ALL other present retailers "
        "= the demand reference (sharpened by research)",
    ),
    ObservationField(
        "competitor_mean_price_region_1",
        ObservationGroup.COMPETITOR,
        "perceived (noised) per-region mean price of ALL other present retailers "
        "= the demand reference (sharpened by research)",
    ),
    ObservationField(
        "service_score_region_0",
        ObservationGroup.SELF,
        "agent's seated per-region service_score in [0,1] (per-region liability — "
        "decays back to 1.0; lower after stockouts)",
    ),
    ObservationField(
        "service_score_region_1",
        ObservationGroup.SELF,
        "agent's seated per-region service_score in [0,1] (per-region liability — "
        "decays back to 1.0; lower after stockouts)",
    ),
    # Seated automation tier (monotone non-decreasing; drives the COGS-savings
    # multiplier, lagged).
    ObservationField(
        "automation_tier",
        ObservationGroup.SELF,
        "agent's seated monotone non-decreasing automation tier (integer in "
        "[0, n_tiers-1] as float) — drives the COGS-savings multiplier (lagged)",
    ),
    ObservationField(
        "employee_happiness",
        ObservationGroup.SELF,
        "agent's seated employee happiness in [0,1] (write-off risk gauge; EMA "
        "toward the wage-vs-required target)",
    ),
    ObservationField(
        "wage_level",
        ObservationGroup.SELF,
        "agent's wage paid last tick (wage_base, or wage_base + wage_spend · "
        "wage_spend_scale once the seam consumes the lever)",
    ),
    ObservationField(
        "warehouse_utilization",
        ObservationGroup.SELF,
        "agent's warehouse utilization last tick in [0,1] (served / capacity; "
        "0.0 when capacity is unmetered/inf)",
    ),
    ObservationField(
        "last_stockout_rate",
        ObservationGroup.SELF,
        "agent's demand-weighted lost-sales fraction last tick in [0,1] "
        "(supplier + warehouse losses)",
    ),
    ObservationField(
        "others_present_region_0",
        ObservationGroup.MARKET,
        "count of OTHER present retailers in region 0 (exact; a pure read, no RNG)",
    ),
    ObservationField(
        "others_present_region_1",
        ObservationGroup.MARKET,
        "count of OTHER present retailers in region 1 (exact; a pure read, no RNG)",
    ),
)


def _region_observation_block(r: int) -> tuple[ObservationField, ...]:
    """Per-region observation fields for region r >= 2 (self, market, competitor blocks)."""
    return (
        # --- SELF (5) ---
        ObservationField(
            f"awareness_region_{r}",
            ObservationGroup.SELF,
            f"agent seated awareness stock in region {r} (read by demand this tick)",
        ),
        ObservationField(
            f"loyalty_region_{r}",
            ObservationGroup.SELF,
            f"agent seated loyalty stock in region {r} (per-region moat)",
        ),
        ObservationField(
            f"stores_region_{r}",
            ObservationGroup.SELF,
            f"agent presence in region {r} (0 = absent, 1 = present)",
        ),
        ObservationField(
            f"market_share_region_{r}",
            ObservationGroup.SELF,
            f"agent realized market share in region {r} last tick",
        ),
        ObservationField(
            f"service_score_region_{r}",
            ObservationGroup.SELF,
            f"agent's seated per-region service_score in [0,1] in region {r} "
            "(per-region liability — decays back to 1.0; lower after stockouts)",
        ),
        # --- MARKET (3) ---
        ObservationField(
            f"total_regional_demand_region_{r}",
            ObservationGroup.MARKET,
            f"total units demanded in region {r} last tick (realized pie)",
        ),
        ObservationField(
            f"stockpile_region_{r}",
            ObservationGroup.MARKET,
            f"seated region-{r} consumer stockpile (suppresses region-{r} demand this tick)",
        ),
        ObservationField(
            f"others_present_region_{r}",
            ObservationGroup.MARKET,
            f"count of OTHER present retailers in region {r} (exact; a pure read, no RNG)",
        ),
        # --- COMPETITOR (4, noised) ---
        ObservationField(
            f"competitor_awareness_region_{r}",
            ObservationGroup.COMPETITOR,
            f"perceived discounter seated awareness stock in region {r} "
            "(noised; sharpened by research fidelity)",
        ),
        ObservationField(
            f"competitor_loyalty_region_{r}",
            ObservationGroup.COMPETITOR,
            f"perceived discounter seated loyalty stock in region {r} "
            "(noised; sharpened by research fidelity)",
        ),
        ObservationField(
            f"competitor_market_share_region_{r}",
            ObservationGroup.COMPETITOR,
            f"perceived discounter realized market share in region {r} last tick "
            "(noised; sharpened by research fidelity)",
        ),
        ObservationField(
            f"competitor_mean_price_region_{r}",
            ObservationGroup.COMPETITOR,
            f"perceived (noised) per-region mean price of ALL other present retailers "
            f"in region {r} = the demand reference (sharpened by research)",
        ),
    )


@cache
def observation_schema(n_regions: int) -> tuple[ObservationField, ...]:
    """The observation layout for n_regions (base layout + one block per extra region)."""
    if n_regions <= 2:
        return OBSERVATION_SCHEMA
    extra = tuple(field for r in range(2, n_regions) for field in _region_observation_block(r))
    return OBSERVATION_SCHEMA + extra


@dataclass(frozen=True)
class ActionSchema:
    """Immutable snapshot of the action contract (version + enabled levers)."""

    version: int = ACTION_SCHEMA_VERSION
    levers: tuple[LeverSpec, ...] = field(default_factory=enabled_levers)

    @property
    def total_dim(self) -> int:
        """Total flat action dimensions across all enabled levers."""
        return sum(lever.width for lever in self.levers)


@lru_cache(maxsize=1)
def default_action_schema() -> ActionSchema:
    """The action schema for the current phase (cached singleton — immutable)."""
    return ActionSchema()
