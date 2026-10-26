"""Resolved domain parameters for the sim core (pure dataclasses, no behavior).

RL-wrapper concerns (gamma, episode length) live outside core; envs/ reads the
MAX_EPISODE_STEPS_DEFAULT hint constant exposed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from retail_simulator.core.state import (
    BRAND_LOYAL,
    CONVENIENCE,
    PRICE_SENSITIVE,
    QUALITY_SEEKING,
    SEGMENT_KEYS,
    SegmentParams,
)

# Training horizon hint (~2 simulated years); an RL-wrapper concern, not core.
MAX_EPISODE_STEPS_DEFAULT: int = 104


def _validate_segment_mix(segment_mix: dict[str, float]) -> None:
    """Validate a segment_mix dict: known keys, non-negative shares, sums to 1.0."""
    if not segment_mix:
        raise ValueError("segment_mix must not be empty")
    unknown = set(segment_mix) - SEGMENT_KEYS
    if unknown:
        raise ValueError(
            f"segment_mix contains unknown segment keys: {sorted(unknown)}; "
            f"legal keys are {sorted(SEGMENT_KEYS)}"
        )
    if any(share < 0.0 for share in segment_mix.values()):
        raise ValueError(f"segment_mix shares must be non-negative: {segment_mix}")
    total = sum(segment_mix.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"segment_mix must sum to 1.0, got {total} for {segment_mix}")


@dataclass(frozen=True)
class RegionConfig:
    """Per-region demand size and segment population mix."""

    base_regional_demand: float = 1000.0
    segment_mix: dict[str, float] = field(
        default_factory=lambda: {
            PRICE_SENSITIVE: 0.4,
            QUALITY_SEEKING: 0.25,
            CONVENIENCE: 0.2,
            BRAND_LOYAL: 0.15,
        }
    )

    def __post_init__(self) -> None:
        _validate_segment_mix(self.segment_mix)


@dataclass(frozen=True)
class DemandConfig:
    """Demand size, elasticity, noise, and the list of regions."""

    base_regional_demand: float = 1000.0
    # Price elasticity of total regional demand (negative: higher price shrinks the pie).
    price_elasticity: float = -1.5
    noise_sigma: float = 0.05
    # Segment population shares (must sum to 1.0); region-0 default / single-region fallback.
    segment_mix: dict[str, float] = field(
        default_factory=lambda: {
            PRICE_SENSITIVE: 0.4,
            QUALITY_SEEKING: 0.25,
            CONVENIENCE: 0.2,
            BRAND_LOYAL: 0.15,
        }
    )
    # Per-region overrides; the default is two regions, region 1 smaller and brand-loyal-heavy.
    regions: tuple[RegionConfig, ...] = field(
        default_factory=lambda: (
            RegionConfig(
                base_regional_demand=1000.0,
                segment_mix={
                    PRICE_SENSITIVE: 0.4,
                    QUALITY_SEEKING: 0.25,
                    CONVENIENCE: 0.2,
                    BRAND_LOYAL: 0.15,
                },
            ),
            RegionConfig(
                base_regional_demand=600.0,
                segment_mix={
                    PRICE_SENSITIVE: 0.25,
                    QUALITY_SEEKING: 0.15,
                    CONVENIENCE: 0.15,
                    BRAND_LOYAL: 0.45,
                },
            ),
        )
    )
    # Max per-region share one retailer may hold; 1.0 = off.
    max_share_per_region: float = 1.0

    def __post_init__(self) -> None:
        _validate_segment_mix(self.segment_mix)
        for region in self.regions:
            _validate_segment_mix(region.segment_mix)
        if not (0.0 < self.max_share_per_region <= 1.0):
            raise ValueError(
                f"max_share_per_region must be in (0.0, 1.0], got {self.max_share_per_region}"
            )


@dataclass(frozen=True)
class MarketingConfig:
    """Marketing (spend_fraction lever) cost parameters."""

    cost_per_unit_spend: float = 200.0
    decay: float = 0.3
    # Convexity of marketing cost in spend_fraction; 1.0 = linear (off).
    marketing_cost_exponent: float = 1.0

    def __post_init__(self) -> None:
        if self.marketing_cost_exponent < 1.0:
            raise ValueError(
                f"marketing_cost_exponent must be >= 1, got {self.marketing_cost_exponent}"
            )


@dataclass(frozen=True)
class AssortmentConfig:
    """Assortment lever: a per-tick setting like price (no seated state), with its cost."""

    assortment_cost_per_unit: float = 120.0


@dataclass(frozen=True)
class PromotionConfig:
    """Promotion lever: per-unit margin compression and the consumer stockpile."""

    # Promo is a variable cost of sales: cost per unit sold per unit of promotion.
    # Tuned so sustained promotion has an interior optimum (not free, not inert).
    promo_cost_per_unit: float = 0.009
    stockpile_build: float = 0.17
    stockpile_decay: float = 0.7
    # Weight netting promotion against present rivals' mean (per region); 0.0 = absolute (off).
    promotion_contest_weight: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.promotion_contest_weight <= 1.0):
            raise ValueError(
                "promotion_contest_weight must be in [0.0, 1.0], got "
                f"{self.promotion_contest_weight}"
            )


@dataclass(frozen=True)
class PricingConfig:
    """Bounds and baseline for the pricing lever (mirrors schema price_index)."""

    min_price_index: float = 0.5
    max_price_index: float = 2.0
    baseline_price_index: float = 1.0


@dataclass(frozen=True)
class CostConfig:
    """Cost structure used by the accounting engine."""

    # Cost of goods as a fraction of the base price (price_index == 1.0).
    cogs_fraction: float = 0.4
    # Fixed operating expense per store per tick.
    fixed_opex_per_store: float = 50.0


@dataclass(frozen=True)
class LoyaltyConfig:
    """Loyalty stock dynamics: L = alpha * L + (1 - alpha) * market_share, accrued per region every tick."""

    alpha: float = 0.85


@dataclass(frozen=True)
class LoyaltyProgramConfig:
    """Loyalty-program spend lever cost."""

    cost_per_unit_spend: float = 0.0

    def __post_init__(self) -> None:
        if self.cost_per_unit_spend < 0.0:
            raise ValueError(f"cost_per_unit_spend must be >= 0, got {self.cost_per_unit_spend}")


@dataclass(frozen=True)
class ExpansionConfig:
    """Expansion lever: opening a store in a new region costs a one-time

    ``expansion_capex`` plus an ongoing ``fixed_opex_per_store`` per tick.
    Open-only and cash-gated; an unaffordable open is a no-op.
    """

    expansion_capex: float = 5_000.0


@dataclass(frozen=True)
class ResearchConfig:
    """Research lever: cost and the saturating fidelity mapping."""

    cost_per_fidelity: float = 15.0
    noise_sigma_base: float = 0.3
    # Denominator of the saturating fidelity mapping; keeps fidelity < 1.
    fidelity_k: float = 0.3

    def __post_init__(self) -> None:
        if self.cost_per_fidelity < 0.0:
            raise ValueError(f"cost_per_fidelity must be >= 0, got {self.cost_per_fidelity}")
        if self.noise_sigma_base < 0.0:
            raise ValueError(f"noise_sigma_base must be >= 0, got {self.noise_sigma_base}")
        if self.fidelity_k <= 0.0:
            raise ValueError(f"fidelity_k must be > 0, got {self.fidelity_k}")

    def fidelity(self, research_spend: float) -> float:
        """Saturating fidelity for a given research spend: spend / (spend + k)."""
        return research_spend / (research_spend + self.fidelity_k)


@dataclass(frozen=True)
class SCMConfig:
    """Supply-chain service: fill rate from service_level and its cost penalty."""

    cogs_premium: float = 0.0
    # Minimum fill rate at service_level = 0; tuned so the cheap end is genuinely punishing.
    fill_rate_floor: float = 0.5
    fill_rate_curvature: float = 1.0
    lost_sales_share_penalty: float = 0.0
    service_score_decay: float = 0.1

    def __post_init__(self) -> None:
        if self.cogs_premium < 0.0:
            raise ValueError(f"cogs_premium must be >= 0, got {self.cogs_premium}")
        if not (0.0 <= self.fill_rate_floor <= 1.0):
            raise ValueError(f"fill_rate_floor must be in [0, 1], got {self.fill_rate_floor}")
        if self.lost_sales_share_penalty < 0.0:
            raise ValueError(
                f"lost_sales_share_penalty must be >= 0, got {self.lost_sales_share_penalty}"
            )
        if not (0.0 < self.service_score_decay <= 1.0):
            raise ValueError(
                f"service_score_decay must be in (0, 1], got {self.service_score_decay}"
            )

    def fill_rate(self, service_level: float) -> float:
        """Fill rate for a given service_level (floor plus linear rise)."""
        return self.fill_rate_floor + (1.0 - self.fill_rate_floor) * service_level


@dataclass(frozen=True)
class AutomationConfig:
    """Automation tiers: one-time capex and per-tick savings per tier."""

    capex_per_tier: tuple[float, float, float] = (0.0, 0.0, 0.0)
    savings_per_tier: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        from retail_simulator.core.schema import N_AUTOMATION_TIERS

        if len(self.capex_per_tier) != N_AUTOMATION_TIERS:
            raise ValueError(
                f"capex_per_tier must have N_AUTOMATION_TIERS={N_AUTOMATION_TIERS} "
                f"entries, got {len(self.capex_per_tier)}"
            )
        if len(self.savings_per_tier) != N_AUTOMATION_TIERS:
            raise ValueError(
                f"savings_per_tier must have N_AUTOMATION_TIERS="
                f"{N_AUTOMATION_TIERS} entries, got {len(self.savings_per_tier)}"
            )
        if any(c < 0.0 for c in self.capex_per_tier):
            raise ValueError(f"capex_per_tier must be non-negative, got {self.capex_per_tier}")
        if self.capex_per_tier[0] != 0.0:
            raise ValueError(
                f"capex_per_tier[0] must be 0.0 (the no-automation anchor), "
                f"got {self.capex_per_tier[0]}"
            )
        for i in range(len(self.capex_per_tier) - 1):
            if self.capex_per_tier[i + 1] < self.capex_per_tier[i]:
                raise ValueError(
                    f"capex_per_tier must be monotone non-decreasing, got {self.capex_per_tier}"
                )
        if any(not (0.0 <= t < 1.0) for t in self.savings_per_tier):
            raise ValueError(f"savings_per_tier must be in [0, 1), got {self.savings_per_tier}")
        if self.savings_per_tier[0] != 0.0:
            raise ValueError(
                f"savings_per_tier[0] must be 0.0 (the no-automation anchor; "
                f"a tier-0 retailer earns zero savings), got "
                f"{self.savings_per_tier[0]}"
            )
        for i in range(len(self.savings_per_tier) - 1):
            if self.savings_per_tier[i + 1] < self.savings_per_tier[i]:
                raise ValueError(
                    f"savings_per_tier must be monotone non-decreasing, got {self.savings_per_tier}"
                )


@dataclass(frozen=True)
class WageConfig:
    """Wage and recovery mechanics: affordability gate, credit line, restructuring, write-offs, overhead."""

    # Affordability-clamp gate; False = no clamp (byte-identical).
    cash_budget_enabled: bool = False
    overdraft_rate: float = 0.0
    # Working-capital credit line: widens the spend clamp's basis by this much.
    # Funds operations only, never expansion/automation capex. 0.0 = off.
    credit_limit: float = 0.0
    restructure_after_ticks: int = 0
    # How many of the best closeable regions survive a restructuring; region 0 is always exempt.
    restructure_keep_stores: int = 1
    # Widens the restructuring trigger's debt line beyond -credit_limit; 0.0 = off.
    restructure_debt_threshold: float = 0.0
    # Rival-coupled reference-wage weight; 0.0 = the write-off gate's rival leg is off.
    wage_coupling_coeff: float = 0.0
    # Own-share term weight; 0.0 = off.
    wage_own_share_coeff: float = 0.0
    # The required-wage intercept. Also the reset wage_level anchor.
    wage_base: float = 0.0
    # Write-off severity scale; 0.0 (with wage_coupling_coeff 0.0) = gate off, no draw.
    writeoff_severity_scale: float = 0.0
    # Bounded write-off draw range (fraction of turnover).
    writeoff_frac_min: float = 0.0
    writeoff_frac_max: float = 0.0
    # Leader-weighting exponent on retailer size; 1.0 = plain % of turnover.
    leader_weight_exponent: float = 1.0
    # Employee-happiness EMA rate; inert at default config (required == paid == 0).
    happiness_decay: float = 0.1
    # The store-count threshold above which super-linear overhead bites.
    overhead_threshold_stores: float = 0.0
    # Overhead coefficient; 0.0 = off (byte-identical).
    overhead_superlinear_coeff: float = 0.0
    # The overhead exponent (super-linear when > 1). Default 1.0.
    overhead_exponent: float = 1.0
    overhead_reference_stores: float = 1.0
    # Money scale of the wage_spend lever; 0.0 = inert (paid_wage stays wage_base).
    wage_spend_scale: float = 0.0
    # Divides the paid-vs-required wage gap before it drives the happiness target; > 0.
    wage_shortfall_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.overdraft_rate < 0.0:
            raise ValueError(f"overdraft_rate must be >= 0, got {self.overdraft_rate}")
        if self.credit_limit < 0.0:
            raise ValueError(f"credit_limit must be >= 0, got {self.credit_limit}")
        if self.overdraft_rate != 0.0 and self.credit_limit == 0.0:
            raise ValueError(
                f"""overdraft_rate ({self.overdraft_rate}) requires a positive credit_limit (got 0.0) - with no credit line to bound the charge, overdraft_rate would price the whole deficit every tick and, with bankruptcy deferred, compound geometrically forever"""
            )
        if self.restructure_after_ticks < 0:
            raise ValueError(
                f"restructure_after_ticks must be >= 0, got {self.restructure_after_ticks}"
            )
        if self.restructure_keep_stores < 1:
            raise ValueError(
                f"restructure_keep_stores must be >= 1, got {self.restructure_keep_stores}"
            )
        if self.restructure_debt_threshold < 0.0:
            raise ValueError(
                f"restructure_debt_threshold must be >= 0, got {self.restructure_debt_threshold}"
            )
        if self.wage_coupling_coeff < 0.0:
            raise ValueError(f"wage_coupling_coeff must be >= 0, got {self.wage_coupling_coeff}")
        if self.wage_own_share_coeff < 0.0:
            raise ValueError(f"wage_own_share_coeff must be >= 0, got {self.wage_own_share_coeff}")
        if self.wage_base < 0.0:
            raise ValueError(f"wage_base must be >= 0, got {self.wage_base}")
        if self.writeoff_severity_scale < 0.0:
            raise ValueError(
                f"writeoff_severity_scale must be >= 0, got {self.writeoff_severity_scale}"
            )
        if self.writeoff_frac_min < 0.0:
            raise ValueError(f"writeoff_frac_min must be >= 0, got {self.writeoff_frac_min}")
        if self.writeoff_frac_max < self.writeoff_frac_min:
            raise ValueError(
                f"writeoff_frac_max ({self.writeoff_frac_max}) must be >= "
                f"writeoff_frac_min ({self.writeoff_frac_min})"
            )
        if not (0.0 < self.happiness_decay <= 1.0):
            raise ValueError(f"happiness_decay must be in (0, 1], got {self.happiness_decay}")
        if self.wage_spend_scale < 0.0:
            raise ValueError(f"wage_spend_scale must be >= 0, got {self.wage_spend_scale}")
        if self.wage_shortfall_scale <= 0.0:
            raise ValueError(f"wage_shortfall_scale must be > 0, got {self.wage_shortfall_scale}")
        if self.overhead_threshold_stores < 0.0:
            raise ValueError(
                f"overhead_threshold_stores must be >= 0, got {self.overhead_threshold_stores}"
            )
        if self.overhead_superlinear_coeff < 0.0:
            raise ValueError(
                f"overhead_superlinear_coeff must be >= 0, got {self.overhead_superlinear_coeff}"
            )
        if self.overhead_exponent < 1.0:
            raise ValueError(f"overhead_exponent must be >= 1, got {self.overhead_exponent}")
        if self.overhead_reference_stores <= 0.0:
            raise ValueError(
                f"overhead_reference_stores must be > 0, got {self.overhead_reference_stores}"
            )

    @property
    def writeoff_gate_active(self) -> bool:
        """True when the wage write-off gate is armed."""
        return self.wage_coupling_coeff != 0.0 or self.writeoff_severity_scale != 0.0

    @property
    def restructuring_gate_active(self) -> bool:
        """True when the restructuring gate is armed."""
        return self.restructure_after_ticks > 0


@dataclass(frozen=True)
class SupplierConfig:
    """Supplier contention: per-region capacity split among present retailers by service."""

    # Master gate; False = no contention (byte-identical).
    supplier_contention_enabled: bool = False
    # Per-region supplier capacity; inf = contention never binds.
    supplier_capacity: float = float("inf")
    # Priority exponent on service_level; 1.0 = linear, 0.0 = equal shares.
    preference_sharpness: float = 1.0

    def __post_init__(self) -> None:
        if not (self.supplier_capacity > 0.0):
            raise ValueError(f"supplier_capacity must be > 0, got {self.supplier_capacity}")
        if self.preference_sharpness < 0.0:
            raise ValueError(f"preference_sharpness must be >= 0, got {self.preference_sharpness}")


@dataclass(frozen=True)
class WarehouseConfig:
    """Warehouse throughput cap, shortfall penalty, and the warehouse_invest lever."""

    # Master gate for the throughput cap and spill; False = units flow untouched (byte-identical).
    throughput_enabled: bool = False
    # Seated warehouse capacity at reset; inf = the cap never binds.
    default_warehouse_capacity: float = float("inf")
    # Footprint-scaled top-up: effective capacity += this * stores_total; 0.0 = off (byte-identical).
    provided_capacity_per_store: float = 0.0
    # Required capacity per store; a shortfall is possible only when this exceeds the provided capacity.
    capacity_per_store: float = 0.0
    # Shortfall penalty severity scale; 0.0 = off (byte-identical).
    penalty_severity_scale: float = 0.0
    # Bounded penalty draw range (fraction of prior turnover).
    penalty_frac_min: float = 0.0
    penalty_frac_max: float = 0.0
    # Half-saturation constant of the penalty severity; > 0.
    penalty_halfsat: float = 1.0
    # Capacity yield per unit of warehouse_invest (a non-depreciating stock); 0.0 = inert.
    capacity_per_unit_invest: float = 0.0
    # Opex cost per unit of warehouse_invest; 0.0 = no cost.
    cost_per_unit_invest: float = 0.0
    ramp_ticks: int = 0

    def __post_init__(self) -> None:
        if self.provided_capacity_per_store < 0.0:
            raise ValueError(
                f"provided_capacity_per_store must be >= 0, got "
                f"{self.provided_capacity_per_store}"
            )
        if self.capacity_per_store < 0.0:
            raise ValueError(f"capacity_per_store must be >= 0, got {self.capacity_per_store}")
        if self.penalty_severity_scale < 0.0:
            raise ValueError(
                f"penalty_severity_scale must be >= 0, got {self.penalty_severity_scale}"
            )
        if self.penalty_frac_min < 0.0:
            raise ValueError(f"penalty_frac_min must be >= 0, got {self.penalty_frac_min}")
        if self.penalty_frac_max < self.penalty_frac_min:
            raise ValueError(
                f"penalty_frac_max ({self.penalty_frac_max}) must be >= "
                f"penalty_frac_min ({self.penalty_frac_min})"
            )
        if not (self.penalty_halfsat > 0.0):
            raise ValueError(f"penalty_halfsat must be > 0, got {self.penalty_halfsat}")
        if self.capacity_per_unit_invest < 0.0:
            raise ValueError(
                f"capacity_per_unit_invest must be >= 0, got {self.capacity_per_unit_invest}"
            )
        if self.cost_per_unit_invest < 0.0:
            raise ValueError(f"cost_per_unit_invest must be >= 0, got {self.cost_per_unit_invest}")
        if self.ramp_ticks < 0:
            raise ValueError(f"ramp_ticks must be >= 0, got {self.ramp_ticks}")

    @property
    def penalty_gate_active(self) -> bool:
        """True when the warehouse shortfall penalty is armed."""
        return self.penalty_severity_scale != 0.0


@dataclass(frozen=True)
class RewardConfig:
    """Reward shape: pure profit or a weighted blend of profit, revenue, share, and loyalty."""

    mode: str = "profit"  # "profit" | "weighted"
    weight_profit: float = 0.5
    weight_revenue: float = 0.2
    weight_market_share: float = 0.2
    weight_loyalty: float = 0.1
    normalizer_warmup: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("profit", "weighted"):
            raise ValueError(f"reward mode must be 'profit' or 'weighted', got {self.mode!r}")
        if self.normalizer_warmup < 0:
            raise ValueError(f"normalizer_warmup must be >= 0, got {self.normalizer_warmup}")
        total = (
            self.weight_profit
            + self.weight_revenue
            + self.weight_market_share
            + self.weight_loyalty
        )
        # Weights must sum to 1 only in weighted mode; profit mode ignores them.
        if self.mode == "weighted" and abs(total - 1.0) > 1e-9:
            raise ValueError(f"weighted reward weights must sum to 1.0, got {total}")


NPC_ARCHETYPES: tuple[str, ...] = ("discounter", "premium", "balanced", "wandering")


@dataclass(frozen=True)
class SeatSpec:
    """One arena seat: a learning player, or an NPC of a named archetype."""

    is_npc: bool
    archetype: str | None = None
    presence: tuple[int, ...] | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if self.is_npc:
            if self.archetype is None:
                raise ValueError("an NPC seat must name an archetype")
            if self.archetype not in NPC_ARCHETYPES:
                raise ValueError(
                    f"unknown NPC archetype {self.archetype!r}; known: {NPC_ARCHETYPES}"
                )
        elif self.archetype is not None:
            raise ValueError("a learning seat must not name an archetype")
        if self.presence is not None and any(s < 0 for s in self.presence):
            raise ValueError(f"seat presence store counts must be non-negative: {self.presence}")


@dataclass(frozen=True)
class CoreConfig:
    """Top-level resolved configuration for the pure world model."""

    demand: DemandConfig = field(default_factory=DemandConfig)
    pricing: PricingConfig = field(default_factory=PricingConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    loyalty: LoyaltyConfig = field(default_factory=LoyaltyConfig)
    marketing: MarketingConfig = field(default_factory=MarketingConfig)
    assortment: AssortmentConfig = field(default_factory=AssortmentConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    # Research lever; default 0.0 keeps the economics byte-identical (perception is the last RNG draw).
    research: ResearchConfig = field(default_factory=ResearchConfig)
    scm: SCMConfig = field(default_factory=SCMConfig)
    automation: AutomationConfig = field(default_factory=AutomationConfig)
    loyalty_program: LoyaltyProgramConfig = field(default_factory=LoyaltyProgramConfig)
    wage: WageConfig = field(default_factory=WageConfig)
    supplier: SupplierConfig = field(default_factory=SupplierConfig)
    warehouse: WarehouseConfig = field(default_factory=WarehouseConfig)
    # Reward shape; default mode="profit" is byte-identical.
    reward: RewardConfig = field(default_factory=RewardConfig)
    seats: tuple[SeatSpec, ...] | None = None
    segments: dict[str, SegmentParams] | None = None

    @classmethod
    def default(cls) -> CoreConfig:
        """The default configuration (reward mode='profit', no seat plan)."""
        return cls()
