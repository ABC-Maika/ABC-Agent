"""World state data structures — plain data, no behavior.

``(WorldState, rng.bit_generator.state)`` fully serializes the sim for replay
and checkpointing. Only ``world.py`` advances a ``WorldState``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported for typing only (string annotation; never evaluated at import).
    from retail_simulator.core.reward import RewardNormalizer

PRICE_SENSITIVE = "price_sensitive"
QUALITY_SEEKING = "quality_seeking"
CONVENIENCE = "convenience"
BRAND_LOYAL = "brand_loyal"
PREMIUM_HUNTER = "premium_hunter"
BARGAIN_SEEKER = "bargain_seeker"


SEGMENT_KEYS: frozenset[str] = frozenset(
    {
        PRICE_SENSITIVE,
        QUALITY_SEEKING,
        CONVENIENCE,
        BRAND_LOYAL,
        PREMIUM_HUNTER,
        BARGAIN_SEEKER,
    }
)


@dataclass(frozen=True)
class SegmentParams:
    """Attractiveness sensitivity weights for one customer segment."""

    beta_price: float = 2.0
    beta_assortment: float = 0.0
    beta_marketing: float = 0.0
    # Promo-lift weight: how strongly a segment is drawn by this tick's promotion.
    beta_promotion: float = 0.0
    beta_loyalty: float = 0.0
    beta_reference: float = 0.0
    beta_under_reference_aversion: float = 0.0
    beta_service: float = 0.0
    beta_program: float = 0.0


def default_segments() -> dict[str, SegmentParams]:
    """The four default segments."""
    return {
        PRICE_SENSITIVE: SegmentParams(
            beta_price=2.0,
            beta_assortment=0.4,
            beta_marketing=0.0,
            beta_promotion=1.5,
            beta_loyalty=0.0,
            beta_program=0.0,
            beta_under_reference_aversion=0.0,
        ),
        QUALITY_SEEKING: SegmentParams(
            beta_price=0.5,
            beta_assortment=2.0,
            beta_marketing=1.0,
            beta_promotion=0.3,
            beta_loyalty=0.0,
            beta_program=0.0,
            beta_under_reference_aversion=0.0,
        ),
        CONVENIENCE: SegmentParams(
            beta_price=1.0,
            beta_assortment=1.0,
            beta_marketing=0.5,
            beta_promotion=0.8,
            beta_loyalty=0.0,
            beta_program=0.0,
            beta_under_reference_aversion=0.0,
        ),
        BRAND_LOYAL: SegmentParams(
            beta_price=0.6,
            beta_assortment=0.5,
            beta_marketing=0.6,
            beta_promotion=0.2,
            # The moat: the only segment with a loyalty weight, paired with the
            # lowest beta_price (sticks with its retailer even under an undercut).
            beta_loyalty=2.5,
            beta_program=0.0,
            beta_under_reference_aversion=0.0,
        ),
    }


@dataclass(frozen=True)
class RegionState:
    """One region's demand size, segment mix, realized demand, and stockpile."""

    population: float = 1000.0
    base_regional_demand: float = 1000.0
    segment_mix: dict[str, float] = field(default_factory=lambda: {PRICE_SENSITIVE: 1.0})
    realized_regional_demand: float = 0.0
    # Seated stockpile: built from prior-tick promos, read by demand to suppress
    # the pie this tick, re-seated each tick. Part of the serialized state.
    stockpile: float = 0.0


@dataclass
class RetailerState:
    """One retailer's cash, levers, and per-region stocks."""

    name: str
    is_npc: bool
    cash: float
    price_index: float = 1.0
    # Per-region arrays, indexed by region (length n_regions).
    stores_per_region: tuple[int, ...] = (1,)
    store_age_per_region: tuple[int, ...] = (0,)
    # Seated awareness stock per region: built from prior-tick spend, read by
    # demand at tick t, re-seated by world.step for t+1.
    awareness_per_region: tuple[float, ...] = (0.0,)
    # Seated loyalty stock per region: EMA of per-region share; read by demand,
    # re-seated each tick.
    loyalty_per_region: tuple[float, ...] = (0.0,)
    last_profit: float = 0.0
    last_revenue: float = 0.0
    # Aggregate (region-0) realized share last tick, kept for the legacy obs slot.
    last_market_share: float = 0.0
    # Per-region realized share last tick (reporting-only).
    last_market_share_per_region: tuple[float, ...] = (0.0,)
    # Assortment breadth acted last tick (reporting-only; never read by demand).
    last_assortment: float = 0.0
    # Promotion intensity acted last tick (reporting-only).
    last_promotion: float = 0.0
    # Expansion choice decoded last tick: 0 = no-op, r = opened region r.
    last_expansion: float = 0.0
    service_score_per_region: tuple[float, ...] = (1.0,)
    automation_tier: int = 0
    # Target automation tier decoded last tick (reporting-only; the seated tier
    # is ``automation_tier``).
    last_automation_tier_action: int = 0
    # Paid wage (seated to wage_base at reset); read by the wage→happiness coupling
    # and as a reference wage by rivals.
    wage_level: float = 0.0
    # Employee happiness in [0, 1]: EMA driven down when paid wage < required
    # wage; starts at 1.0 (the fixed point at default config).
    employee_happiness: float = 1.0
    # Growth-overhead cost charged last tick (reporting-only).
    accumulated_overhead_basis: float = 0.0
    # Prior-tick revenue; the write-off (a % of turnover) reads this (lagged).
    last_turnover: float = 0.0
    warehouse_capacity: float = float("inf")
    # Served-demand / seated-capacity ratio from the last resolved tick
    # (0.0 when capacity is unmetered/inf or at reset).
    last_warehouse_utilization: float = 0.0
    # Demand-weighted lost-sales fraction from the last resolved tick.
    last_stockout_rate: float = 0.0
    # Consecutive ticks cash has stayed below the restructure debt line; 0 once
    # cash recovers (a restructuring also resets it).
    insolvent_ticks: int = 0
    # Count of restructurings this episode (reporting-only).
    # template — never read by any economic array).
    restructurings: int = 0
    # NPC seat's pending flat action: decided last tick, consumed this tick
    # (the same one-tick lag as price; no is_npc branch downstream). None for
    # learning seats and for NPCs that have not acted yet (their levers fall
    # back to the registry defaults). Stored as plain floats so it serializes.
    pending_action: tuple[float, ...] | None = None

    @property
    def stores(self) -> int:
        """Total open-store count across regions (legacy accessor; per-region SoT is stores_per_region)."""
        return sum(self.stores_per_region)

    @property
    def awareness(self) -> float:
        """Legacy accessor: region-0 awareness (the unchanged legacy obs slot)."""
        return self.awareness_per_region[0]

    @property
    def loyalty_stock(self) -> float:
        """Legacy accessor: region-0 loyalty (the unchanged legacy obs slot)."""
        return self.loyalty_per_region[0]

    @property
    def service_score(self) -> float:
        """Region-0 service_score (legacy accessor; per-region SoT is service_score_per_region)."""
        return self.service_score_per_region[0]


@dataclass(frozen=True)
class PerceivedCompetitor:
    """One observer's perceived (possibly noised) view of a competitor."""

    competitor_price: float = 0.0
    competitor_market_share: float = 0.0
    competitor_awareness: float = 0.0
    competitor_assortment: float = 0.0
    competitor_promotion: float = 0.0
    competitor_loyalty_region_0: float = 0.0
    competitor_awareness_region_1: float = 0.0
    competitor_loyalty_region_1: float = 0.0
    competitor_market_share_region_1: float = 0.0
    competitor_mean_price_region_0: float = 0.0
    competitor_mean_price_region_1: float = 0.0
    extra_regions: tuple[tuple[float, float, float, float], ...] = ()


@dataclass
class WorldState:
    """The complete, serializable world (sans RNG, threaded separately).

    ``retailers`` is ordered: index 0 is the learning agent, subsequent indices
    are NPCs (the order NPC actions are drawn in).
    """

    tick: int
    regions: list[RegionState]
    retailers: list[RetailerState]
    segments: dict[str, SegmentParams] = field(
        default_factory=lambda: {PRICE_SENSITIVE: SegmentParams()}
    )
    # One RewardNormalizer per learning seat (keyed by seat index; NPCs get none).
    # Advanced once per tick after accounting; empty for mode="profit".
    reward_stats: dict[int, RewardNormalizer] = field(default_factory=dict)
    perceived_competitors: dict[int, PerceivedCompetitor] = field(default_factory=dict)
    research_fidelity: dict[int, float] = field(default_factory=dict)
