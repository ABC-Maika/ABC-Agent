"""Build Gymnasium Spaces from the core schema (single source of truth).

The flat layout is owned by ``core``; this module turns it into concrete Box
spaces so the adapter never hardcodes a dimension or offset.
"""

from __future__ import annotations

import re

import numpy as np
from gymnasium import spaces

from retail_simulator.core.config import CoreConfig
from retail_simulator.core.encoding import OBS_DTYPE, flatten_action_space_layout
from retail_simulator.core.schema import (
    N_REGIONS_MAX,
    ActionSchema,
    LeverKind,
    ObservationField,
    ObservationGroup,
    default_action_schema,
    observation_schema,
)

# Generous finite bound for unbounded monetary fields (cash/profit/revenue).
_MONETARY_BOUND: float = 1e9

# Per-field bounds by semantics. Shares/loyalty live in [0, 1]; price fields
# mirror the pricing lever bounds; monetary magnitudes use _MONETARY_BOUND.
_SHARE_BOUNDS: tuple[float, float] = (0.0, 1.0)

_REGION_SUFFIX_RE: re.Pattern[str] = re.compile(r"^(.+)_region_(\d+)$")

# One bound rule per per-region-block role (keyed by the bare role name).
_REGION_ROLE_SHARE: frozenset[str] = frozenset(
    {
        "loyalty",
        "stores",
        "market_share",
        "service_score",
        "competitor_loyalty",
        "competitor_market_share",
    }
)
_REGION_ROLE_AWARENESS: frozenset[str] = frozenset({"awareness", "competitor_awareness"})
_REGION_ROLE_MONETARY: frozenset[str] = frozenset({"total_regional_demand", "others_present"})
_REGION_ROLE_STOCKPILE: str = "stockpile"
_REGION_ROLE_PRICE: str = "competitor_mean_price"


def _region_role_bounds(
    role: str, price_high: float, awareness_cap: float, stockpile_cap: float
) -> tuple[float, float] | None:
    """Bound for a per-region-block field role (name without region digits), or ``None`` if unknown."""
    if role in _REGION_ROLE_SHARE:
        return _SHARE_BOUNDS
    if role in _REGION_ROLE_AWARENESS:
        return (0.0, awareness_cap)
    if role in _REGION_ROLE_MONETARY:
        # A count with no natural tight cap at this layer.
        return (0.0, _MONETARY_BOUND)
    if role == _REGION_ROLE_STOCKPILE:
        return (0.0, stockpile_cap)
    if role == _REGION_ROLE_PRICE:
        # Low is 0.0: a bare perceived view encodes this field as the 0.0 absent sentinel.
        return (0.0, price_high)
    return None


def _observation_field_bounds(
    field: ObservationField, config: CoreConfig, n_regions: int
) -> tuple[float, float]:
    """Resolve (low, high) for one observation field from its semantics.

    Shares/loyalty are bounded to [0, 1]; monetary magnitudes use a generous finite
    bound; profit may be negative, so its low is the negative of that bound.
    """
    price_low = config.pricing.min_price_index
    price_high = config.pricing.max_price_index

    match = _REGION_SUFFIX_RE.match(field.name)
    if match is not None:
        role = match.group(1)
        awareness_cap = 1.0 / config.marketing.decay
        stockpile_cap = 1.0 / config.promotion.stockpile_decay
        region_bounds = _region_role_bounds(role, price_high, awareness_cap, stockpile_cap)
        if region_bounds is not None:
            return region_bounds

    # Fields whose range is a known closed interval, regardless of group.
    if field.name in ("market_share", "loyalty", "competitor_market_share"):
        return _SHARE_BOUNDS
    if field.name == "competitor_price":
        # Absent competitor encodes as the 0.0 sentinel, so low is 0.0, not min_price_index.
        return (0.0, price_high)
    if field.name in ("price_level", "mean_competitor_price"):
        # Both sit within the pricing bounds (no absent-competitor sentinel).
        return (price_low, price_high)
    if field.name in ("awareness", "competitor_awareness"):
        # Awareness is capped at 1/decay by the recurrence clip, so [0, 1/decay] is the tight bound.
        # fallback (env_checker/SB3 prefer tight finite bounds for stability).
        return (0.0, 1.0 / config.marketing.decay)
    if field.name in (
        "assortment",
        "competitor_assortment",
        "promotion",
        "competitor_promotion",
    ):
        # Fractions in [0, 1] (the levers' action bounds).
        # fallback.
        return _SHARE_BOUNDS
    if field.name == "research_fidelity":
        # fidelity = spend / (spend + k) saturates in [0, 1).
        return _SHARE_BOUNDS
    # "service_score_region_r" resolves via the region-role path above (role "service_score").
    if field.name == "automation_tier":
        # Seated automation tier: an integer in [0, N_AUTOMATION_TIERS - 1] carried as float.
        # prefer tight bounds). Own state is EXACT — no clamp is needed at encode.
        from retail_simulator.core.schema import N_AUTOMATION_TIERS

        return (0.0, float(N_AUTOMATION_TIERS - 1))
    if field.name in (
        "employee_happiness",
        "warehouse_utilization",
        "last_stockout_rate",
    ):
        return _SHARE_BOUNDS
    if field.name == "stockpile":
        return (0.0, n_regions / config.promotion.stockpile_decay)
    if field.name == "expansion":
        return (0.0, float(N_REGIONS_MAX - 1))
    if field.name == "last_profit":
        # Profit can be negative when opex outweighs margin at a given price.
        return (-_MONETARY_BOUND, _MONETARY_BOUND)
    if field.name in (
        "cash",
        "last_revenue",
        "total_regional_demand",
        "stores",
        # Wage actually paid last tick: unbounded-above money magnitude like cash.
        # revenue, so it shares the same generous non-negative bound.
        "wage_level",
    ):
        # Per-region demand/other-present fields share this bound via the region-role path.
        return (0.0, _MONETARY_BOUND)

    # Fallback: SELF/MARKET magnitudes get the generous bound, so no field is ever unbounded.
    # unknown field is never left unbounded.
    if field.group in (ObservationGroup.SELF, ObservationGroup.MARKET):
        return (0.0, _MONETARY_BOUND)
    return (0.0, _MONETARY_BOUND)


def build_observation_space(config: CoreConfig | None = None) -> spaces.Box:
    """Build the observation Box (one entry per observation field)."""
    if config is None:
        config = CoreConfig.default()

    n_regions = len(config.demand.regions) if config.demand.regions else 1
    fields = observation_schema(n_regions)

    lows = np.array(
        [_observation_field_bounds(f, config, n_regions)[0] for f in fields],
        dtype=OBS_DTYPE,
    )
    highs = np.array(
        [_observation_field_bounds(f, config, n_regions)[1] for f in fields],
        dtype=OBS_DTYPE,
    )
    return spaces.Box(low=lows, high=highs, shape=(len(fields),), dtype=OBS_DTYPE)


def build_action_space(schema: ActionSchema | None = None) -> spaces.Box:
    """Build the flat action Box from the enabled levers.

    Continuous levers contribute their [low, high]; discrete levers contribute a
    logits block (argmax-decoded by ``action_wrapper``).
    """
    if schema is None:
        schema = default_action_schema()
    layout = flatten_action_space_layout(schema)

    lows = np.empty(layout.total_dim, dtype=OBS_DTYPE)
    highs = np.empty(layout.total_dim, dtype=OBS_DTYPE)
    for slot in layout.levers:
        if slot.spec.kind is LeverKind.CONTINUOUS:
            assert slot.spec.low is not None and slot.spec.high is not None
            lows[slot.offset] = slot.spec.low
            highs[slot.offset] = slot.spec.high
        else:
            # Logits are unbounded reals (decoded by argmax); the monetary bound is a finite proxy.
            # (decoded by argmax), so the Box spans the full float range here.
            lows[slot.offset : slot.offset + slot.width] = -_MONETARY_BOUND
            highs[slot.offset : slot.offset + slot.width] = _MONETARY_BOUND
    return spaces.Box(low=lows, high=highs, shape=(layout.total_dim,), dtype=OBS_DTYPE)
