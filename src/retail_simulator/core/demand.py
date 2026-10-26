"""Vectorized demand resolution per region (Huff/MNL).

Each region is resolved independently: per-segment softmax shares over the
retailers present there, a constant-elasticity pie, and a lognormal noise
draw, then service, supplier, and warehouse capacity caps. Awareness,
loyalty, and stockpiles are read as lagged per-region stocks; the levers
(price, marketing, assortment, promotion, ...) are read for this tick.
Pure numpy, no mutation, no hidden RNG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from retail_simulator.core.affordability import DecodedLevers
from retail_simulator.core.config import CoreConfig, WarehouseConfig
from retail_simulator.core.encoding import decode_action
from retail_simulator.core.schema import ActionSchema
from retail_simulator.core.state import WorldState


@dataclass(frozen=True)
class DemandResult:
    """One tick of demand output (region x retailer tensors)."""

    units: npt.NDArray[np.float64]  # [R, K]
    shares: npt.NDArray[np.float64]  # [R, K]
    total_demand: npt.NDArray[np.float64]  # [R]
    price_indices: npt.NDArray[np.float64]  # [K]
    marketing_spend: npt.NDArray[np.float64]  # [K]
    spend_fractions: npt.NDArray[np.float64]  # [K]
    assortment: npt.NDArray[np.float64]  # [K]
    promotion: npt.NDArray[np.float64]  # [K]
    # Decoded expansion choice per retailer this tick; NPCs/absent -> 0.
    expansion: npt.NDArray[np.float64]  # [K]
    # Presence mask [R, K]: stores_per_region[r, k] > 0.
    presence: npt.NDArray[np.bool_]  # [R, K]
    # Decoded research spend; perception and opex only, no demand effect.
    research: npt.NDArray[np.float64]  # [K]
    # Decoded SCM service level; drives the fill-rate cap and COGS.
    service_level: npt.NDArray[np.float64]  # [K]
    # Effective COGS fraction this tick (base + service premium).
    cogs_fraction: npt.NDArray[np.float64]  # [K]
    # Per-region stockout rate in [0, 1]; input to the service seater.
    stockout_rate: npt.NDArray[np.float64]  # [R, K]
    # Target automation tier requested this tick (0-2); the seam gates it.
    automation_tier_action: npt.NDArray[np.int64]  # [K]
    loyalty_spend: npt.NDArray[np.float64]  # [K]
    # Decoded wage spend in [0, 1]; consumed by the seam.
    wage_spend: npt.NDArray[np.float64]  # [K]
    # Decoded warehouse invest in [0, 1]; consumed by the seam.
    warehouse_invest: npt.NDArray[np.float64]  # [K]
    # Demand-weighted lost-sales fraction across regions (reporting only).
    lost_sales_fraction: npt.NDArray[np.float64]  # [K]
    # Affordability clamp that scaled the levers this tick (reporting only).
    spend_clamp: npt.NDArray[np.float64]  # [K]


def _retailer_price_indices(state: WorldState) -> npt.NDArray[np.float64]:
    """Price index per retailer, in agent-index order."""
    return np.array([r.price_index for r in state.retailers], dtype=np.float64)


def _decode_levers(
    state: WorldState,
    joint_action: dict[int, npt.NDArray[np.float32]] | None,
    schema: ActionSchema | None,
) -> DecodedLevers:
    """Decode every seat's action into per-retailer lever arrays."""
    prices = _retailer_price_indices(state)
    n = len(state.retailers)
    spend = np.zeros(n, dtype=np.float64)
    assortment = np.zeros(n, dtype=np.float64)
    promotion = np.zeros(n, dtype=np.float64)
    expansion = np.zeros(n, dtype=np.float64)
    research = np.zeros(n, dtype=np.float64)
    service_level = np.ones(n, dtype=np.float64)
    automation = np.zeros(n, dtype=np.int64)
    loyalty_spend = np.zeros(n, dtype=np.float64)
    wage_spend = np.zeros(n, dtype=np.float64)
    warehouse_invest = np.zeros(n, dtype=np.float64)
    # Seats with no action keep the registry defaults above.
    for idx, retailer in enumerate(state.retailers):
        flat: npt.NDArray[np.float32] | tuple[float, ...] | None = (
            joint_action.get(idx) if joint_action else None
        )
        if flat is None:
            # Only scripted seats carry a pending action.
            if not (retailer.is_npc and retailer.pending_action is not None):
                continue
            flat = retailer.pending_action
        decoded = decode_action(flat, schema)
        prices[idx] = decoded.levers["price_index"]
        # Levers disabled in the schema are absent; fall back to the default.
        spend[idx] = decoded.levers.get("marketing", 0.0)
        assortment[idx] = decoded.levers.get("assortment", 0.0)
        promotion[idx] = decoded.levers.get("promotion", 0.0)
        expansion[idx] = decoded.levers.get("expansion", 0.0)
        research[idx] = decoded.levers.get("research", 0.0)
        service_level[idx] = decoded.levers.get("service_level", 1.0)
        automation[idx] = int(decoded.levers.get("automation", 0.0))
        loyalty_spend[idx] = decoded.levers.get("loyalty_spend", 0.0)
        wage_spend[idx] = decoded.levers.get("wage_spend", 0.0)
        warehouse_invest[idx] = decoded.levers.get("warehouse_invest", 0.0)
    return DecodedLevers(
        price_indices=prices,
        spend_fraction=spend,
        assortment=assortment,
        promotion=promotion,
        expansion=expansion,
        research=research,
        service_level=service_level,
        automation_tier=automation,
        loyalty_spend=loyalty_spend,
        wage_spend=wage_spend,
        warehouse_invest=warehouse_invest,
    )


def _present_others_mean(
    values: npt.NDArray[np.float64],
    presence: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Mean of the other present retailers' values, per region."""
    pres = presence.astype(np.float64)  # [R, K]
    masked = values[None, :] * pres  # [R, K], absent -> 0
    sum_present = masked.sum(axis=1, keepdims=True)  # [R, 1]
    count_present = pres.sum(axis=1, keepdims=True)  # [R, 1]
    # Drop self from both numerator and denominator.
    others_sum = sum_present - masked  # [R, K]
    others_count = count_present - pres  # [R, K]
    # Sole-present (or absent) retailers fall back to their own value.
    return np.where(
        others_count > 0.0,
        others_sum / np.maximum(others_count, 1.0),
        values[None, :],
    )


def _reference_prices(
    price_indices: npt.NDArray[np.float64],
    presence: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Mean present-rival price index, per region."""
    return _present_others_mean(price_indices, presence)


def _reference_promotion(
    promotion: npt.NDArray[np.float64],
    presence: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Mean present-rival promotion, per region."""
    return _present_others_mean(promotion, presence)


def _segment_region_shares(
    price_indices: npt.NDArray[np.float64],
    awareness_per_region: npt.NDArray[np.float64],
    loyalty_per_region: npt.NDArray[np.float64],
    assortment: npt.NDArray[np.float64],
    promotion: npt.NDArray[np.float64],
    presence: npt.NDArray[np.bool_],
    beta_prices: npt.NDArray[np.float64],
    beta_marketings: npt.NDArray[np.float64],
    beta_assortments: npt.NDArray[np.float64],
    beta_promotions: npt.NDArray[np.float64],
    beta_loyalties: npt.NDArray[np.float64],
    baseline: float,
    beta_references: npt.NDArray[np.float64] | None = None,
    service_score_per_region: npt.NDArray[np.float64] | None = None,
    beta_services: npt.NDArray[np.float64] | None = None,
    beta_programs: npt.NDArray[np.float64] | None = None,
    loyalty_spend: npt.NDArray[np.float64] | None = None,
    beta_under_reference_aversions: npt.NDArray[np.float64] | None = None,
    promotion_contest_weight: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Per-segment per-region MNL shares over present retailers."""
    log_price = np.log(price_indices / baseline)  # [K]
    # Broadcast: betas [S,1,1], seated stocks [1,R,K], scalar levers [1,1,K].
    bp = beta_prices[:, None, None]
    bm = beta_marketings[:, None, None]
    ba = beta_assortments[:, None, None]
    bpr = beta_promotions[:, None, None]
    bl = beta_loyalties[:, None, None]
    if promotion_contest_weight == 0.0:
        promo_term = promotion[None, None, :]
    else:
        promo_ref = _reference_promotion(promotion, presence)  # [R, K]
        promo_effective = promotion[None, :] - promotion_contest_weight * promo_ref  # [R, K]
        promo_term = promo_effective[None, :, :]
    utility = (
        -bp * log_price[None, None, :]
        + bm * awareness_per_region[None, :, :]
        + ba * assortment[None, None, :]
        + bpr * promo_term
        + bl * loyalty_per_region[None, :, :]
    )  # [S, R, K]
    if beta_references is not None and bool(np.any(beta_references != 0.0)):
        bref = beta_references[:, None, None]  # [S, 1, 1]
        ref = _reference_prices(price_indices, presence)  # [R, K]
        penalty = np.maximum(0.0, np.log(price_indices[None, None, :] / ref[None, :, :]))
        utility = utility - bref * penalty  # over-pricing only
    if beta_under_reference_aversions is not None and bool(
        np.any(beta_under_reference_aversions != 0.0)
    ):
        bunder = beta_under_reference_aversions[:, None, None]  # [S, 1, 1]
        ref_under = _reference_prices(price_indices, presence)  # [R, K]
        under_penalty = np.maximum(
            0.0, np.log(ref_under[None, :, :] / price_indices[None, None, :])
        )
        utility = utility - bunder * under_penalty  # under-pricing only
    if (
        beta_services is not None
        and service_score_per_region is not None
        and bool(np.any(beta_services != 0.0))
    ):
        bsvc = beta_services[:, None, None]  # [S, 1, 1]
        utility = utility + bsvc * service_score_per_region[None, :, :]  # [S, R, K]
    if (
        beta_programs is not None
        and loyalty_spend is not None
        and bool(np.any(beta_programs != 0.0))
    ):
        bprog = beta_programs[:, None, None]  # [S, 1, 1]
        # Same-tick boost: scales seated loyalty by this tick's program spend.
        utility = utility + bprog * (loyalty_spend[None, None, :] * loyalty_per_region[None, :, :])
    utility = np.where(presence[None, :, :], utility, -np.inf)
    shifted = utility - np.max(utility, axis=2, keepdims=True)
    weights = np.exp(shifted)  # absent -> exp(-inf - max) = 0
    return weights / np.sum(weights, axis=2, keepdims=True)


def _presence_matrix(state: WorldState, n_regions: int) -> npt.NDArray[np.bool_]:
    """Presence mask: does retailer k operate in region r."""
    n = len(state.retailers)
    presence = np.zeros((n_regions, n), dtype=bool)
    for k, retailer in enumerate(state.retailers):
        spr = retailer.stores_per_region
        length = min(len(spr), n_regions)
        if length:
            presence[:length, k] = np.asarray(spr[:length]) > 0
    return presence


def per_region_stock(state: WorldState, attr: str, n_regions: int) -> npt.NDArray[np.float64]:
    """Build a [R, K] matrix from a per-region retailer attribute.

    Shorter per-region tuples read 0.0 for missing regions. Public because
    core/accounting.py reuses it across the module boundary.
    """
    n = len(state.retailers)
    out = np.zeros((n_regions, n), dtype=np.float64)
    for k, retailer in enumerate(state.retailers):
        per_region: tuple[float, ...] | tuple[int, ...] = getattr(retailer, attr)
        length = min(len(per_region), n_regions)
        if length:
            out[:length, k] = per_region[:length]
    return out


def _footprint_totals(state: WorldState, n_regions: int) -> npt.NDArray[np.float64]:
    """Total stores per retailer across regions."""
    n = len(state.retailers)
    out = np.zeros(n, dtype=np.float64)
    for k, retailer in enumerate(state.retailers):
        spr = retailer.stores_per_region
        out[k] = sum(spr[r] for r in range(n_regions) if r < len(spr))
    return out


def _mature_store_equivalents(
    state: WorldState, n_regions: int, ramp_ticks: int
) -> npt.NDArray[np.float64]:
    """Stores weighted by ramp progress (age / ramp_ticks)."""
    n = len(state.retailers)
    out = np.zeros(n, dtype=np.float64)
    for k, retailer in enumerate(state.retailers):
        spr = retailer.stores_per_region
        ages = retailer.store_age_per_region
        total = 0.0
        for r in range(n_regions):
            if r >= len(spr) or spr[r] <= 0:
                continue
            age_r = ages[r] if r < len(ages) else 0
            total += spr[r] * min(1.0, age_r / ramp_ticks)
        out[k] = total
    return out


def _capacity_footprint(
    state: WorldState, n_regions: int, warehouse_cfg: WarehouseConfig
) -> npt.NDArray[np.float64]:
    """Mature-store equivalents while ramping, else total stores."""
    if warehouse_cfg.ramp_ticks > 0:
        return _mature_store_equivalents(state, n_regions, warehouse_cfg.ramp_ticks)
    return _footprint_totals(state, n_regions)


def effective_warehouse_capacity(
    config: CoreConfig,
    warehouse_capacity: npt.NDArray[np.float64],
    stores_total: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Warehouse capacity plus any per-store provided capacity."""
    provided = config.warehouse.provided_capacity_per_store
    if provided == 0.0:
        return warehouse_capacity
    return warehouse_capacity + provided * stores_total


def _apply_share_ceiling(
    shares: npt.NDArray[np.float64],
    presence: npt.NDArray[np.bool_],
    cap: float,
) -> npt.NDArray[np.float64]:
    """Cap each retailer's share at ``cap`` and redistribute the excess."""
    if cap >= 1.0:
        return shares
    n_retailers = shares.shape[1]
    s = shares.copy()
    # Capping is permanent: a capped retailer never receives more.
    uncapped = presence.copy()
    for _ in range(n_retailers):
        over = uncapped & (s > cap)
        if not bool(np.any(over)):
            break  # converged
        excess = np.where(over, s - cap, 0.0).sum(axis=1, keepdims=True)  # [R, 1]
        s = np.where(over, cap, s)
        uncapped = uncapped & ~over
        # Redistribute each region's excess over its uncapped present
        # retailers; a fully capped region keeps the clamped (sum < 1) shares.
        recipient_sum = np.where(uncapped, s, 0.0).sum(axis=1, keepdims=True)  # [R, 1]
        safe_recipient_sum = np.where(recipient_sum > 0.0, recipient_sum, 1.0)
        add = np.where(
            uncapped & (recipient_sum > 0.0),
            excess * s / safe_recipient_sum,
            0.0,
        )
        s = s + add
    return s


def resolve_demand(
    state: WorldState,
    joint_action: dict[int, npt.NDArray[np.float32]] | None,
    rng: np.random.Generator,
    config: CoreConfig,
    schema: ActionSchema | None = None,
    levers: DecodedLevers | None = None,
    spend_clamp: npt.NDArray[np.float64] | None = None,
) -> DemandResult:
    """Resolve demand per region: shares, pie, caps, and final units."""
    demand_cfg = config.demand
    pricing_cfg = config.pricing
    marketing_cfg = config.marketing
    promotion_cfg = config.promotion
    cost_cfg = config.cost
    scm_cfg = config.scm
    baseline = pricing_cfg.baseline_price_index

    # The caller may pass pre-decoded (possibly clamp-scaled) levers.
    decoded_levers = _decode_levers(state, joint_action, schema) if levers is None else levers
    price_indices = decoded_levers.price_indices
    spend_fraction = decoded_levers.spend_fraction
    assortment = decoded_levers.assortment
    promotion = decoded_levers.promotion
    expansion = decoded_levers.expansion
    research = decoded_levers.research
    service_level = decoded_levers.service_level
    automation_tier = decoded_levers.automation_tier
    loyalty_spend = decoded_levers.loyalty_spend
    wage_spend = decoded_levers.wage_spend
    warehouse_invest = decoded_levers.warehouse_invest

    regions = state.regions
    n_regions = len(regions)

    presence = _presence_matrix(state, n_regions)
    region_has_presence = presence.any(axis=1)  # [R]

    # Seated per-region stocks (lagged; advanced by world.step).
    awareness_per_region = per_region_stock(state, "awareness_per_region", n_regions)
    loyalty_per_region = per_region_stock(state, "loyalty_per_region", n_regions)
    service_score_per_region = per_region_stock(state, "service_score_per_region", n_regions)

    shares = np.zeros((n_regions, len(state.retailers)), dtype=np.float64)
    suppressed_pie = np.zeros(n_regions, dtype=np.float64)

    region_keys_union: set[str] = set()
    for region in regions:
        region_keys_union.update(region.segment_mix.keys())
    segment_order = {key: position for position, key in enumerate(state.segments)}
    segment_keys: list[str] = sorted(region_keys_union, key=lambda key: segment_order[key])
    beta_prices = np.array(
        [state.segments[key].beta_price for key in segment_keys], dtype=np.float64
    )
    beta_marketings = np.array(
        [state.segments[key].beta_marketing for key in segment_keys], dtype=np.float64
    )
    beta_assortments = np.array(
        [state.segments[key].beta_assortment for key in segment_keys], dtype=np.float64
    )
    beta_promotions = np.array(
        [state.segments[key].beta_promotion for key in segment_keys], dtype=np.float64
    )
    beta_loyalties = np.array(
        [state.segments[key].beta_loyalty for key in segment_keys], dtype=np.float64
    )
    beta_references = np.array(
        [state.segments[key].beta_reference for key in segment_keys], dtype=np.float64
    )
    beta_services = np.array(
        [state.segments[key].beta_service for key in segment_keys], dtype=np.float64
    )
    beta_programs = np.array(
        [state.segments[key].beta_program for key in segment_keys], dtype=np.float64
    )
    beta_under_reference_aversions = np.array(
        [state.segments[key].beta_under_reference_aversion for key in segment_keys],
        dtype=np.float64,
    )
    # Row index of each segment in the batched shares call.
    canonical_index = {key: position for position, key in enumerate(segment_keys)}

    covered = np.flatnonzero(region_has_presence)
    uncovered = np.flatnonzero(~region_has_presence)

    for region_idx in uncovered:
        region = regions[int(region_idx)]
        avg_price = baseline
        base_demand = (
            region.base_regional_demand * (avg_price / baseline) ** demand_cfg.price_elasticity
        )
        suppressed_pie[region_idx] = base_demand * (1.0 / (1.0 + region.stockpile))

    if covered.size:
        # One batched [S, R_covered, K] shares call for all covered regions.
        seg_region_shares_all = _segment_region_shares(
            price_indices,
            awareness_per_region[covered, :],
            loyalty_per_region[covered, :],
            assortment,
            promotion,
            presence[covered, :],
            beta_prices,
            beta_marketings,
            beta_assortments,
            beta_promotions,
            beta_loyalties,
            baseline,
            beta_references,
            service_score_per_region[covered, :],
            beta_services,
            beta_programs=beta_programs,
            loyalty_spend=loyalty_spend,
            beta_under_reference_aversions=beta_under_reference_aversions,
            promotion_contest_weight=promotion_cfg.promotion_contest_weight,
        )  # [S, R_covered, K]

        for i, region_idx in enumerate(covered):
            region = regions[int(region_idx)]
            # Weight per-segment shares by this region's own population
            # weights, summed in the region's native segment order for
            # bit-exact results.
            native_keys = list(region.segment_mix.keys())
            native_weight = np.array(
                [region.segment_mix[key] for key in native_keys], dtype=np.float64
            )
            native_rows = [canonical_index[key] for key in native_keys]
            shares[region_idx, :] = native_weight @ seg_region_shares_all[native_rows, i, :]

            # Pie: constant-elasticity response to this region's present average price.
            present_prices = price_indices[presence[region_idx, :]]
            avg_price = float(np.mean(present_prices))
            base_demand = (
                region.base_regional_demand * (avg_price / baseline) ** demand_cfg.price_elasticity
            )
            suppressed_pie[region_idx] = base_demand * (1.0 / (1.0 + region.stockpile))

    if demand_cfg.max_share_per_region < 1.0:
        shares = _apply_share_ceiling(shares, presence, demand_cfg.max_share_per_region)

    noise = rng.lognormal(mean=0.0, sigma=demand_cfg.noise_sigma, size=n_regions)
    realized_total = suppressed_pie * noise  # [R]

    # Absent retailers have 0 share; the presence multiply is a dtype guard.
    demand_units = realized_total[:, None] * shares * presence.astype(np.float64)  # [R, K]

    fill_rates = np.array(
        [scm_cfg.fill_rate(float(s)) for s in service_level], dtype=np.float64
    )  # [K]
    available_units = realized_total[:, None] * fill_rates[None, :]  # [R, K]
    units = np.minimum(demand_units, available_units)  # [R, K]

    supplier_cfg = config.supplier
    if supplier_cfg.supplier_contention_enabled and np.isfinite(supplier_cfg.supplier_capacity):
        capacity = supplier_cfg.supplier_capacity
        pres_f = presence.astype(np.float64)  # [R, K]
        # Priority weight per present retailer: service_level ** sharpness.
        priority = service_level**supplier_cfg.preference_sharpness  # [K]
        weights = priority[None, :] * pres_f  # [R, K]
        weight_sum = weights.sum(axis=1, keepdims=True)  # [R, 1]
        present_count = pres_f.sum(axis=1, keepdims=True)  # [R, 1]
        no_preference = (weight_sum <= 0.0) & (present_count > 0.0)  # [R, 1]
        weights = np.where(no_preference, pres_f, weights)
        weight_sum = np.where(no_preference, present_count, weight_sum)
        # Split scarce capacity by priority weight over present rivals; a
        # 1.0 placeholder is used only where the true weight sum is 0.
        safe_priority_weight_sum = np.where(weight_sum > 0.0, weight_sum, 1.0)
        capacity_share = np.where(
            weight_sum > 0.0, weights / safe_priority_weight_sum, 0.0
        )  # [R, K]
        supplier_cap = capacity_share * capacity  # [R, K]
        # Cap only in regions where total demanded units exceed capacity.
        regional_demand = demand_units.sum(axis=1, keepdims=True)  # [R, 1]
        contended = regional_demand > capacity  # [R, 1]
        units = np.where(contended, np.minimum(units, supplier_cap), units)  # [R, K]

    warehouse_cfg = config.warehouse
    warehouse_capacity = np.array(
        [retailer.warehouse_capacity for retailer in state.retailers], dtype=np.float64
    )  # [K]
    if warehouse_cfg.throughput_enabled and bool(np.any(np.isfinite(warehouse_capacity))):
        units_pre_wh = units
        # Stores per region, padded to [R, K].
        stores = per_region_stock(state, "stores_per_region", n_regions)  # [R, K]
        # Split each retailer's capacity across its regions by store count.
        stores_total = stores.sum(axis=0, keepdims=True)  # [1, K]
        region_share = np.where(
            stores_total > 0.0, stores / np.maximum(stores_total, 1.0), 0.0
        )  # [R, K]
        stores_total_effective = _capacity_footprint(state, n_regions, warehouse_cfg)
        effective_capacity = effective_warehouse_capacity(
            config, warehouse_capacity, stores_total_effective
        )  # [K]
        # Cap = capacity * region share, computed only where the share is > 0
        # so an inf capacity times a 0 share never produces NaN.
        with np.errstate(invalid="ignore"):
            # inf * 0 would be NaN; the where() below masks it out.
            warehouse_cap = np.where(
                region_share > 0.0, effective_capacity[None, :] * region_share, 0.0
            )  # [R, K]
        units_wh = np.minimum(units_pre_wh, warehouse_cap)  # [R, K] — capped served demand
        # SPILL: units lost to the cap go to present rivals in proportion
        # to spare capacity, single pass (recipients never re-spill).
        lost = np.maximum(0.0, units_pre_wh - units_wh)  # [R, K], >= 0
        total_lost = lost.sum(axis=1, keepdims=True)  # [R, 1]
        spare = np.maximum(0.0, warehouse_cap - units_wh)  # [R, K], >= 0
        weight_spare = np.where(np.isfinite(spare), spare, total_lost)  # [R, K], finite
        weight_sum = weight_spare.sum(axis=1, keepdims=True)  # [R, 1]
        safe_spare_weight_sum = np.where(weight_sum > 0.0, weight_sum, 1.0)
        spill_in_raw = total_lost * np.where(
            weight_sum > 0.0, weight_spare / safe_spare_weight_sum, 0.0
        )  # [R, K]
        # Never let a recipient exceed its own spare; the residual is lost.
        spill_in = np.minimum(spill_in_raw, spare)  # [R, K]
        units = units_wh + spill_in  # [R, K]

    stockout_rate = np.maximum(0.0, demand_units - units) / np.maximum(demand_units, 1.0)

    # Demand-weighted lost-sales fraction per retailer (reporting only).
    lost_units_per_retailer = np.maximum(0.0, demand_units - units).sum(axis=0)  # [K]
    demanded_per_retailer = demand_units.sum(axis=0)  # [K]
    lost_sales_fraction = lost_units_per_retailer / np.maximum(demanded_per_retailer, 1.0)  # [K]

    cogs_fraction = cost_cfg.cogs_fraction * (1.0 + scm_cfg.cogs_premium * service_level)

    automation_cfg = config.automation
    savings_per_tier_arr = np.asarray(
        automation_cfg.savings_per_tier, dtype=np.float64
    )  # [n_tiers]
    automation_tier_prev = np.array(
        [retailer.automation_tier for retailer in state.retailers], dtype=np.int64
    )  # [K]
    if bool(np.any(automation_tier_prev != 0)):
        cogs_fraction = cogs_fraction * (1.0 - savings_per_tier_arr[automation_tier_prev])

    marketing_exponent = marketing_cfg.marketing_cost_exponent
    if marketing_exponent == 1.0:
        marketing_spend = spend_fraction * marketing_cfg.cost_per_unit_spend
    else:
        marketing_spend = spend_fraction**marketing_exponent * marketing_cfg.cost_per_unit_spend
    return DemandResult(
        units=units,
        shares=shares,
        total_demand=realized_total,
        price_indices=price_indices,
        marketing_spend=marketing_spend,
        spend_fractions=spend_fraction,
        assortment=assortment,
        promotion=promotion,
        expansion=expansion,
        presence=presence,
        # Decoded SoT only: no demand effect.
        research=research,
        service_level=service_level,
        cogs_fraction=cogs_fraction,
        stockout_rate=stockout_rate,
        automation_tier_action=automation_tier,
        loyalty_spend=loyalty_spend,
        wage_spend=wage_spend,
        warehouse_invest=warehouse_invest,
        lost_sales_fraction=lost_sales_fraction,
        # Reporting only: the clamp was applied before this call.
        spend_clamp=(
            np.ones(len(state.retailers), dtype=np.float64) if spend_clamp is None else spend_clamp
        ),
    )
