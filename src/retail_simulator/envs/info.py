"""Shared info-dict enrichment for the Gym and PettingZoo adapters."""

from __future__ import annotations

from typing import Any

from retail_simulator.core.config import CoreConfig
from retail_simulator.core.state import WorldState


def _stockout_rate_from_service_score_change(
    prior: float,
    next_score: float,
    *,
    decay: float,
    lost_penalty: float,
) -> float:
    """Invert the service-score EMA recurrence to recover the stockout rate."""
    if lost_penalty <= 0.0:
        # No stockouts are possible at the default config (service_level=1.0, cogs_premium=0); report 0.0.
        return 0.0
    raw = next_score - (1.0 - decay) * prior
    # raw = decay * (1 - lost_penalty * sr); solve for sr.
    sr = (1.0 - raw / decay) / lost_penalty
    # Clamp to [0, 1]: the inversion may exceed bounds if the EMA saturated.
    return float(min(max(sr, 0.0), 1.0))


def enrich_agent_info(
    base_info: dict[str, Any],
    state: WorldState,
    seat_index: int,
    episode_step: int,
    *,
    marketing_spend: float,
    assortment: float,
    promotion: float,
    expansion_action: float,
    research_action: float,
    service_level_action: float = 1.0,
    prior_service_score_per_region: tuple[float, ...] | None = None,
    config: CoreConfig | None = None,
    automation_choice: float = 0.0,
    prior_automation_tier: int | None = None,
    loyalty_spend_action: float = 0.0,
    wage_spend_action: float = 0.0,
    warehouse_invest_action: float = 0.0,
) -> dict[str, Any]:
    """Return the agent's enriched info dict: seated state, decoded actions, stockout rates."""
    me = state.retailers[seat_index]
    stockpile = sum(region.stockpile for region in state.regions)
    # Seated fidelity is the seam's authority (0.0 when no view is seated).
    research_fidelity = state.research_fidelity.get(seat_index, 0.0)
    # Own seated per-region service_score (exact); pad with the 1.0 fixed point.
    service_score = me.service_score_per_region
    service_score_region_0 = float(service_score[0]) if len(service_score) > 0 else 1.0
    service_score_region_1 = float(service_score[1]) if len(service_score) > 1 else 1.0

    if prior_service_score_per_region is None or config is None:
        stockout_rate_region_0 = 0.0
        stockout_rate_region_1 = 0.0
    else:
        decay = config.scm.service_score_decay
        lost_penalty = config.scm.lost_sales_share_penalty
        prior_0 = (
            prior_service_score_per_region[0] if len(prior_service_score_per_region) > 0 else 1.0
        )
        prior_1 = (
            prior_service_score_per_region[1] if len(prior_service_score_per_region) > 1 else 1.0
        )
        stockout_rate_region_0 = _stockout_rate_from_service_score_change(
            prior_0, service_score_region_0, decay=decay, lost_penalty=lost_penalty
        )
        stockout_rate_region_1 = _stockout_rate_from_service_score_change(
            prior_1, service_score_region_1, decay=decay, lost_penalty=lost_penalty
        )

    # Seated automation tier: the value the next tick's demand will read.
    automation_tier_seated = int(me.automation_tier)
    # Decoded target tier before the upgrade gate resolved it (reporting-only).
    last_automation_tier_action = int(me.last_automation_tier_action)
    # Cash debited this tick: the differential capex the upgrade gate charged.
    if config is None or prior_automation_tier is None:
        automation_capex = 0.0
    else:
        capex_per_tier = config.automation.capex_per_tier
        seated_safe = (
            automation_tier_seated if 0 <= automation_tier_seated < len(capex_per_tier) else 0
        )
        prior_safe = (
            prior_automation_tier if 0 <= prior_automation_tier < len(capex_per_tier) else 0
        )
        differential = float(capex_per_tier[seated_safe]) - float(capex_per_tier[prior_safe])
        automation_capex = max(differential, 0.0)

    info = dict(base_info)
    info.update(
        {
            "cash": float(me.cash),
            "revenue": float(me.last_revenue),
            "profit": float(me.last_profit),
            "market_share": float(me.last_market_share),
            "loyalty": float(me.loyalty_stock),
            # Total open stores across regions.
            "stores": int(sum(me.stores_per_region)),
            "tick": int(state.tick),
            "episode_step": int(episode_step),
            "awareness": float(me.awareness),
            "marketing_spend": float(marketing_spend),
            "assortment": float(assortment),
            "promotion": float(promotion),
            "stockpile": float(stockpile),
            "expansion_action": float(expansion_action),
            # Decoded research spend this tick + seated perception fidelity.
            "research_action": float(research_action),
            "research_fidelity": float(research_fidelity),
            # Decoded SCM action this tick + seated per-region service_score +
            # per-region stockout rate recovered from the EMA recurrence.
            "service_level_action": float(service_level_action),
            "service_score_region_0": service_score_region_0,
            "service_score_region_1": service_score_region_1,
            "stockout_rate_region_0": stockout_rate_region_0,
            "stockout_rate_region_1": stockout_rate_region_1,
            "automation_tier_action": float(automation_choice),
            "automation_tier": automation_tier_seated,
            "last_automation_tier_action": last_automation_tier_action,
            "automation_capex": automation_capex,
            "loyalty_spend_action": float(loyalty_spend_action),
            # Decoded wage/warehouse actions this tick + seated happiness/utilization/
            # stockout (own state, exact).
            "wage_spend_action": float(wage_spend_action),
            "warehouse_invest_action": float(warehouse_invest_action),
            "employee_happiness": float(me.employee_happiness),
            "warehouse_utilization": float(me.last_warehouse_utilization),
            "last_stockout_rate": float(me.last_stockout_rate),
        }
    )
    # action_mask is computed by the seam and carried in base_info; setdefault never overwrites it.
    info.setdefault("action_mask", None)
    return info
