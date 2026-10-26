"""Accounting engine: revenue, COGS, opex, profit, and cash.

Turns the units sold by demand.resolve_demand into per-retailer financial outcomes, the next
cash position, and the per-region loyalty update. Pure ledger; never mutates state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import CoreConfig
from retail_simulator.core.demand import DemandResult, per_region_stock
from retail_simulator.core.state import WorldState

# Reference unit price; price_index is a multiplier on it.
BASE_PRICE: float = 1.0


def _as_float_array(
    value: npt.NDArray[np.float64] | None, like: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Coerce an optional [K] array to float64, defaulting to zeros shaped like ``like``."""
    return np.zeros_like(like) if value is None else np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class AccountingResult:
    """Per-retailer financial outcome of one tick (regions summed)."""

    revenue: npt.NDArray[np.float64]  # [K]
    cogs: npt.NDArray[np.float64]  # [K]
    opex: npt.NDArray[np.float64]  # [K]
    marketing: npt.NDArray[np.float64]  # [K]
    capex: npt.NDArray[np.float64]  # [K]
    profit: npt.NDArray[np.float64]  # [K]
    new_cash: npt.NDArray[np.float64]  # [K]
    market_share: npt.NDArray[np.float64]  # [R, K]
    new_loyalty: npt.NDArray[np.float64]  # [R, K]
    overhead: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    cash_negative: npt.NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))


def apply_accounting(
    state: WorldState,
    demand_result: DemandResult,
    config: CoreConfig,
    capex: npt.NDArray[np.float64] | None = None,
    automation_capex: npt.NDArray[np.float64] | None = None,
    wage_bill: npt.NDArray[np.float64] | None = None,
    overhead: npt.NDArray[np.float64] | None = None,
    warehouse_spend: npt.NDArray[np.float64] | None = None,
) -> AccountingResult:
    """Compute one tick of revenue, COGS, opex, profit, and new cash."""
    cost_cfg = config.cost
    alpha = config.loyalty.alpha

    units = np.asarray(demand_result.units, dtype=np.float64)  # [R, K]
    price_indices = demand_result.price_indices  # [K]
    cash = np.array([r.cash for r in state.retailers], dtype=np.float64)
    # Per-store opex scales with the number of open stores.
    stores_total = np.array([sum(r.stores_per_region) for r in state.retailers], dtype=np.float64)
    # Seated per-region loyalty [R, K].
    n_regions = len(state.regions)
    loyalty_per_region = per_region_stock(state, "loyalty_per_region", n_regions)  # [R, K]

    # Sum per-region revenue to a per-retailer total.
    revenue = (units * price_indices[None, :] * BASE_PRICE).sum(axis=0)  # [K]
    # COGS uses the per-retailer effective fraction from DemandResult.
    cogs_fraction = np.asarray(demand_result.cogs_fraction, dtype=np.float64)  # [K]
    cogs = (cogs_fraction[None, :] * units * BASE_PRICE).sum(axis=0)  # [K]

    # opex = per-store opex + assortment cost (once) + promo cost (per unit sold).
    assortment = np.asarray(demand_result.assortment, dtype=np.float64)  # [K]
    promotion = np.asarray(demand_result.promotion, dtype=np.float64)  # [K]
    promo_cost = (config.promotion.promo_cost_per_unit * promotion[None, :] * units).sum(
        axis=0
    )  # [K]
    # Research opex: cost_per_fidelity * research spend, once per retailer.
    # The ONLY economic effect of the research lever — deterministic, no RNG.
    research = np.asarray(demand_result.research, dtype=np.float64)  # [K]
    research_cost = config.research.cost_per_fidelity * research  # [K]
    loyalty_spend_arr = np.asarray(demand_result.loyalty_spend, dtype=np.float64)  # [K]
    loyalty_program_cost = config.loyalty_program.cost_per_unit_spend * loyalty_spend_arr  # [K]
    opex = (
        cost_cfg.fixed_opex_per_store * stores_total
        + config.assortment.assortment_cost_per_unit * assortment
        + promo_cost
        + research_cost
        + loyalty_program_cost
    )
    marketing = np.asarray(demand_result.marketing_spend, dtype=np.float64)  # [K]

    # Total capex = expansion + automation; both arrive pre-gated by the
    # affordability check (no double-application).
    expansion_capex_arr = _as_float_array(capex, revenue)
    automation_capex_arr = _as_float_array(automation_capex, revenue)
    capex_arr = expansion_capex_arr + automation_capex_arr

    wage_bill_arr = _as_float_array(wage_bill, revenue)
    overhead_arr = _as_float_array(overhead, revenue)
    warehouse_spend_arr = _as_float_array(warehouse_spend, revenue)
    cash_budget_enabled = config.wage.cash_budget_enabled
    # Any ledger input (or a nonzero overdraft rate) takes the ledger path.
    ledger_active = (
        wage_bill is not None
        or overhead is not None
        or cash_budget_enabled
        or warehouse_spend is not None
        or config.wage.overdraft_rate != 0.0
    )

    # Negative-cash signal; empty on the default path.
    cash_negative = np.zeros(0, dtype=bool)
    if not ledger_active:
        # Default path (byte-identical float-op order).
        profit = revenue - cogs - opex - marketing
        new_cash = cash + profit - capex_arr
    else:
        # Ledger path: levers arrive pre-clamped and capex is charged in full, as gated.
        assortment_cost = config.assortment.assortment_cost_per_unit * assortment
        non_clamped_opex = research_cost + loyalty_program_cost
        opex = (
            cost_cfg.fixed_opex_per_store * stores_total
            + assortment_cost
            + promo_cost
            + non_clamped_opex
            + wage_bill_arr
            + warehouse_spend_arr
            + overhead_arr
        )
        # Interest is charged only on drawn credit, capped at credit_limit.
        overdraft_rate = config.wage.overdraft_rate
        if overdraft_rate != 0.0:
            drawn = np.minimum(np.maximum(0.0, -cash), config.wage.credit_limit)
            opex = opex + overdraft_rate * drawn
        profit = revenue - cogs - opex - marketing
        new_cash = cash + profit - capex_arr
        cash_negative = new_cash < 0.0

    market_share = demand_result.shares  # [R, K]
    # Per-region loyalty accrual: EMA of seated stock and this-tick share.
    new_loyalty = alpha * loyalty_per_region + (1.0 - alpha) * market_share  # [R, K]

    return AccountingResult(
        revenue=revenue,
        cogs=cogs,
        opex=opex,
        marketing=marketing,
        capex=capex_arr,
        profit=profit,
        new_cash=new_cash,
        market_share=market_share,
        new_loyalty=new_loyalty,
        overhead=overhead_arr,
        cash_negative=cash_negative,
    )
