"""Affordability: how much of its intended spend each retailer can actually pay.

With the cash-budget gate on, ``World.step`` derives a per-retailer clamp in [0, 1] and scales
the discretionary levers by it so cost and demand effect move together; capex and promotion are
never scaled. With the gate off, every clamp is 1.0 and the pre-existing sequence runs untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import CoreConfig
from retail_simulator.core.state import WorldState


@dataclass(frozen=True)
class DecodedLevers:
    """Per-retailer levers decoded from the joint action, one [K] array each.

    Decode semantics (validation, NPC defaults, which levers are contemporaneous
    vs lagged vs perception-only) live in ``demand.py::_decode_levers``.
    """

    price_indices: npt.NDArray[np.float64]  # [K]
    spend_fraction: npt.NDArray[np.float64]  # [K]
    assortment: npt.NDArray[np.float64]  # [K]
    promotion: npt.NDArray[np.float64]  # [K]
    expansion: npt.NDArray[np.float64]  # [K]
    research: npt.NDArray[np.float64]  # [K]
    service_level: npt.NDArray[np.float64]  # [K]
    automation_tier: npt.NDArray[np.int64]  # [K]
    loyalty_spend: npt.NDArray[np.float64]  # [K]
    wage_spend: npt.NDArray[np.float64]  # [K]
    warehouse_invest: npt.NDArray[np.float64]  # [K]


def gated_expansion_capex(
    state: WorldState,
    choices: npt.NDArray[np.float64],
    config: CoreConfig,
) -> tuple[npt.NDArray[np.float64], list[tuple[int, ...]]]:
    """Charge expansion capex only where the store is affordable from pre-capex cash."""
    n_regions = len(state.regions)
    capex_cfg = config.expansion.expansion_capex
    capex = np.zeros(len(state.retailers), dtype=np.float64)
    next_stores_per_region: list[tuple[int, ...]] = []
    for idx, retailer in enumerate(state.retailers):
        choice = int(choices[idx])
        stores = list(retailer.stores_per_region)
        # Pad to n_regions defensively (1-region fallback states).
        while len(stores) < n_regions:
            stores.append(0)
        valid_open = (
            1 <= choice < n_regions
            and stores[choice] == 0
            and retailer.cash >= capex_cfg  # PRE-capex (start-of-tick) cash
        )
        if valid_open:
            capex[idx] = capex_cfg
            stores[choice] = 1  # open-only here; restructuring is the only path that closes a store
        next_stores_per_region.append(tuple(stores))
    return capex, next_stores_per_region


def gated_automation_capex(
    state: WorldState,
    targets: npt.NDArray[np.int64],
    config: CoreConfig,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Charge only the affordable automation tier differential; tier never decreases."""
    capex_per_tier = config.automation.capex_per_tier
    n = len(state.retailers)
    automation_capex = np.zeros(n, dtype=np.float64)
    new_tier = np.zeros(n, dtype=np.int64)
    reported_target = np.zeros(n, dtype=np.int64)
    for idx, retailer in enumerate(state.retailers):
        target = int(targets[idx])
        prev = retailer.automation_tier
        # Clamp out-of-range targets: keep the previous tier.
        if not 0 <= target < len(capex_per_tier):
            target = prev
        differential = capex_per_tier[target] - capex_per_tier[prev]
        monotone = target >= prev
        affordable = retailer.cash >= differential
        if monotone and affordable:
            new_tier[idx] = target
            automation_capex[idx] = differential
        else:
            new_tier[idx] = prev
            automation_capex[idx] = 0.0
        reported_target[idx] = target
    return automation_capex, new_tier, reported_target


def intended_discretionary_spend(
    levers: DecodedLevers,
    wage_bill: npt.NDArray[np.float64],
    config: CoreConfig,
) -> npt.NDArray[np.float64]:
    """Total intended per-tick spend from the discretionary levers plus the wage bill."""
    marketing_exponent = config.marketing.marketing_cost_exponent
    if marketing_exponent == 1.0:
        marketing_term = levers.spend_fraction * config.marketing.cost_per_unit_spend
    else:
        marketing_term = (
            levers.spend_fraction**marketing_exponent * config.marketing.cost_per_unit_spend
        )
    return (
        marketing_term
        + config.assortment.assortment_cost_per_unit * levers.assortment
        + config.research.cost_per_fidelity * levers.research
        + config.loyalty_program.cost_per_unit_spend * levers.loyalty_spend
        + wage_bill
        + levers.warehouse_invest * config.warehouse.cost_per_unit_invest
    )


def scale_discretionary_levers(
    levers: DecodedLevers,
    clamp: npt.NDArray[np.float64],
) -> DecodedLevers:
    """Scale the discretionary levers by the per-retailer clamp.

    Scaled: spend_fraction, assortment, research, loyalty_spend, wage_spend,
    warehouse_invest, the levers whose value is a purchase. Unscaled:
    price_indices, promotion, expansion, automation_tier. The wage bill cannot be
    re-derived from the scaled wage_spend alone. Returns a new instance; the input
    is not mutated.
    """
    return DecodedLevers(
        price_indices=levers.price_indices,
        spend_fraction=levers.spend_fraction * clamp,
        assortment=levers.assortment * clamp,
        promotion=levers.promotion,
        expansion=levers.expansion,
        research=levers.research * clamp,
        service_level=levers.service_level,
        automation_tier=levers.automation_tier,
        loyalty_spend=levers.loyalty_spend * clamp,
        wage_spend=levers.wage_spend * clamp,
        warehouse_invest=levers.warehouse_invest * clamp,
    )


def compute_spend_clamp(
    cash: npt.NDArray[np.float64],
    intended_discretionary: npt.NDArray[np.float64],
    committed_capex: npt.NDArray[np.float64],
    credit_limit: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Per-retailer clamp in [0, 1]: the affordable fraction of intended spend."""
    if credit_limit == 0.0:
        # credit_limit == 0 keeps the original float-op order.
        available = np.maximum(cash - committed_capex, 0.0)
    else:
        available = np.maximum(cash + credit_limit - committed_capex, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(intended_discretionary > 0.0, available / intended_discretionary, 1.0)
    return np.minimum(ratio, 1.0)
