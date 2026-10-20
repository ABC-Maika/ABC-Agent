"""Loyalty-axis sweep implementation."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import CoreConfig, LoyaltyProgramConfig
from retail_simulator.core.state import BRAND_LOYAL, SegmentParams, default_segments
from retail_simulator.core.world import AGENT_INDEX, World
from retail_simulator.envs.sweep.axes import _seat_present_both
from retail_simulator.envs.sweep.rows import (
    _BOUNDARY_ATOL,
    LOYALTY_GATE_BRAND_LOYAL_BETA_PROGRAM,
    LOYALTY_GATE_COST_PER_UNIT_SPEND,
    LOYALTY_HIGH,
    LOYALTY_LOW,
    PROMO_HIGH,
    PROMO_LOW,
    SweepRow1DLoyalty,
    _mean_and_std_error,
)


def loyalty_gate_segments(
    *, beta_program_loyal: float = LOYALTY_GATE_BRAND_LOYAL_BETA_PROGRAM
) -> dict[str, SegmentParams]:
    """The demand segments the loyalty leg runs on."""
    segments = default_segments()
    segments[BRAND_LOYAL] = replace(segments[BRAND_LOYAL], beta_program=beta_program_loyal)
    return segments


def loyalty_gate_config(
    base_config: CoreConfig | None = None,
    *,
    beta_program_loyal: float = LOYALTY_GATE_BRAND_LOYAL_BETA_PROGRAM,
    cost_per_unit_spend: float = LOYALTY_GATE_COST_PER_UNIT_SPEND,
) -> CoreConfig:
    """A config with the loyalty-program scenario enabled."""
    base = base_config if base_config is not None else CoreConfig.default()
    return replace(
        base,
        loyalty_program=LoyaltyProgramConfig(cost_per_unit_spend=cost_per_unit_spend),
        segments=loyalty_gate_segments(beta_program_loyal=beta_program_loyal),
    )


def _loyalty_action(
    loyalty_spend: float,
    *,
    price: float,
    spend: float,
    assortment: float,
    promotion: float,
) -> npt.NDArray[np.float32]:
    """Build the flat action holding the pinned optima with ``loyalty_spend`` set."""
    return np.array(
        [
            price,
            spend,
            assortment,
            promotion,
            0.0,
            0.0,
            *([0.0] * 30),
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            loyalty_spend,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _loyalty_samples_for_cell(
    loyalty_spend: float,
    *,
    config: CoreConfig,
    price: float,
    spend: float,
    assortment: float,
    promotion: float,
    ticks: int,
    burn_in_ticks: int,
    seed: int,
    start_present_both: bool = False,
) -> tuple[list[float], list[float]]:
    """Per-tick agent-profit samples for one loyalty_spend cell over ``seeds``."""
    world = World(config=config, seed=seed)
    state = world.reset(seed)
    if start_present_both:
        _seat_present_both(state)
    fixed_action = _loyalty_action(
        loyalty_spend, price=price, spend=spend, assortment=assortment, promotion=promotion
    )

    profits: list[float] = []
    market_shares: list[float] = []
    for tick in range(ticks):
        result = world.step(state, {AGENT_INDEX: fixed_action})
        state = result.next_state
        if tick < burn_in_ticks:
            continue
        agent = state.retailers[AGENT_INDEX]
        profits.append(float(agent.last_profit))
        market_shares.append(float(agent.last_market_share))
    return profits, market_shares


def _sweep_loyalty_axis(
    config: CoreConfig,
    *,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    fixed_promotion: float,
    loyalty_spends: tuple[float, ...],
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
    start_present_both: bool,
) -> list[SweepRow1DLoyalty]:
    """Sweep the loyalty-spend grid over ``seeds``."""
    rows: list[SweepRow1DLoyalty] = []
    for loyalty_spend in loyalty_spends:
        l_f = float(loyalty_spend)
        all_profits: list[float] = []
        all_shares: list[float] = []
        for seed in seeds:
            profits, shares = _loyalty_samples_for_cell(
                l_f,
                config=config,
                price=fixed_price,
                spend=fixed_spend,
                assortment=fixed_assortment,
                promotion=fixed_promotion,
                ticks=ticks_per_point,
                burn_in_ticks=burn_in_ticks,
                seed=seed,
                start_present_both=start_present_both,
            )
            all_profits.extend(profits)
            all_shares.extend(shares)

        mean_profit, std_error, n = _mean_and_std_error(all_profits)
        rows.append(
            SweepRow1DLoyalty(
                loyalty_spend=l_f,
                fixed_price=fixed_price,
                fixed_spend=fixed_spend,
                fixed_assortment=fixed_assortment,
                fixed_promotion=fixed_promotion,
                mean_profit=mean_profit,
                std_error=std_error,
                mean_market_share=float(np.asarray(all_shares, dtype=np.float64).mean()),
                n_samples=n,
            )
        )
    return rows


def _loyalty_promotion_swap_action(
    loyalty_spend: float,
    promotion: float,
    *,
    price: float,
    spend: float,
    assortment: float,
) -> npt.NDArray[np.float32]:
    """Build the flat action with loyalty/promotion swapped by the delta amounts."""
    return np.array(
        [
            price,
            spend,
            assortment,
            promotion,
            0.0,
            0.0,
            *([0.0] * 30),
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            loyalty_spend,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _sweep_loyalty_orthogonality_probe(
    config: CoreConfig,
    *,
    base_loyalty: float,
    base_promotion: float,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    swap_deltas: tuple[float, ...],
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
    start_present_both: bool,
) -> tuple[float, float, tuple[tuple[float, float, float, float], ...]]:
    """Run the loyalty<->promotion substitution cells at the joint optima; return (base_profit, best_cell, swap_profits)."""

    def _run(loyalty: float, promotion: float) -> tuple[float, float]:
        all_profits: list[float] = []
        for seed in seeds:
            world = World(config=config, seed=seed)
            state = world.reset(seed)
            if start_present_both:
                _seat_present_both(state)
            action = _loyalty_promotion_swap_action(
                loyalty,
                promotion,
                price=fixed_price,
                spend=fixed_spend,
                assortment=fixed_assortment,
            )
            for tick in range(ticks_per_point):
                result = world.step(state, {AGENT_INDEX: action})
                state = result.next_state
                if tick < burn_in_ticks:
                    continue
                all_profits.append(float(state.retailers[AGENT_INDEX].last_profit))
        mean, se, _n = _mean_and_std_error(all_profits)
        return mean, se

    base_mean, base_se = _run(base_loyalty, base_promotion)

    swap_cells: list[tuple[float, float, float, float]] = []
    seen: set[tuple[float, float]] = set()
    for delta in swap_deltas:
        d = float(delta)
        if d <= 0.0:
            continue
        candidates = (
            (
                min(max(base_loyalty + d, LOYALTY_LOW), LOYALTY_HIGH),
                min(max(base_promotion - d, PROMO_LOW), PROMO_HIGH),
            ),
            (
                min(max(base_loyalty - d, LOYALTY_LOW), LOYALTY_HIGH),
                min(max(base_promotion + d, PROMO_LOW), PROMO_HIGH),
            ),
        )
        for new_loyalty, new_promotion in candidates:
            key = (round(new_loyalty, 6), round(new_promotion, 6))
            if (
                math.isclose(new_loyalty, base_loyalty, abs_tol=_BOUNDARY_ATOL)
                and math.isclose(new_promotion, base_promotion, abs_tol=_BOUNDARY_ATOL)
            ) or key in seen:
                continue
            seen.add(key)
            mean, se = _run(new_loyalty, new_promotion)
            swap_cells.append((float(new_loyalty), float(new_promotion), float(mean), float(se)))

    return float(base_mean), float(base_se), tuple(swap_cells)
