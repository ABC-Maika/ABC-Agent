"""Per-axis sweep implementations: 2-D grid, assortment, promotion, expansion-timing, and research."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import CoreConfig, SeatSpec
from retail_simulator.core.state import (
    PRICE_SENSITIVE,
    SegmentParams,
    WorldState,
    default_segments,
)
from retail_simulator.core.world import AGENT_INDEX, World
from retail_simulator.envs.sweep.rows import (
    _COMPETITOR_PRICE_OBS_NAME,
    ASSORT_HIGH,
    ASSORT_LOW,
    NEVER_EXPAND_T,
    PROMO_HIGH,
    PROMO_LOW,
    RESEARCH_GATE_BETA_REFERENCE,
    RESEARCH_OPPONENT_ARCHETYPE,
    SPEND_HIGH,
    SPEND_LOW,
    ExpansionTimingRow,
    ResearchFidelityRow,
    SweepRow1D,
    SweepRow1DPromotion,
    SweepRow2D,
    _mean_and_std_error,
)


def _seat_present_both(state: WorldState) -> None:
    """Set the agent's presence (and store age) in every region of ``state``."""
    agent = state.retailers[AGENT_INDEX]
    mature_age = agent.store_age_per_region[0] if agent.store_age_per_region else 0
    agent.stores_per_region = tuple(1 for _ in agent.stores_per_region)
    agent.store_age_per_region = tuple(mature_age for _ in agent.store_age_per_region)


def _profit_samples_for_cell(
    price: float,
    spend: float,
    assortment: float,
    promotion: float,
    *,
    config: CoreConfig,
    ticks: int,
    burn_in_ticks: int,
    seed: int,
    start_present_both: bool = False,
) -> tuple[list[float], list[float]]:
    """Per-tick agent-profit samples for one (price, spend) cell over ``seeds``."""
    world = World(config=config, seed=seed)
    state = world.reset(seed)
    if start_present_both:
        _seat_present_both(state)
    fixed_action = np.array(
        [price, spend, assortment, promotion, 0.0, 0.0]
        + [0.0] * 30
        + [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
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


def _default_burn_in_ticks(config: CoreConfig) -> int:
    """The default burn-in ticks from ``config``."""
    awareness_burn_in = math.ceil(3.0 / config.marketing.decay)
    stockpile_burn_in = math.ceil(3.0 / config.promotion.stockpile_decay)
    return max(awareness_burn_in, stockpile_burn_in)


def _sweep_2d(
    config: CoreConfig,
    *,
    grid_points: int,
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
    pinned_assortment: float,
    pinned_promotion: float,
    start_present_both: bool,
) -> list[SweepRow2D]:
    """Run the 2-D price x marketing grid at a pinned assortment + promotion."""
    price_low = config.pricing.min_price_index
    price_high = config.pricing.max_price_index
    price_grid = np.linspace(price_low, price_high, grid_points)
    spend_grid = np.linspace(SPEND_LOW, SPEND_HIGH, grid_points)

    rows: list[SweepRow2D] = []
    for price in price_grid:
        price_f = float(price)
        for spend in spend_grid:
            spend_f = float(spend)
            all_profits: list[float] = []
            all_shares: list[float] = []
            for seed in seeds:
                profits, shares = _profit_samples_for_cell(
                    price_f,
                    spend_f,
                    pinned_assortment,
                    pinned_promotion,
                    config=config,
                    ticks=ticks_per_point,
                    burn_in_ticks=burn_in_ticks,
                    seed=seed,
                    start_present_both=start_present_both,
                )
                all_profits.extend(profits)
                all_shares.extend(shares)

            mean_profit, std_error, n = _mean_and_std_error(all_profits)
            rows.append(
                SweepRow2D(
                    price_index=price_f,
                    spend_fraction=spend_f,
                    mean_profit=mean_profit,
                    std_error=std_error,
                    mean_market_share=float(np.asarray(all_shares, dtype=np.float64).mean()),
                    n_samples=n,
                )
            )
    return rows


def _sweep_assortment_axis(
    config: CoreConfig,
    *,
    fixed_price: float,
    fixed_spend: float,
    pinned_promotion: float,
    grid_points: int,
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
    start_present_both: bool,
) -> list[SweepRow1D]:
    """Run the 1-D assortment axis at the (price, marketing) optimum, promo pinned."""
    assort_grid = np.linspace(ASSORT_LOW, ASSORT_HIGH, grid_points)

    rows: list[SweepRow1D] = []
    for assortment in assort_grid:
        assortment_f = float(assortment)
        all_profits: list[float] = []
        all_shares: list[float] = []
        for seed in seeds:
            profits, shares = _profit_samples_for_cell(
                fixed_price,
                fixed_spend,
                assortment_f,
                pinned_promotion,
                config=config,
                ticks=ticks_per_point,
                burn_in_ticks=burn_in_ticks,
                seed=seed,
                start_present_both=start_present_both,
            )
            all_profits.extend(profits)
            all_shares.extend(shares)

        mean_profit, std_error, n = _mean_and_std_error(all_profits)
        rows.append(
            SweepRow1D(
                assortment=assortment_f,
                fixed_price=fixed_price,
                fixed_spend=fixed_spend,
                mean_profit=mean_profit,
                std_error=std_error,
                mean_market_share=float(np.asarray(all_shares, dtype=np.float64).mean()),
                n_samples=n,
            )
        )
    return rows


def _sweep_promotion_axis(
    config: CoreConfig,
    *,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    grid_points: int,
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
    start_present_both: bool,
) -> list[SweepRow1DPromotion]:
    """Run the 1-D promotion axis at the (price, marketing, assortment) optimum, measured at steady state."""
    promo_grid = np.linspace(PROMO_LOW, PROMO_HIGH, grid_points)

    rows: list[SweepRow1DPromotion] = []
    for promotion in promo_grid:
        promotion_f = float(promotion)
        all_profits: list[float] = []
        all_shares: list[float] = []
        for seed in seeds:
            profits, shares = _profit_samples_for_cell(
                fixed_price,
                fixed_spend,
                fixed_assortment,
                promotion_f,
                config=config,
                ticks=ticks_per_point,
                burn_in_ticks=burn_in_ticks,
                seed=seed,
                start_present_both=start_present_both,
            )
            all_profits.extend(profits)
            all_shares.extend(shares)

        mean_profit, std_error, n = _mean_and_std_error(all_profits)
        rows.append(
            SweepRow1DPromotion(
                promotion=promotion_f,
                fixed_price=fixed_price,
                fixed_spend=fixed_spend,
                fixed_assortment=fixed_assortment,
                mean_profit=mean_profit,
                std_error=std_error,
                mean_market_share=float(np.asarray(all_shares, dtype=np.float64).mean()),
                n_samples=n,
            )
        )
    return rows


def _expansion_action(
    price: float,
    spend: float,
    assortment: float,
    promotion: float,
    *,
    open_region_1: bool,
) -> npt.NDArray[np.float32]:
    """Build the flat action holding the pinned optima, opening region 1 iff ``open_region_1``."""
    open_logit = 1.0 if open_region_1 else 0.0
    return np.array(
        [
            price,
            spend,
            assortment,
            promotion,
            0.0,
            open_logit,
            *([0.0] * 30),
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _expansion_timing_window_total(
    *,
    open_tick: int,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    fixed_promotion: float,
    config: CoreConfig,
    window_ticks: int,
    seed: int,
) -> float:
    """Total agent profit over the scoring window for ONE expansion-timing policy.

    Runs a full trajectory on the World seam: the agent holds the continuous optimum every
    tick and emits the open-region-1 action ONLY at ``open_tick``; the seam's cash/presence
    gate decides whether the open fires. The score is the SUM of per-tick profit over the window.
    """
    world = World(config=config, seed=seed)
    state = world.reset(seed)
    total_profit = 0.0
    for tick in range(window_ticks):
        action = _expansion_action(
            fixed_price,
            fixed_spend,
            fixed_assortment,
            fixed_promotion,
            open_region_1=(tick == open_tick),
        )
        result = world.step(state, {AGENT_INDEX: action})
        state = result.next_state
        total_profit += float(state.retailers[AGENT_INDEX].last_profit)
    return total_profit


def _sweep_expansion_timing(
    config: CoreConfig,
    *,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    fixed_promotion: float,
    timings: tuple[int, ...],
    window_ticks: int,
    seeds: tuple[int, ...],
) -> list[ExpansionTimingRow]:
    """Sweep one open tick per policy (never-expand + each expand-at-T) over ``seeds``."""
    rows: list[ExpansionTimingRow] = []
    for open_tick in (NEVER_EXPAND_T, *timings):
        per_seed_totals = [
            _expansion_timing_window_total(
                open_tick=open_tick,
                fixed_price=fixed_price,
                fixed_spend=fixed_spend,
                fixed_assortment=fixed_assortment,
                fixed_promotion=fixed_promotion,
                config=config,
                window_ticks=window_ticks,
                seed=seed,
            )
            for seed in seeds
        ]
        mean_profit, std_error, n = _mean_and_std_error(per_seed_totals)
        rows.append(
            ExpansionTimingRow(
                open_tick=open_tick,
                mean_profit=mean_profit,
                std_error=std_error,
                n_seeds=n,
            )
        )
    return rows


def _competitor_price_obs_index(config: CoreConfig) -> int:
    """The flat-obs index of the perceived ``competitor_price`` field, read from OBSERVATION_SCHEMA."""
    from retail_simulator.core.schema import OBSERVATION_SCHEMA

    for idx, field in enumerate(OBSERVATION_SCHEMA):
        if field.name == _COMPETITOR_PRICE_OBS_NAME:
            return idx
    raise ValueError(f"no {_COMPETITOR_PRICE_OBS_NAME!r} field in OBSERVATION_SCHEMA")


def research_gate_segments() -> dict[str, SegmentParams]:
    """The demand segments the research leg runs on."""
    segments = default_segments()
    segments[PRICE_SENSITIVE] = replace(
        segments[PRICE_SENSITIVE], beta_reference=RESEARCH_GATE_BETA_REFERENCE
    )
    return segments


def research_opponent_config(
    base_config: CoreConfig,
    *,
    opponent_archetype: str = RESEARCH_OPPONENT_ARCHETYPE,
) -> CoreConfig:
    """A config whose opponent NPC is the reactive research heuristic (the axis's varying opponent)."""
    region_cfgs = base_config.demand.regions
    n_regions = len(region_cfgs) if region_cfgs else 1
    seat_plan = (
        SeatSpec(is_npc=False, name="agent", presence=(1,) * n_regions),
        SeatSpec(is_npc=True, archetype=opponent_archetype, name=opponent_archetype),
    )
    return replace(base_config, seats=seat_plan, segments=research_gate_segments())


def _research_samples_for_cell(
    research_spend: float,
    *,
    config: CoreConfig,
    competitor_price_idx: int,
    ticks: int,
    burn_in_ticks: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    """Per-tick agent-score samples for one research-spend cell over ``seeds``."""
    world = World(config=config, seed=seed)
    state = world.reset(seed)
    price_low = config.pricing.min_price_index
    price_high = config.pricing.max_price_index
    reset_obs, _ = world.agent_observation(state, AGENT_INDEX)
    perceived_competitor_price = float(reset_obs[competitor_price_idx])

    profits: list[float] = []
    market_shares: list[float] = []
    for tick in range(ticks):
        if perceived_competitor_price > price_low:
            target = perceived_competitor_price * 0.97
        else:
            target = 0.5 * (price_low + price_high)
        price = float(min(max(target, price_low), price_high))
        action = np.array(
            [price, 0.4, 0.5, 0.0, 0.0, 0.0]
            + [0.0] * 30
            + [research_spend, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )
        result = world.step(state, {AGENT_INDEX: action})
        state = result.next_state
        perceived_competitor_price = float(result.observations[AGENT_INDEX][competitor_price_idx])
        if tick < burn_in_ticks:
            continue
        agent = state.retailers[AGENT_INDEX]
        profits.append(float(agent.last_profit))
        market_shares.append(float(agent.last_market_share))
    return profits, market_shares


def _sweep_research_axis(
    config: CoreConfig,
    *,
    research_spends: tuple[float, ...],
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
) -> list[ResearchFidelityRow]:
    """Sweep the research-spend grid against the reactive opponent over ``seeds``."""
    opponent_config = research_opponent_config(config)
    competitor_price_idx = _competitor_price_obs_index(opponent_config)

    rows: list[ResearchFidelityRow] = []
    for research_spend in research_spends:
        spend_f = float(research_spend)
        fidelity = opponent_config.research.fidelity(spend_f)
        all_profits: list[float] = []
        all_shares: list[float] = []
        for seed in seeds:
            profits, shares = _research_samples_for_cell(
                spend_f,
                config=opponent_config,
                competitor_price_idx=competitor_price_idx,
                ticks=ticks_per_point,
                burn_in_ticks=burn_in_ticks,
                seed=seed,
            )
            all_profits.extend(profits)
            all_shares.extend(shares)

        mean_score, std_error, n = _mean_and_std_error(all_profits)
        rows.append(
            ResearchFidelityRow(
                research_spend=spend_f,
                fidelity=float(fidelity),
                mean_score=mean_score,
                std_error=std_error,
                mean_market_share=float(np.asarray(all_shares, dtype=np.float64).mean()),
                n_samples=n,
            )
        )
    return rows
