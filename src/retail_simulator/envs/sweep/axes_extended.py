"""Extended per-axis sweep implementations: SCM and automation."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import (
    AutomationConfig,
    CoreConfig,
    SCMConfig,
)
from retail_simulator.core.state import (
    CONVENIENCE,
    SegmentParams,
    WorldState,
    default_segments,
)
from retail_simulator.core.world import AGENT_INDEX, World
from retail_simulator.envs.sweep.axes import _seat_present_both
from retail_simulator.envs.sweep.rows import (
    AUTOMATION_GATE_CAPEX_PER_TIER,
    AUTOMATION_GATE_SAVINGS_PER_TIER,
    NEVER_UPGRADE_T,
    RAMP_POLICY_T,
    SCM_GATE_COGS_PREMIUM,
    SCM_GATE_CONVENIENCE_BETA_SERVICE,
    SCM_GATE_LOST_SALES_SHARE_PENALTY,
    SweepRow1DAutomation,
    SweepRow1DService,
    _mean_and_std_error,
)


def scm_gate_segments(
    base_segments: dict[str, SegmentParams] | None = None,
) -> dict[str, SegmentParams]:
    """The demand segments the SCM leg runs on."""
    segments = (
        {key: value for key, value in base_segments.items()}
        if base_segments is not None
        else default_segments()
    )
    if CONVENIENCE not in segments:
        raise ValueError(
            "scm_gate_segments requires the CONVENIENCE segment in base_segments "
            "(the SCM lever concentrates beta_service in CONVENIENCE); got keys "
            + repr(sorted(segments))
        )
    segments[CONVENIENCE] = replace(
        segments[CONVENIENCE], beta_service=SCM_GATE_CONVENIENCE_BETA_SERVICE
    )
    return segments


def scm_gate_config(
    base_config: CoreConfig | None = None,
    *,
    base_segments: dict[str, SegmentParams] | None = None,
) -> CoreConfig:
    """A config with the SCM (value-of-service) scenario enabled."""
    base = base_config if base_config is not None else CoreConfig.default()
    return replace(
        base,
        scm=SCMConfig(
            cogs_premium=SCM_GATE_COGS_PREMIUM,
            fill_rate_floor=base.scm.fill_rate_floor,
            fill_rate_curvature=base.scm.fill_rate_curvature,
            lost_sales_share_penalty=SCM_GATE_LOST_SALES_SHARE_PENALTY,
            service_score_decay=base.scm.service_score_decay,
        ),
        segments=scm_gate_segments(base_segments=base_segments),
    )


def _scm_action(
    service_level: float,
    *,
    price: float,
    spend: float,
    assortment: float,
    promotion: float,
) -> npt.NDArray[np.float32]:
    """Build the flat action holding the pinned optima with ``service_level`` set."""
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
            service_level,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _scm_samples_for_cell(
    service_level: float,
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
    """Per-tick agent-profit samples for one service_level cell over ``seeds``."""
    world = World(config=config, seed=seed)
    state = world.reset(seed)
    if start_present_both:
        _seat_present_both(state)
    fixed_action = _scm_action(
        service_level, price=price, spend=spend, assortment=assortment, promotion=promotion
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


def _sweep_scm_service_axis(
    config: CoreConfig,
    *,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    fixed_promotion: float,
    service_levels: tuple[float, ...],
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
    start_present_both: bool,
) -> list[SweepRow1DService]:
    """Sweep the service-level grid over ``seeds``."""
    rows: list[SweepRow1DService] = []
    for service_level in service_levels:
        s_f = float(service_level)
        all_profits: list[float] = []
        all_shares: list[float] = []
        for seed in seeds:
            profits, shares = _scm_samples_for_cell(
                s_f,
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
            SweepRow1DService(
                service_level=s_f,
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


def _per_region_optimal_score_for_scm(
    config: CoreConfig,
    *,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    fixed_promotion: float,
    service_levels: tuple[float, ...],
    ticks_per_point: int,
    burn_in_ticks: int,
    seeds: tuple[int, ...],
) -> tuple[float, float]:
    """The per-region-optimal oracle: the sum of each region's best constant-service score."""
    region_cfgs = config.demand.regions
    n_regions = len(region_cfgs) if region_cfgs else 1

    total_mean = 0.0
    summed_variance = 0.0
    for r in range(n_regions):
        per_region_rows: list[SweepRow1DService] = []
        for service_level in service_levels:
            s_f = float(service_level)
            all_profits: list[float] = []
            for seed in seeds:
                profits, _shares = _scm_samples_for_cell_region(
                    s_f,
                    region_index=r,
                    config=config,
                    price=fixed_price,
                    spend=fixed_spend,
                    assortment=fixed_assortment,
                    promotion=fixed_promotion,
                    ticks=ticks_per_point,
                    burn_in_ticks=burn_in_ticks,
                    seed=seed,
                )
                all_profits.extend(profits)

            mean_profit, std_error, n = _mean_and_std_error(all_profits)
            per_region_rows.append(
                SweepRow1DService(
                    service_level=s_f,
                    fixed_price=fixed_price,
                    fixed_spend=fixed_spend,
                    fixed_assortment=fixed_assortment,
                    fixed_promotion=fixed_promotion,
                    mean_profit=mean_profit,
                    std_error=std_error,
                    mean_market_share=0.0,
                    n_samples=n,
                )
            )

        per_region_best = max(per_region_rows, key=lambda r_: r_.mean_profit)
        total_mean += per_region_best.mean_profit
        summed_variance += per_region_best.std_error**2

    variable_std_error = math.sqrt(summed_variance)
    return total_mean, variable_std_error


def _seat_present_in_region_only(state: WorldState, region_index: int) -> None:
    """Seat the agent PRESENT IN ONLY ``region_index`` at reset.

    Flips the agent's ``stores_per_region`` to a one-hot mask at ``region_index``.
    """
    agent = state.retailers[AGENT_INDEX]
    mature_age = agent.store_age_per_region[0] if agent.store_age_per_region else 0
    agent.stores_per_region = tuple(
        1 if r == region_index else 0 for r in range(len(agent.stores_per_region))
    )
    agent.store_age_per_region = tuple(
        mature_age if r == region_index else 0 for r in range(len(agent.store_age_per_region))
    )


def _scm_samples_for_cell_region(
    service_level: float,
    *,
    region_index: int,
    config: CoreConfig,
    price: float,
    spend: float,
    assortment: float,
    promotion: float,
    ticks: int,
    burn_in_ticks: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    """Single-region variant of :func:`_scm_samples_for_cell`: the agent is seated present in only ``region_index``."""
    world = World(config=config, seed=seed)
    state = world.reset(seed)
    _seat_present_in_region_only(state, region_index)
    fixed_action = _scm_action(
        service_level, price=price, spend=spend, assortment=assortment, promotion=promotion
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


def automation_gate_segments() -> dict[str, SegmentParams]:
    """The demand segments the automation leg runs on."""
    return default_segments()


def automation_gate_config(base_config: CoreConfig | None = None) -> CoreConfig:
    """A config with the automation scenario enabled."""
    base = base_config if base_config is not None else CoreConfig.default()
    return replace(
        base,
        automation=AutomationConfig(
            capex_per_tier=AUTOMATION_GATE_CAPEX_PER_TIER,
            savings_per_tier=AUTOMATION_GATE_SAVINGS_PER_TIER,
        ),
    )


def _automation_action(
    automation_tier: int,
    *,
    price: float,
    spend: float,
    assortment: float,
    promotion: float,
) -> npt.NDArray[np.float32]:
    """Build the flat action holding the pinned optima with the automation target tier set."""
    auto_logits = [0.0, 0.0, 0.0]
    auto_logits[automation_tier] = 1.0
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
            *auto_logits,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _automation_policy_name(*, target_tier: int, upgrade_tick: int, ramp_tick: int = 0) -> str:
    """Human-readable label for an automation upgrade-timing policy (CSV / format)."""
    if upgrade_tick == NEVER_UPGRADE_T:
        return "never_upgrade"
    if upgrade_tick == RAMP_POLICY_T:
        return f"ramp_0_to_1_to_2_at_t={ramp_tick}"
    return f"upgrade_to_{target_tier}_at_t={upgrade_tick}"


def _automation_target_for_tick(
    tick: int, *, target_tier: int, upgrade_tick: int, ramp_tick: int
) -> int:
    """The TARGET tier the agent picks at ``tick`` under the given upgrade-timing policy."""
    if upgrade_tick == NEVER_UPGRADE_T:
        return 0
    if upgrade_tick == RAMP_POLICY_T:
        return 1 if tick < ramp_tick else 2
    return target_tier if tick >= upgrade_tick else 0


def _automation_timing_window_total(
    *,
    target_tier: int,
    upgrade_tick: int,
    ramp_tick: int,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    fixed_promotion: float,
    config: CoreConfig,
    window_ticks: int,
    seed: int,
    start_present_both: bool,
) -> float:
    """Total agent profit over the scoring window for ONE automation upgrade-timing policy.

    Runs a full trajectory on the World seam: the agent holds the continuous optimum every
    tick and emits the TARGET automation tier; the seam's upgrade gate decides whether each
    tick's target fires. The score is the SUM of per-tick profit over the window.
    """
    world = World(config=config, seed=seed)
    state = world.reset(seed)
    if start_present_both:
        _seat_present_both(state)
    total_profit = 0.0
    for tick in range(window_ticks):
        tier = _automation_target_for_tick(
            tick,
            target_tier=target_tier,
            upgrade_tick=upgrade_tick,
            ramp_tick=ramp_tick,
        )
        action = _automation_action(
            tier,
            price=fixed_price,
            spend=fixed_spend,
            assortment=fixed_assortment,
            promotion=fixed_promotion,
        )
        result = world.step(state, {AGENT_INDEX: action})
        state = result.next_state
        total_profit += float(state.retailers[AGENT_INDEX].last_profit)
    return total_profit


def _sweep_automation_axis(
    config: CoreConfig,
    *,
    fixed_price: float,
    fixed_spend: float,
    fixed_assortment: float,
    fixed_promotion: float,
    timings: tuple[int, ...],
    window_ticks: int,
    seeds: tuple[int, ...],
    start_present_both: bool,
) -> list[SweepRow1DAutomation]:
    """Sweep the upgrade-timing policies (never, per-T, ramp) over ``seeds``."""
    interior_timings = [t for t in timings if t > 0]
    ramp_tick = interior_timings[0] if interior_timings else 5

    policy_specs: list[tuple[int, int, int]] = [(0, NEVER_UPGRADE_T, 0)]
    policy_specs.append((1, 0, 0))
    for t in timings:
        policy_specs.append((2, int(t), 0))
    policy_specs.append((2, RAMP_POLICY_T, ramp_tick))

    rows: list[SweepRow1DAutomation] = []
    for target_tier, upgrade_tick, ramp_t in policy_specs:
        per_seed_totals = [
            _automation_timing_window_total(
                target_tier=target_tier,
                upgrade_tick=upgrade_tick,
                ramp_tick=ramp_t,
                fixed_price=fixed_price,
                fixed_spend=fixed_spend,
                fixed_assortment=fixed_assortment,
                fixed_promotion=fixed_promotion,
                config=config,
                window_ticks=window_ticks,
                seed=seed,
                start_present_both=start_present_both,
            )
            for seed in seeds
        ]
        mean_profit, std_error, n = _mean_and_std_error(per_seed_totals)
        rows.append(
            SweepRow1DAutomation(
                policy_name=_automation_policy_name(
                    target_tier=target_tier,
                    upgrade_tick=upgrade_tick,
                    ramp_tick=ramp_t,
                ),
                target_tier=target_tier,
                upgrade_tick=upgrade_tick,
                ramp_tick=ramp_t,
                fixed_price=fixed_price,
                fixed_spend=fixed_spend,
                fixed_assortment=fixed_assortment,
                fixed_promotion=fixed_promotion,
                mean_profit=mean_profit,
                std_error=std_error,
                n_seeds=n,
            )
        )
    return rows

