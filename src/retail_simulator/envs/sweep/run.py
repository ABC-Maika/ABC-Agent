"""Run every configured sweep leg and assemble the SweepResult."""

from __future__ import annotations

from retail_simulator.core.config import CoreConfig
from retail_simulator.envs.sweep.axes import (
    _default_burn_in_ticks,
    _sweep_2d,
    _sweep_assortment_axis,
    _sweep_expansion_timing,
    _sweep_promotion_axis,
    _sweep_research_axis,
)
from retail_simulator.envs.sweep.axes_extended import (
    _per_region_optimal_score_for_scm,
    _sweep_automation_axis,
    _sweep_scm_service_axis,
)
from retail_simulator.envs.sweep.axes_loyalty import (
    _sweep_loyalty_axis,
    _sweep_loyalty_orthogonality_probe,
)
from retail_simulator.envs.sweep.rows import (
    DEFAULT_AUTOMATION_TIMINGS,
    DEFAULT_EXPANSION_TIMINGS,
    DEFAULT_GRID_POINTS,
    DEFAULT_LOYALTY_SPENDS,
    DEFAULT_RESEARCH_SPENDS,
    DEFAULT_SEEDS,
    DEFAULT_SERVICE_LEVELS,
    DEFAULT_TICKS_PER_POINT,
    MARGIN_SIGMA_MULTIPLE,
    PINNED_ASSORTMENT,
    PINNED_PROMOTION,
    SCORING_WINDOW_TICKS,
    ExpansionTimingRow,
    LoyaltyVsPromotionOrthogonality,
    ResearchFidelityRow,
    SweepResult,
    SweepRow1DAutomation,
    SweepRow1DLoyalty,
    SweepRow1DService,
)
from retail_simulator.envs.sweep.verdicts import (
    assortment_verdict_for_axis,
    expansion_verdict_for_timing,
    promotion_verdict_for_axis,
    research_verdict_for_fidelity,
    verdict_for_sweep,
)
from retail_simulator.envs.sweep.verdicts_extended import (
    automation_verdict_for_timing,
    loyalty_verdict_for_spend,
    loyalty_vs_promotion_orthogonality_verdict,
    scm_verdict_for_service,
)

DEFAULT_LOYALTY_ORTHO_DELTAS: tuple[float, ...] = (0.1, 0.3)


def run_sweep(
    config: CoreConfig | None = None,
    *,
    grid_points: int = DEFAULT_GRID_POINTS,
    ticks_per_point: int = DEFAULT_TICKS_PER_POINT,
    burn_in_ticks: int | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    pinned_assortment: float = PINNED_ASSORTMENT,
    pinned_promotion: float = PINNED_PROMOTION,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
    start_present_both: bool = False,
    run_expansion_timing: bool = True,
    expansion_timings: tuple[int, ...] = DEFAULT_EXPANSION_TIMINGS,
    expansion_window_ticks: int = SCORING_WINDOW_TICKS,
    run_research: bool = True,
    research_spends: tuple[float, ...] = DEFAULT_RESEARCH_SPENDS,
    run_scm: bool = True,
    scm_service_levels: tuple[float, ...] = DEFAULT_SERVICE_LEVELS,
    run_automation: bool = True,
    automation_timings: tuple[int, ...] = DEFAULT_AUTOMATION_TIMINGS,
    automation_window_ticks: int = SCORING_WINDOW_TICKS,
    run_loyalty: bool = True,
    loyalty_spends: tuple[float, ...] = DEFAULT_LOYALTY_SPENDS,
    loyalty_ortho_deltas: tuple[float, ...] = DEFAULT_LOYALTY_ORTHO_DELTAS,
) -> SweepResult:
    """Run every configured sweep leg and assemble the SweepResult."""
    if config is None:
        config = CoreConfig.default()
    if grid_points < 2:
        raise ValueError("grid_points must be >= 2 (need both range endpoints)")
    if burn_in_ticks is None:
        burn_in_ticks = _default_burn_in_ticks(config)
    if burn_in_ticks < 0:
        raise ValueError("burn_in_ticks must be >= 0")
    if ticks_per_point <= burn_in_ticks:
        raise ValueError(
            f"ticks_per_point ({ticks_per_point}) must exceed burn_in_ticks "
            f"({burn_in_ticks}) so a post-burn-in sample remains"
        )
    if not seeds:
        raise ValueError("at least one seed is required")

    price_low = config.pricing.min_price_index
    price_high = config.pricing.max_price_index

    # --- Pass 1: the 2-D price x marketing surface at pinned assortment + promo. ---
    rows = _sweep_2d(
        config,
        grid_points=grid_points,
        ticks_per_point=ticks_per_point,
        burn_in_ticks=burn_in_ticks,
        seeds=seeds,
        pinned_assortment=pinned_assortment,
        pinned_promotion=pinned_promotion,
        start_present_both=start_present_both,
    )
    row_tuple = tuple(rows)
    is_degenerate_2d, reason_2d, profit_max_cell, profit_max_value = verdict_for_sweep(
        row_tuple,
        price_low=price_low,
        price_high=price_high,
        margin_sigma_multiple=margin_sigma_multiple,
    )

    # --- Pass 2: the 1-D assortment axis at the joint (price*, marketing*). ---
    best_price, best_spend = profit_max_cell
    assortment_rows = _sweep_assortment_axis(
        config,
        fixed_price=best_price,
        fixed_spend=best_spend,
        pinned_promotion=pinned_promotion,
        grid_points=grid_points,
        ticks_per_point=ticks_per_point,
        burn_in_ticks=burn_in_ticks,
        seeds=seeds,
        start_present_both=start_present_both,
    )
    assortment_tuple = tuple(assortment_rows)
    (
        is_degenerate_assort,
        reason_assort,
        best_assortment,
        assortment_max_value,
    ) = assortment_verdict_for_axis(
        assortment_tuple,
        margin_sigma_multiple=margin_sigma_multiple,
    )

    # --- Pass 3: the 1-D promotion axis at (price*, marketing*, assortment*). ---
    promotion_rows = _sweep_promotion_axis(
        config,
        fixed_price=best_price,
        fixed_spend=best_spend,
        fixed_assortment=best_assortment,
        grid_points=grid_points,
        ticks_per_point=ticks_per_point,
        burn_in_ticks=burn_in_ticks,
        seeds=seeds,
        start_present_both=start_present_both,
    )
    promotion_tuple = tuple(promotion_rows)
    (
        is_degenerate_promo,
        reason_promo,
        best_promotion,
        promotion_max_value,
    ) = promotion_verdict_for_axis(
        promotion_tuple,
        margin_sigma_multiple=margin_sigma_multiple,
    )

    continuous_degenerate = is_degenerate_2d or is_degenerate_assort or is_degenerate_promo

    region_cfgs = config.demand.regions
    n_regions = len(region_cfgs) if region_cfgs else 1
    expansion_timing_tuple: tuple[ExpansionTimingRow, ...] = ()
    expansion_timing_degenerate: bool | None = None
    expansion_timing_reason = ""
    if run_expansion_timing and n_regions >= 2:
        expansion_timing_rows = _sweep_expansion_timing(
            config,
            fixed_price=best_price,
            fixed_spend=best_spend,
            fixed_assortment=best_assortment,
            fixed_promotion=best_promotion,
            timings=expansion_timings,
            window_ticks=expansion_window_ticks,
            seeds=seeds,
        )
        expansion_timing_tuple = tuple(expansion_timing_rows)
        (
            expansion_timing_degenerate,
            expansion_timing_reason,
            _best_open_tick,
            _best_timing_profit,
        ) = expansion_verdict_for_timing(
            expansion_timing_tuple,
            margin_sigma_multiple=margin_sigma_multiple,
        )

    research_tuple: tuple[ResearchFidelityRow, ...] = ()
    research_degenerate: bool | None = None
    research_reason = ""
    if run_research:
        research_rows = _sweep_research_axis(
            config,
            research_spends=research_spends,
            ticks_per_point=ticks_per_point,
            burn_in_ticks=burn_in_ticks,
            seeds=seeds,
        )
        research_tuple = tuple(research_rows)
        (
            research_degenerate,
            research_reason,
            _best_research_spend,
            _best_research_score,
        ) = research_verdict_for_fidelity(
            research_tuple,
            margin_sigma_multiple=margin_sigma_multiple,
        )

    service_tuple: tuple[SweepRow1DService, ...] = ()
    service_variable_mean: float | None = None
    service_variable_std_error: float | None = None
    service_degenerate: bool | None = None
    service_reason = ""
    if run_scm and n_regions >= 2:
        scm_rows = _sweep_scm_service_axis(
            config,
            fixed_price=best_price,
            fixed_spend=best_spend,
            fixed_assortment=best_assortment,
            fixed_promotion=best_promotion,
            service_levels=scm_service_levels,
            ticks_per_point=ticks_per_point,
            burn_in_ticks=burn_in_ticks,
            seeds=seeds,
            start_present_both=start_present_both,
        )
        service_tuple = tuple(scm_rows)
        service_variable_mean, service_variable_std_error = _per_region_optimal_score_for_scm(
            config,
            fixed_price=best_price,
            fixed_spend=best_spend,
            fixed_assortment=best_assortment,
            fixed_promotion=best_promotion,
            service_levels=scm_service_levels,
            ticks_per_point=ticks_per_point,
            burn_in_ticks=burn_in_ticks,
            seeds=seeds,
        )
        (
            service_degenerate,
            service_reason,
            _best_constant_service,
            _best_constant_profit,
        ) = scm_verdict_for_service(
            service_tuple,
            service_variable_mean,
            service_variable_std_error,
            margin_sigma_multiple=margin_sigma_multiple,
        )

    automation_tuple: tuple[SweepRow1DAutomation, ...] = ()
    automation_degenerate: bool | None = None
    automation_reason = ""
    if run_automation:
        automation_rows = _sweep_automation_axis(
            config,
            fixed_price=best_price,
            fixed_spend=best_spend,
            fixed_assortment=best_assortment,
            fixed_promotion=best_promotion,
            timings=automation_timings,
            window_ticks=automation_window_ticks,
            seeds=seeds,
            start_present_both=start_present_both,
        )
        automation_tuple = tuple(automation_rows)
        (
            automation_degenerate,
            automation_reason,
            _best_policy_name,
            _best_profit,
        ) = automation_verdict_for_timing(
            automation_tuple,
            margin_sigma_multiple=margin_sigma_multiple,
        )

    loyalty_tuple: tuple[SweepRow1DLoyalty, ...] = ()
    loyalty_degenerate: bool | None = None
    loyalty_reason = ""
    loyalty_orthogonality: LoyaltyVsPromotionOrthogonality | None = None
    if run_loyalty:
        loyalty_rows = _sweep_loyalty_axis(
            config,
            fixed_price=best_price,
            fixed_spend=best_spend,
            fixed_assortment=best_assortment,
            fixed_promotion=best_promotion,
            loyalty_spends=loyalty_spends,
            ticks_per_point=ticks_per_point,
            burn_in_ticks=burn_in_ticks,
            seeds=seeds,
            start_present_both=start_present_both,
        )
        loyalty_tuple = tuple(loyalty_rows)
        (
            loyalty_degenerate,
            loyalty_reason,
            best_loyalty_spend,
            _best_loyalty_profit,
        ) = loyalty_verdict_for_spend(
            loyalty_tuple,
            margin_sigma_multiple=margin_sigma_multiple,
        )

        base_mean, base_se, swap_cells = _sweep_loyalty_orthogonality_probe(
            config,
            base_loyalty=best_loyalty_spend,
            base_promotion=best_promotion,
            fixed_price=best_price,
            fixed_spend=best_spend,
            fixed_assortment=best_assortment,
            swap_deltas=loyalty_ortho_deltas,
            ticks_per_point=ticks_per_point,
            burn_in_ticks=burn_in_ticks,
            seeds=seeds,
            start_present_both=start_present_both,
        )
        loyalty_orthogonality = loyalty_vs_promotion_orthogonality_verdict(
            base_loyalty=best_loyalty_spend,
            base_promotion=best_promotion,
            base_profit=base_mean,
            base_std_error=base_se,
            swap_profits=swap_cells,
            margin_sigma_multiple=margin_sigma_multiple,
        )

    timing_fails = bool(expansion_timing_degenerate)
    research_fails = bool(research_degenerate)
    scm_fails = bool(service_degenerate)
    automation_fails = bool(automation_degenerate)
    loyalty_fails = bool(loyalty_degenerate) or (
        loyalty_orthogonality is not None and not loyalty_orthogonality.passed
    )
    is_degenerate = (
        continuous_degenerate
        or timing_fails
        or research_fails
        or scm_fails
        or automation_fails
        or loyalty_fails
    )

    return SweepResult(
        rows=row_tuple,
        assortment_rows=assortment_tuple,
        promotion_rows=promotion_tuple,
        profit_max_cell=profit_max_cell,
        profit_max_value=profit_max_value,
        assortment_max_value=assortment_max_value,
        assortment_max_cell=best_assortment,
        promotion_max_value=promotion_max_value,
        is_degenerate=is_degenerate,
        reason=reason_2d,
        assortment_reason=reason_assort,
        promotion_reason=reason_promo,
        pinned_assortment=pinned_assortment,
        pinned_promotion=pinned_promotion,
        ticks_per_point=ticks_per_point,
        burn_in_ticks=burn_in_ticks,
        seeds=tuple(seeds),
        expansion_timing_rows=expansion_timing_tuple,
        expansion_timing_degenerate=expansion_timing_degenerate,
        expansion_timing_reason=expansion_timing_reason,
        start_present_both=start_present_both,
        research_rows=research_tuple,
        research_degenerate=research_degenerate,
        research_reason=research_reason,
        service_rows=service_tuple,
        service_variable_mean=service_variable_mean,
        service_variable_std_error=service_variable_std_error,
        service_degenerate=service_degenerate,
        service_reason=service_reason,
        automation_rows=automation_tuple,
        automation_degenerate=automation_degenerate,
        automation_reason=automation_reason,
        loyalty_rows=loyalty_tuple,
        loyalty_degenerate=loyalty_degenerate,
        loyalty_reason=loyalty_reason,
        loyalty_orthogonality=loyalty_orthogonality,
    )
