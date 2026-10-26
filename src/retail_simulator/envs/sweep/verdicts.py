"""Core sweep verdicts: 2-D grid, assortment, promotion, expansion-timing, and research."""

from __future__ import annotations

import math

from retail_simulator.envs.sweep.rows import (
    _BOUNDARY_ATOL,
    ASSORT_HIGH,
    ASSORT_LOW,
    MARGIN_SIGMA_MULTIPLE,
    NEVER_EXPAND_T,
    PROMO_HIGH,
    PROMO_LOW,
    RESEARCH_HIGH,
    RESEARCH_LOW,
    SPEND_HIGH,
    SPEND_LOW,
    TIMING_FLAT_SIGMA_MULTIPLE,
    ExpansionTimingRow,
    ResearchFidelityRow,
    SweepRow1D,
    SweepRow1DPromotion,
    SweepRow2D,
    _beats,
    _beats_profit,
    _is_boundary,
)


def verdict_for_sweep(
    rows: tuple[SweepRow2D, ...],
    *,
    price_low: float,
    price_high: float,
    spend_low: float = SPEND_LOW,
    spend_high: float = SPEND_HIGH,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
) -> tuple[bool, str, tuple[float, float], float]:
    """The 2-D degeneracy verdict: (not_degenerate, reason, best_cell, best_value) over the price x marketing grid."""
    if not rows:
        raise ValueError("cannot judge an empty sweep")

    best = max(rows, key=lambda r: r.mean_profit)
    profit_max_cell = (best.price_index, best.spend_fraction)
    profit_max_value = best.mean_profit

    # --- Price leg: compare along the price axis AT the best spend. ---
    same_spend = [
        r
        for r in rows
        if math.isclose(r.spend_fraction, best.spend_fraction, abs_tol=_BOUNDARY_ATOL)
    ]
    price_boundaries = [r for r in same_spend if _is_boundary(r.price_index, price_low, price_high)]
    if not price_boundaries:
        raise ValueError(
            "sweep grid has no price-boundary cells at the best spend; "
            "include the pricing-range endpoints"
        )

    if _is_boundary(best.price_index, price_low, price_high):
        return (
            True,
            (
                f"price-boundary: profit-max price {best.price_index:.4f} is at a boundary "
                f"([{price_low:.2f}, {price_high:.2f}]) at spend {best.spend_fraction:.3f} "
                f"-> dominant boundary pricing strategy"
            ),
            profit_max_cell,
            profit_max_value,
        )
    for boundary in price_boundaries:
        if not _beats(best, boundary, margin_sigma_multiple):
            combined_se = math.hypot(best.std_error, boundary.std_error)
            return (
                True,
                (
                    f"price-boundary: interior price {best.price_index:.4f} "
                    f"(profit {best.mean_profit:.2f}) does not beat price boundary "
                    f"{boundary.price_index:.2f} (profit {boundary.mean_profit:.2f}) at "
                    f"spend {best.spend_fraction:.3f} by >{margin_sigma_multiple:g}x std-error: "
                    f"margin {best.mean_profit - boundary.mean_profit:.2f} "
                    f"<= threshold {margin_sigma_multiple * combined_se:.2f}"
                ),
                profit_max_cell,
                profit_max_value,
            )

    # --- Marketing leg: the best spend must be non-trivial AND pay off vs 0. ---
    if math.isclose(best.spend_fraction, spend_low, abs_tol=_BOUNDARY_ATOL):
        # best spend == spend_low (== 0): marketing is inert.
        return (
            True,
            (
                f"marketing-inert: profit-max spend {best.spend_fraction:.3f} is the floor "
                f"({spend_low:.2f}) -> marketing never worth using (lever is inert)"
            ),
            profit_max_cell,
            profit_max_value,
        )
    if math.isclose(best.spend_fraction, spend_high, abs_tol=_BOUNDARY_ATOL):
        # best spend pinned at max: trivial "spend everything".
        return (
            True,
            (
                f"marketing-saturated: profit-max spend {best.spend_fraction:.3f} is pinned at "
                f"the max ({spend_high:.2f}) -> trivial spend-everything strategy"
            ),
            profit_max_cell,
            profit_max_value,
        )

    # ROI: the best spend must beat spend=0 at the SAME price by margin.
    same_price_zero_spend = [
        r
        for r in rows
        if math.isclose(r.price_index, best.price_index, abs_tol=_BOUNDARY_ATOL)
        and math.isclose(r.spend_fraction, spend_low, abs_tol=_BOUNDARY_ATOL)
    ]
    if not same_price_zero_spend:
        raise ValueError(
            "sweep grid has no spend=0 cell at the best price; include the spend floor"
        )
    zero_spend = same_price_zero_spend[0]
    if not _beats(best, zero_spend, margin_sigma_multiple):
        combined_se = math.hypot(best.std_error, zero_spend.std_error)
        return (
            True,
            (
                f"marketing-no-payoff: best spend {best.spend_fraction:.3f} at price "
                f"{best.price_index:.4f} (profit {best.mean_profit:.2f}) does not beat spend=0 "
                f"(profit {zero_spend.mean_profit:.2f}) by >{margin_sigma_multiple:g}x std-error: "
                f"margin {best.mean_profit - zero_spend.mean_profit:.2f} "
                f"<= threshold {margin_sigma_multiple * combined_se:.2f}"
            ),
            profit_max_cell,
            profit_max_value,
        )

    return (
        False,
        (
            f"clear interior optimum at price {best.price_index:.4f}, spend "
            f"{best.spend_fraction:.3f} (profit {best.mean_profit:.2f}): beats both price "
            f"boundaries and spend=0 by >{margin_sigma_multiple:g}x std-error (both levers pay)"
        ),
        profit_max_cell,
        profit_max_value,
    )


def assortment_verdict_for_axis(
    rows: tuple[SweepRow1D, ...],
    *,
    assort_low: float = ASSORT_LOW,
    assort_high: float = ASSORT_HIGH,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
) -> tuple[bool, str, float, float]:
    """The assortment-axis degeneracy verdict: (not_degenerate, reason, best_assortment, best_value)."""
    if not rows:
        raise ValueError("cannot judge an empty assortment axis")

    best = max(rows, key=lambda r: r.mean_profit)
    best_assortment = best.assortment
    best_profit = best.mean_profit

    zero_cells = [r for r in rows if math.isclose(r.assortment, assort_low, abs_tol=_BOUNDARY_ATOL)]
    if not zero_cells:
        raise ValueError("assortment axis has no assortment=0 cell; include the assortment floor")
    zero_assort = zero_cells[0]

    if math.isclose(best.assortment, assort_low, abs_tol=_BOUNDARY_ATOL):
        # best assortment == floor (== 0): breadth never helps.
        return (
            True,
            (
                f"assortment-inert: profit-max assortment {best.assortment:.3f} is the floor "
                f"({assort_low:.2f}) -> breadth never worth setting (lever is inert)"
            ),
            best_assortment,
            best_profit,
        )
    if math.isclose(best.assortment, assort_high, abs_tol=_BOUNDARY_ATOL):
        # best assortment pinned at max: trivial "always broadest".
        return (
            True,
            (
                f"assortment-saturated: profit-max assortment {best.assortment:.3f} is pinned at "
                f"the max ({assort_high:.2f}) -> trivial always-broadest strategy"
            ),
            best_assortment,
            best_profit,
        )

    # ROI: the best (interior) assortment must beat assortment=0 by margin.
    if not _beats_profit(
        best.mean_profit,
        best.std_error,
        zero_assort.mean_profit,
        zero_assort.std_error,
        margin_sigma_multiple,
    ):
        combined_se = math.hypot(best.std_error, zero_assort.std_error)
        return (
            True,
            (
                f"assortment-no-payoff: best assortment {best.assortment:.3f} "
                f"(profit {best.mean_profit:.2f}) does not beat assortment=0 "
                f"(profit {zero_assort.mean_profit:.2f}) by >{margin_sigma_multiple:g}x std-error: "
                f"margin {best.mean_profit - zero_assort.mean_profit:.2f} "
                f"<= threshold {margin_sigma_multiple * combined_se:.2f}"
            ),
            best_assortment,
            best_profit,
        )

    return (
        False,
        (
            f"clear interior assortment optimum at {best.assortment:.3f} "
            f"(profit {best.mean_profit:.2f}) at price {best.fixed_price:.4f}, spend "
            f"{best.fixed_spend:.3f}: beats assortment=0 by >{margin_sigma_multiple:g}x "
            f"std-error (breadth pays its opex)"
        ),
        best_assortment,
        best_profit,
    )


def promotion_verdict_for_axis(
    rows: tuple[SweepRow1DPromotion, ...],
    *,
    promo_low: float = PROMO_LOW,
    promo_high: float = PROMO_HIGH,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
) -> tuple[bool, str, float, float]:
    """The promotion-axis degeneracy verdict: (not_degenerate, reason, best_promotion, best_value)."""
    if not rows:
        raise ValueError("cannot judge an empty promotion axis")

    best = max(rows, key=lambda r: r.mean_profit)
    best_promotion = best.promotion
    best_profit = best.mean_profit

    zero_cells = [r for r in rows if math.isclose(r.promotion, promo_low, abs_tol=_BOUNDARY_ATOL)]
    if not zero_cells:
        raise ValueError("promotion axis has no promotion=0 cell; include the promotion floor")
    zero_promo = zero_cells[0]

    if math.isclose(best.promotion, promo_low, abs_tol=_BOUNDARY_ATOL):
        # best promotion == floor (== 0): promo never helps once costed + debted.
        return (
            True,
            (
                f"promotion-inert: profit-max promotion {best.promotion:.3f} is the floor "
                f"({promo_low:.2f}) -> promotion never worth using once the per-unit cost + "
                f"stockpile debt are paid (lever is inert)"
            ),
            best_promotion,
            best_profit,
        )
    if math.isclose(best.promotion, promo_high, abs_tol=_BOUNDARY_ATOL):
        # best promotion pinned at max: trivial "always-max promo" -> the
        # anti-arbitrage is not biting at steady state (stockpile/cost too weak).
        return (
            True,
            (
                f"promotion-saturated: profit-max promotion {best.promotion:.3f} is pinned at "
                f"the max ({promo_high:.2f}) -> trivial always-max-promo strategy "
                f"(the stockpile/cost is too weak; the anti-arbitrage is not biting)"
            ),
            best_promotion,
            best_profit,
        )

    # ROI: the best (interior) promotion must beat promotion=0 by margin.
    if not _beats_profit(
        best.mean_profit,
        best.std_error,
        zero_promo.mean_profit,
        zero_promo.std_error,
        margin_sigma_multiple,
    ):
        combined_se = math.hypot(best.std_error, zero_promo.std_error)
        return (
            True,
            (
                f"promotion-no-payoff: best promotion {best.promotion:.3f} "
                f"(profit {best.mean_profit:.2f}) does not beat promotion=0 "
                f"(profit {zero_promo.mean_profit:.2f}) by >{margin_sigma_multiple:g}x std-error: "
                f"margin {best.mean_profit - zero_promo.mean_profit:.2f} "
                f"<= threshold {margin_sigma_multiple * combined_se:.2f}"
            ),
            best_promotion,
            best_profit,
        )

    return (
        False,
        (
            f"clear interior promotion optimum at {best.promotion:.3f} "
            f"(profit {best.mean_profit:.2f}) at price {best.fixed_price:.4f}, spend "
            f"{best.fixed_spend:.3f}, assortment {best.fixed_assortment:.3f}: beats "
            f"promotion=0 by >{margin_sigma_multiple:g}x std-error (a sustained, "
            f"margin-costed promo pays for itself)"
        ),
        best_promotion,
        best_profit,
    )


def expansion_verdict_for_timing(
    rows: tuple[ExpansionTimingRow, ...],
    *,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
    flat_sigma_multiple: float = TIMING_FLAT_SIGMA_MULTIPLE,
) -> tuple[bool, str, int, float]:
    """The expansion-timing degeneracy verdict: (not_degenerate, reason, best_open_tick, best_total)."""
    if not rows:
        raise ValueError("cannot judge an empty expansion-timing leg")

    never_cells = [r for r in rows if r.is_never_expand]
    if not never_cells:
        raise ValueError(
            "expansion-timing leg has no never_expand baseline row "
            f"(open_tick == {NEVER_EXPAND_T}); include the never-expand baseline"
        )
    never = never_cells[0]

    expand_rows = [r for r in rows if not r.is_never_expand]
    if not expand_rows:
        raise ValueError(
            "expansion-timing leg has no expand_at_T policy rows; "
            "include at least one expand_at_T policy"
        )

    best = max(expand_rows, key=lambda r: r.mean_profit)
    best_open_tick = best.open_tick
    best_profit = best.mean_profit

    # --- Leg (a): the best expand_at_T must beat never_expand by margin. ---
    if not _beats_profit(
        best.mean_profit,
        best.std_error,
        never.mean_profit,
        never.std_error,
        margin_sigma_multiple,
    ):
        combined_se = math.hypot(best.std_error, never.std_error)
        return (
            True,
            (
                f"expansion-never-pays: best expand_at_T (T={best.open_tick}, profit "
                f"{best.mean_profit:.2f}) does not beat never_expand (profit "
                f"{never.mean_profit:.2f}) by >{margin_sigma_multiple:g}x std-error: "
                f"margin {best.mean_profit - never.mean_profit:.2f} "
                f"<= threshold {margin_sigma_multiple * combined_se:.2f} "
                f"(capex/opex too high or region-1 too small/unloyal)"
            ),
            best_open_tick,
            best_profit,
        )

    # --- Leg (b): the timing curve must be NON-flat (some T meaningfully worse). ---
    worst = min(expand_rows, key=lambda r: r.mean_profit)
    timing_matters = _beats_profit(
        best.mean_profit,
        best.std_error,
        worst.mean_profit,
        worst.std_error,
        flat_sigma_multiple,
    )
    if not timing_matters:
        # The curve is flat. If T==0 is the best (immediate dominates flat) it is the
        # free-lunch verdict; otherwise it is still flat-degenerate (timing inert).
        combined_se = math.hypot(best.std_error, worst.std_error)
        immediate_best = best.open_tick == 0
        label = "expansion-free-lunch" if immediate_best else "expansion-timing-flat"
        detail = (
            "expand_immediately (T=0) dominates and the timing curve is flat -> "
            "expansion is a no-brainer always-open-at-0 (capex/opex too low or "
            "region-1 too rich)"
            if immediate_best
            else (
                "the best expand_at_T beats never_expand but the timing curve is flat "
                "-> WHEN to open does not matter (timing is not a real decision)"
            )
        )
        return (
            True,
            (
                f"{label}: timing curve is flat (best T={best.open_tick} profit "
                f"{best.mean_profit:.2f} does not beat worst T={worst.open_tick} profit "
                f"{worst.mean_profit:.2f} by >{flat_sigma_multiple:g}x std-error: "
                f"margin {best.mean_profit - worst.mean_profit:.2f} "
                f"<= threshold {flat_sigma_multiple * combined_se:.2f}); {detail}"
            ),
            best_open_tick,
            best_profit,
        )

    return (
        False,
        (
            f"non-degenerate: interior timing optimum at T={best.open_tick} "
            f"(profit {best.mean_profit:.2f}) beats never_expand (profit "
            f"{never.mean_profit:.2f}) by >{margin_sigma_multiple:g}x std-error AND the "
            f"timing curve is non-flat (worst T={worst.open_tick} profit "
            f"{worst.mean_profit:.2f} is meaningfully worse) -> expansion pays AND "
            f"WHEN to open is a real decision (timing + cash-gating matter)"
        ),
        best_open_tick,
        best_profit,
    )


def research_verdict_for_fidelity(
    rows: tuple[ResearchFidelityRow, ...],
    *,
    research_low: float = RESEARCH_LOW,
    research_high: float = RESEARCH_HIGH,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
) -> tuple[bool, str, float, float]:
    """The research-axis degeneracy verdict: (not_degenerate, reason, best_spend, best_value)."""
    if not rows:
        raise ValueError("cannot judge an empty research axis")

    zero_cells = [
        r for r in rows if math.isclose(r.research_spend, research_low, abs_tol=_BOUNDARY_ATOL)
    ]
    if not zero_cells:
        raise ValueError(
            "research axis has no research=0 baseline cell (the noisiest-read floor); "
            "include the research-spend floor"
        )
    baseline = zero_cells[0]

    best = max(rows, key=lambda r: r.mean_score)
    best_research_spend = best.research_spend
    best_score = best.mean_score

    # --- Leg (a): a sharper read must beat the research=0 (noisiest-read) baseline. ---
    if not _beats_profit(
        best.mean_score,
        best.std_error,
        baseline.mean_score,
        baseline.std_error,
        margin_sigma_multiple,
    ):
        combined_se = math.hypot(best.std_error, baseline.std_error)
        return (
            True,
            (
                f"research-inert: best research spend {best.research_spend:.3f} "
                f"(score {best.mean_score:.2f}) does not beat research=0 "
                f"(score {baseline.mean_score:.2f}) by >{margin_sigma_multiple:g}x std-error: "
                f"margin {best.mean_score - baseline.mean_score:.2f} "
                f"<= threshold {margin_sigma_multiple * combined_se:.2f} "
                f"(information worthless against this opponent — noise too weak or "
                f"opponent too constant)"
            ),
            best_research_spend,
            best_score,
        )

    # --- Leg (b): the optimal research spend must be INTERIOR (not pinned at max). ---
    # research=0 best is impossible (leg (a) requires beating the baseline), so the only
    # degenerate corner left is 'pinned at the max' (free information ⇒ always buy max).
    if math.isclose(best.research_spend, research_high, abs_tol=_BOUNDARY_ATOL):
        return (
            True,
            (
                f"research-free: optimal research spend {best.research_spend:.3f} is pinned at "
                f"the max ({research_high:.2f}) -> a sharper read pays but research never costs "
                f"enough to hold back (free information, always-buy-max is trivially best); "
                f"the cost (cost_per_fidelity) is too low to make the optimum interior"
            ),
            best_research_spend,
            best_score,
        )

    return (
        False,
        (
            f"non-degenerate: interior research optimum at spend {best.research_spend:.3f} "
            f"(fidelity {best.fidelity:.3f}, score {best.mean_score:.2f}) beats research=0 "
            f"(score {baseline.mean_score:.2f}) by >{margin_sigma_multiple:g}x std-error AND "
            f"is interior (not pinned at the max) -> a sharper read is worth more than the "
            f"noise AND the research cost makes the optimal fidelity a real spend-vs-value "
            f"tradeoff (value of information against a varying opponent)"
        ),
        best_research_spend,
        best_score,
    )
