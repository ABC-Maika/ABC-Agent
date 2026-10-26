"""Sweep result formatting and CSV row extraction."""

from __future__ import annotations

import math

from retail_simulator.envs.sweep.rows import SCORING_WINDOW_TICKS, SweepResult


def format_sweep_table(result: SweepResult) -> str:
    """Render the sweep result as a human-readable table."""
    header = f"{'price':>8} {'spend':>8} {'mean_profit':>14} {'std_err':>10} {'mkt_share':>10}"
    sep = "-" * len(header)
    lines = [
        (
            f"== 2-D price x marketing sweep (assortment pinned at {result.pinned_assortment:.3f}, "
            f"promotion pinned at {result.pinned_promotion:.3f}) =="
        ),
        header,
        sep,
    ]
    best_price, best_spend = result.profit_max_cell
    for row in result.rows:
        is_best = math.isclose(row.price_index, best_price) and math.isclose(
            row.spend_fraction, best_spend
        )
        flag = " *" if is_best else "  "
        lines.append(
            f"{row.price_index:>8.4f} {row.spend_fraction:>8.3f} {row.mean_profit:>14.2f} "
            f"{row.std_error:>10.3f} {row.mean_market_share:>10.4f}{flag}"
        )
    lines.append(sep)
    lines.append(
        f"profit-max cell: price {best_price:.4f}, spend {best_spend:.3f}  "
        f"(profit {result.profit_max_value:.2f})"
    )
    lines.append(f"reason: {result.reason}")

    a_header = f"{'assort':>8} {'mean_profit':>14} {'std_err':>10} {'mkt_share':>10}"
    a_sep = "-" * len(a_header)
    lines.append("")
    lines.append(f"== 1-D assortment axis at price {best_price:.4f}, spend {best_spend:.3f} ==")
    lines.append(a_header)
    lines.append(a_sep)
    best_assort_row = max(result.assortment_rows, key=lambda r: r.mean_profit, default=None)
    best_assort = best_assort_row.assortment if best_assort_row is not None else float("nan")
    for arow in result.assortment_rows:
        is_best = math.isclose(arow.assortment, best_assort)
        flag = " *" if is_best else "  "
        lines.append(
            f"{arow.assortment:>8.3f} {arow.mean_profit:>14.2f} "
            f"{arow.std_error:>10.3f} {arow.mean_market_share:>10.4f}{flag}"
        )
    lines.append(a_sep)
    lines.append(
        f"profit-max assortment: {best_assort:.3f}  (profit {result.assortment_max_value:.2f})"
    )
    lines.append(f"reason: {result.assortment_reason}")

    p_header = f"{'promo':>8} {'mean_profit':>14} {'std_err':>10} {'mkt_share':>10}"
    p_sep = "-" * len(p_header)
    lines.append("")
    lines.append(
        f"== 1-D promotion axis at price {best_price:.4f}, spend {best_spend:.3f}, "
        f"assortment {result.assortment_max_cell:.3f} (steady state) =="
    )
    lines.append(p_header)
    lines.append(p_sep)
    best_promo_row = max(result.promotion_rows, key=lambda r: r.mean_profit, default=None)
    best_promo = best_promo_row.promotion if best_promo_row is not None else float("nan")
    for prow in result.promotion_rows:
        is_best = math.isclose(prow.promotion, best_promo)
        flag = " *" if is_best else "  "
        lines.append(
            f"{prow.promotion:>8.3f} {prow.mean_profit:>14.2f} "
            f"{prow.std_error:>10.3f} {prow.mean_market_share:>10.4f}{flag}"
        )
    lines.append(p_sep)
    lines.append(
        f"profit-max promotion: {best_promo:.3f}  (profit {result.promotion_max_value:.2f})"
    )
    lines.append(f"reason: {result.promotion_reason}")

    if result.expansion_timing_rows:
        t_header = f"{'open_tick':>10} {'total_profit':>14} {'std_err':>10}"
        t_sep = "-" * len(t_header)
        lines.append("")
        lines.append(
            f"== expansion-timing leg at price {best_price:.4f}, spend {best_spend:.3f}, "
            f"assortment {result.assortment_max_cell:.3f}, promotion {best_promo:.3f} "
            f"(total profit over {SCORING_WINDOW_TICKS}-tick window) =="
        )
        lines.append(t_header)
        lines.append(t_sep)
        expand_rows = [r for r in result.expansion_timing_rows if not r.is_never_expand]
        best_timing = max(expand_rows, key=lambda r: r.mean_profit, default=None)
        best_tick = best_timing.open_tick if best_timing is not None else None
        for trow in result.expansion_timing_rows:
            label = "never" if trow.is_never_expand else f"{trow.open_tick}"
            is_best = (not trow.is_never_expand) and trow.open_tick == best_tick
            flag = " *" if is_best else "  "
            lines.append(f"{label:>10} {trow.mean_profit:>14.2f} {trow.std_error:>10.3f}{flag}")
        lines.append(t_sep)
        lines.append(f"reason: {result.expansion_timing_reason}")

    if result.research_rows:
        r_header = (
            f"{'research':>10} {'fidelity':>10} {'mean_score':>14} "
            f"{'std_err':>10} {'mkt_share':>10}"
        )
        r_sep = "-" * len(r_header)
        lines.append("")
        lines.append(
            "== research (value-of-information) axis vs the wandering NPC "
            "(exogenous, non-inferable varying opponent; reactive best-response heuristic) =="
        )
        lines.append(r_header)
        lines.append(r_sep)
        best_research = max(result.research_rows, key=lambda r: r.mean_score, default=None)
        best_spend = best_research.research_spend if best_research is not None else float("nan")
        for rrow in result.research_rows:
            is_best = math.isclose(rrow.research_spend, best_spend)
            flag = " *" if is_best else "  "
            lines.append(
                f"{rrow.research_spend:>10.3f} {rrow.fidelity:>10.3f} {rrow.mean_score:>14.2f} "
                f"{rrow.std_error:>10.3f} {rrow.mean_market_share:>10.4f}{flag}"
            )
        lines.append(r_sep)
        lines.append(f"reason: {result.research_reason}")

    if result.service_rows:
        s_header = f"{'service':>10} {'mean_profit':>14} {'std_err':>10} {'mkt_share':>10}"
        s_sep = "-" * len(s_header)
        lines.append("")
        lines.append(
            "== SCM (value-of-service) axis vs the default Discounter (SCM scenario ON; "
            "the constant-service grid is the FREE no-SCM baseline a free PPO learner "
            "collapses to) =="
        )
        lines.append(s_header)
        lines.append(s_sep)
        best_scm_row = max(result.service_rows, key=lambda r: r.mean_profit, default=None)
        best_scm = best_scm_row.service_level if best_scm_row is not None else float("nan")
        for srow in result.service_rows:
            is_best = math.isclose(srow.service_level, best_scm)
            flag = " *" if is_best else "  "
            lines.append(
                f"{srow.service_level:>10.3f} {srow.mean_profit:>14.2f} "
                f"{srow.std_error:>10.3f} {srow.mean_market_share:>10.4f}{flag}"
            )
        lines.append(s_sep)
        if result.service_variable_mean is not None:
            v_mean = result.service_variable_mean
            v_se = result.service_variable_std_error if result.service_variable_std_error else 0.0
            lines.append(
                f"per-region-optimal (variable) score: {v_mean:.2f} +/- {v_se:.3f} "
                f"(the FREE-variable alternative — beats best constant by margin ⇒ "
                f"non-degenerate)"
            )
        lines.append(f"reason: {result.service_reason}")

    if result.automation_rows:
        a2_header = f"{'policy':>24} {'total_profit':>14} {'std_err':>10}"
        a2_sep = "-" * len(a2_header)
        lines.append("")
        lines.append(
            "== automation upgrade-timing leg vs the default Discounter "
            "(automation scenario ON; never_upgrade is the FREE no-automation "
            "baseline) =="
        )
        lines.append(a2_header)
        lines.append(a2_sep)
        upgrade_rows = [r for r in result.automation_rows if not r.is_never_upgrade]
        best_auto_row = max(upgrade_rows, key=lambda r: r.mean_profit, default=None)
        best_policy_name = best_auto_row.policy_name if best_auto_row is not None else ""
        for auto_row in result.automation_rows:
            is_best = (not auto_row.is_never_upgrade) and auto_row.policy_name == best_policy_name
            flag = " *" if is_best else "  "
            lines.append(
                f"{auto_row.policy_name:>24} {auto_row.mean_profit:>14.2f} "
                f"{auto_row.std_error:>10.3f}{flag}"
            )
        lines.append(a2_sep)
        lines.append(f"reason: {result.automation_reason}")

    if result.loyalty_rows:
        l_header = f"{'loyalty':>10} {'mean_profit':>14} {'std_err':>10} {'mkt_share':>10}"
        l_sep = "-" * len(l_header)
        lines.append("")
        lines.append(
            "== loyalty-program (value-of-active-loyalty) axis vs the default "
            "Discounter (loyalty scenario ON; the constant-loyalty grid is the "
            "FREE no-program baseline a free PPO learner collapses to) =="
        )
        lines.append(l_header)
        lines.append(l_sep)
        best_loy_row = max(result.loyalty_rows, key=lambda r: r.mean_profit, default=None)
        best_loy = best_loy_row.loyalty_spend if best_loy_row is not None else float("nan")
        for lrow in result.loyalty_rows:
            is_best = math.isclose(lrow.loyalty_spend, best_loy)
            flag = " *" if is_best else "  "
            lines.append(
                f"{lrow.loyalty_spend:>10.3f} {lrow.mean_profit:>14.2f} "
                f"{lrow.std_error:>10.3f} {lrow.mean_market_share:>10.4f}{flag}"
            )
        lines.append(l_sep)
        lines.append(f"reason: {result.loyalty_reason}")
        if result.loyalty_orthogonality is not None:
            ortho = result.loyalty_orthogonality
            lines.append(
                f"orthogonality-vs-promotion: max-swap |delta| "
                f"{ortho.max_swap_delta:.2f} vs threshold {ortho.threshold:.2f} "
                f"(passed={ortho.passed}); {ortho.reason}"
            )

    lines.append("")
    verdict = "DEGENERATE" if result.is_degenerate else "NON-DEGENERATE"
    legs = "price AND marketing AND assortment AND promotion"
    if result.expansion_timing_rows:
        legs += " AND expansion-timing"
    if result.research_rows:
        legs += " AND research"
    if result.service_rows:
        legs += " AND SCM-service"
    if result.automation_rows:
        legs += " AND automation"
    if result.loyalty_rows:
        legs += " AND loyalty-program"
    lines.append(f"overall verdict: {verdict} (non-degenerate on {legs})")
    return "\n".join(lines)


def sweep_csv_rows(result: SweepResult) -> list[dict[str, float | int]]:
    """Return the 2-D sweep cells as flat dicts for CSV writing (pandas-readable)."""
    return [
        {
            "price_index": row.price_index,
            "spend_fraction": row.spend_fraction,
            "mean_profit": row.mean_profit,
            "std_error": row.std_error,
            "mean_market_share": row.mean_market_share,
            "n_samples": row.n_samples,
        }
        for row in result.rows
    ]


def assortment_axis_csv_rows(result: SweepResult) -> list[dict[str, float | int]]:
    """Return the assortment-axis cells as flat dicts for CSV writing."""
    return [
        {
            "assortment": row.assortment,
            "fixed_price": row.fixed_price,
            "fixed_spend": row.fixed_spend,
            "mean_profit": row.mean_profit,
            "std_error": row.std_error,
            "mean_market_share": row.mean_market_share,
            "n_samples": row.n_samples,
        }
        for row in result.assortment_rows
    ]


def promotion_axis_csv_rows(result: SweepResult) -> list[dict[str, float | int]]:
    """Return the promotion-axis cells as flat dicts for CSV writing."""
    return [
        {
            "promotion": row.promotion,
            "fixed_price": row.fixed_price,
            "fixed_spend": row.fixed_spend,
            "fixed_assortment": row.fixed_assortment,
            "mean_profit": row.mean_profit,
            "std_error": row.std_error,
            "mean_market_share": row.mean_market_share,
            "n_samples": row.n_samples,
        }
        for row in result.promotion_rows
    ]


def expansion_timing_csv_rows(result: SweepResult) -> list[dict[str, float | int]]:
    """Return the expansion-timing policies as flat dicts for CSV writing."""
    return [
        {
            "open_tick": row.open_tick,
            "mean_profit": row.mean_profit,
            "std_error": row.std_error,
            "n_seeds": row.n_seeds,
        }
        for row in result.expansion_timing_rows
    ]


def research_axis_csv_rows(result: SweepResult) -> list[dict[str, float | int]]:
    """Return the research axis cells as flat dicts for CSV writing."""
    return [
        {
            "research_spend": row.research_spend,
            "fidelity": row.fidelity,
            "mean_score": row.mean_score,
            "std_error": row.std_error,
            "mean_market_share": row.mean_market_share,
            "n_samples": row.n_samples,
        }
        for row in result.research_rows
    ]


def scm_axis_csv_rows(result: SweepResult) -> list[dict[str, float | int]]:
    """Return the SCM axis cells as flat dicts for CSV writing."""
    return [
        {
            "service_level": row.service_level,
            "fixed_price": row.fixed_price,
            "fixed_spend": row.fixed_spend,
            "fixed_assortment": row.fixed_assortment,
            "fixed_promotion": row.fixed_promotion,
            "mean_profit": row.mean_profit,
            "std_error": row.std_error,
            "mean_market_share": row.mean_market_share,
            "n_samples": row.n_samples,
        }
        for row in result.service_rows
    ]


def automation_axis_csv_rows(result: SweepResult) -> list[dict[str, float | int | str]]:
    """Return the automation upgrade-timing axis cells as flat dicts for CSV writing."""
    return [
        {
            "policy_name": row.policy_name,
            "target_tier": row.target_tier,
            "upgrade_tick": row.upgrade_tick,
            "ramp_tick": row.ramp_tick,
            "fixed_price": row.fixed_price,
            "fixed_spend": row.fixed_spend,
            "fixed_assortment": row.fixed_assortment,
            "fixed_promotion": row.fixed_promotion,
            "mean_profit": row.mean_profit,
            "std_error": row.std_error,
            "n_seeds": row.n_seeds,
        }
        for row in result.automation_rows
    ]


def loyalty_axis_csv_rows(result: SweepResult) -> list[dict[str, float | int]]:
    """Return the loyalty-program axis cells as flat dicts for CSV writing."""
    return [
        {
            "loyalty_spend": row.loyalty_spend,
            "fixed_price": row.fixed_price,
            "fixed_spend": row.fixed_spend,
            "fixed_assortment": row.fixed_assortment,
            "fixed_promotion": row.fixed_promotion,
            "mean_profit": row.mean_profit,
            "std_error": row.std_error,
            "mean_market_share": row.mean_market_share,
            "n_samples": row.n_samples,
        }
        for row in result.loyalty_rows
    ]
