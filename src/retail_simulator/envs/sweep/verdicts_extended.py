"""Extended sweep verdicts: SCM service-level, automation-timing, loyalty-spend, and orthogonality."""

from __future__ import annotations

import math

from retail_simulator.envs.sweep.rows import (
    _BOUNDARY_ATOL,
    LOYALTY_HIGH,
    LOYALTY_LOW,
    MARGIN_SIGMA_MULTIPLE,
    NEVER_UPGRADE_T,
    SERVICE_HIGH,
    SERVICE_LOW,
    TIMING_FLAT_SIGMA_MULTIPLE,
    LoyaltyVsPromotionOrthogonality,
    SweepRow1DAutomation,
    SweepRow1DLoyalty,
    SweepRow1DService,
    _beats_profit,
)


def scm_verdict_for_service(
    rows: tuple[SweepRow1DService, ...],
    variable_mean_profit: float,
    variable_std_error: float,
    *,
    service_low: float = SERVICE_LOW,
    service_high: float = SERVICE_HIGH,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
) -> tuple[bool, str, float, float]:
    """The SCM-axis degeneracy verdict: (not_degenerate, reason, best_service, best_value)."""
    if not rows:
        raise ValueError("cannot judge an empty SCM service-level axis")

    low_cells = [
        r for r in rows if math.isclose(r.service_level, service_low, abs_tol=_BOUNDARY_ATOL)
    ]
    high_cells = [
        r for r in rows if math.isclose(r.service_level, service_high, abs_tol=_BOUNDARY_ATOL)
    ]
    if not low_cells or not high_cells:
        raise ValueError(
            "SCM service-level axis must include both the service_low (0.0) and "
            "service_high (1.0) endpoint cells so the constant-baseline 'pinned at "
            "extreme' classification is well-defined"
        )

    best = max(rows, key=lambda r: r.mean_profit)
    best_constant_service = best.service_level
    best_constant_profit = best.mean_profit

    # --- Leg (a): best constant must be INTERIOR (not pinned at either extreme). ---
    if math.isclose(best.service_level, service_low, abs_tol=_BOUNDARY_ATOL):
        return (
            True,
            (
                f"scm-pinned-at-extreme: best constant-service {best.service_level:.3f} is "
                f"pinned at the floor ({service_low:.2f}) -> always-cheapest-supplier is a "
                f"free cost cut (a free PPO learner converges to s=0 and does not use the "
                f"lever); cogs_premium too high or lost_sales_share_penalty too low"
            ),
            best_constant_service,
            best_constant_profit,
        )
    if math.isclose(best.service_level, service_high, abs_tol=_BOUNDARY_ATOL):
        return (
            True,
            (
                f"scm-pinned-at-extreme: best constant-service {best.service_level:.3f} is "
                f"pinned at the max ({service_high:.2f}) -> always-perfect-service is a "
                f"trivial dominant strategy (a free PPO learner converges to s=1 and does "
                f"not use the lever); cogs_premium too low or lost_sales_share_penalty too high"
            ),
            best_constant_service,
            best_constant_profit,
        )

    # --- Leg (b): the variable comparator must beat the best constant by margin. ---
    if not _beats_profit(
        variable_mean_profit,
        variable_std_error,
        best_constant_profit,
        best.std_error,
        margin_sigma_multiple,
    ):
        combined_se = math.hypot(variable_std_error, best.std_error)
        return (
            True,
            (
                f"scm-inert: per-region-optimal variable-service score "
                f"{variable_mean_profit:.2f} does not beat best constant-service "
                f"{best_constant_service:.3f} (profit {best_constant_profit:.2f}) by "
                f">{margin_sigma_multiple:g}x std-error: margin "
                f"{variable_mean_profit - best_constant_profit:.2f} <= threshold "
                f"{margin_sigma_multiple * combined_se:.2f} (the per-region asymmetry is not "
                f"worth conditioning on -> a free PPO learner that picks the best fixed "
                f"value matches the per-region-conditional alternative)"
            ),
            best_constant_service,
            best_constant_profit,
        )

    return (
        False,
        (
            f"non-degenerate: best constant-service {best_constant_service:.3f} (profit "
            f"{best_constant_profit:.2f}) is INTERIOR (not pinned at either extreme) AND "
            f"per-region-optimal variable-service score {variable_mean_profit:.2f} beats "
            f"the best constant by >{margin_sigma_multiple:g}x std-error -> a free PPO "
            f"learner that conditions on observable own state can extract value beyond the "
            f"best fixed setting (the SCM lever's per-region tradeoff is genuinely worth "
            f"conditioning on)"
        ),
        best_constant_service,
        best_constant_profit,
    )


def automation_verdict_for_timing(
    rows: tuple[SweepRow1DAutomation, ...],
    *,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
    flat_sigma_multiple: float = TIMING_FLAT_SIGMA_MULTIPLE,
) -> tuple[bool, str, str, float]:
    """The automation-timing degeneracy verdict: (not_degenerate, reason, best_policy_name, best_total)."""
    if not rows:
        raise ValueError("cannot judge an empty automation upgrade-timing leg")

    never_cells = [r for r in rows if r.is_never_upgrade]
    if not never_cells:
        raise ValueError(
            "automation upgrade-timing leg has no never_upgrade baseline row "
            f"(upgrade_tick == {NEVER_UPGRADE_T}); include the never-upgrade baseline"
        )
    never = never_cells[0]

    upgrade_rows = [r for r in rows if not r.is_never_upgrade]
    if not upgrade_rows:
        raise ValueError(
            "automation upgrade-timing leg has no upgrade-policy rows; "
            "include at least one upgrade_to_K_at_t=T policy"
        )

    best = max(upgrade_rows, key=lambda r: r.mean_profit)
    best_policy_name = best.policy_name
    best_profit = best.mean_profit

    # --- Leg (a): the best upgrade policy must beat never_upgrade by margin. ---
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
                f"automation-never-pays: best upgrade policy {best.policy_name} "
                f"(profit {best.mean_profit:.2f}) does not beat never_upgrade "
                f"(profit {never.mean_profit:.2f}) by "
                f">{margin_sigma_multiple:g}x std-error: margin "
                f"{best.mean_profit - never.mean_profit:.2f} <= threshold "
                f"{margin_sigma_multiple * combined_se:.2f} (capex_per_tier too "
                f"high OR savings_per_tier too low — no upgrade timing recoups "
                f"the one-shot capex over the remaining window)"
            ),
            best_policy_name,
            best_profit,
        )

    # --- Leg (b): the timing curve must be NON-flat AND t=0 must NOT be a free dominator. ---
    worst = min(upgrade_rows, key=lambda r: r.mean_profit)
    timing_matters = _beats_profit(
        best.mean_profit,
        best.std_error,
        worst.mean_profit,
        worst.std_error,
        flat_sigma_multiple,
    )
    immediate_t2 = next(
        (r for r in upgrade_rows if r.target_tier == 2 and r.upgrade_tick == 0),
        None,
    )
    immediate_is_best = immediate_t2 is not None and immediate_t2.policy_name == best.policy_name

    if not timing_matters:
        combined_se = math.hypot(best.std_error, worst.std_error)
        label = "automation-free-lunch" if immediate_is_best else "automation-timing-flat"
        detail = (
            "upgrade_to_2_at_t=0 dominates and the timing curve is flat -> "
            "automation is a no-brainer always-upgrade-at-0 (capex_per_tier "
            "too low OR savings_per_tier too high OR STARTING_CASH too high — "
            "the cash-gating is not biting)"
            if immediate_is_best
            else (
                "the best upgrade policy beats never_upgrade but the timing "
                "curve is flat -> WHEN to upgrade does not matter (timing is "
                "not a real decision)"
            )
        )
        return (
            True,
            (
                f"{label}: timing curve is flat (best {best.policy_name} profit "
                f"{best.mean_profit:.2f} does not beat worst {worst.policy_name} "
                f"profit {worst.mean_profit:.2f} by >{flat_sigma_multiple:g}x "
                f"std-error: margin {best.mean_profit - worst.mean_profit:.2f} "
                f"<= threshold {flat_sigma_multiple * combined_se:.2f}); {detail}"
            ),
            best_policy_name,
            best_profit,
        )

    return (
        False,
        (
            f"non-degenerate: interior upgrade-timing optimum at "
            f"{best.policy_name} (profit {best.mean_profit:.2f}) beats "
            f"never_upgrade (profit {never.mean_profit:.2f}) by "
            f">{margin_sigma_multiple:g}x std-error AND the timing curve is "
            f"non-flat (worst {worst.policy_name} profit {worst.mean_profit:.2f} "
            f"is meaningfully worse) -> automation pays AND WHEN to upgrade is "
            f"a real decision (timing + cash-gating matter)"
        ),
        best_policy_name,
        best_profit,
    )


def loyalty_verdict_for_spend(
    rows: tuple[SweepRow1DLoyalty, ...],
    *,
    loyalty_low: float = LOYALTY_LOW,
    loyalty_high: float = LOYALTY_HIGH,
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
) -> tuple[bool, str, float, float]:
    """The loyalty-axis degeneracy verdict: (not_degenerate, reason, best_spend, best_value)."""
    if not rows:
        raise ValueError("cannot judge an empty loyalty-spend axis")

    low_cells = [
        r for r in rows if math.isclose(r.loyalty_spend, loyalty_low, abs_tol=_BOUNDARY_ATOL)
    ]
    high_cells = [
        r for r in rows if math.isclose(r.loyalty_spend, loyalty_high, abs_tol=_BOUNDARY_ATOL)
    ]
    if not low_cells or not high_cells:
        raise ValueError(
            "loyalty-spend axis must include both the loyalty_low (0.0) and "
            "loyalty_high (1.0) endpoint cells so the constant-baseline 'pinned at "
            "extreme' classification is well-defined"
        )

    best = max(rows, key=lambda r: r.mean_profit)
    best_loyalty_spend = best.loyalty_spend
    best_profit = best.mean_profit

    low_cell = low_cells[0]  # the no-program baseline (the FREE alternative anchor)
    high_cell = high_cells[0]  # the max-spend cell — the "always-spend-max" candidate

    # --- Leg (a): pinned-at-zero — 'no program' dominates ---
    if math.isclose(best.loyalty_spend, loyalty_low, abs_tol=_BOUNDARY_ATOL):
        return (
            True,
            (
                f"loyalty-pinned-at-zero: best constant loyalty_spend "
                f"{best_loyalty_spend:.3f} is pinned at the floor "
                f"({loyalty_low:.2f}) -> the no-program baseline dominates (a free "
                f"PPO learner converges to spend=0 and does not USE the lever); "
                f"beta_program too low OR cost_per_unit_spend too high"
            ),
            best_loyalty_spend,
            best_profit,
        )

    # --- Leg (b): loyalty-free — 'always spend max' dominates ---
    if math.isclose(best.loyalty_spend, loyalty_high, abs_tol=_BOUNDARY_ATOL):
        return (
            True,
            (
                f"loyalty-free: best constant loyalty_spend "
                f"{best_loyalty_spend:.3f} is pinned at the max "
                f"({loyalty_high:.2f}) -> always-spend-max is the trivial dominant "
                f"strategy (a free PPO learner converges to spend=1 and the gate "
                f"trivially passes); cost_per_unit_spend too low OR beta_program "
                f"too high"
            ),
            best_loyalty_spend,
            best_profit,
        )

    # --- Leg (c): loyalty-inert — the lever does not move profit by margin ---
    beats_low = _beats_profit(
        best.mean_profit,
        best.std_error,
        low_cell.mean_profit,
        low_cell.std_error,
        margin_sigma_multiple,
    )
    beats_high = _beats_profit(
        best.mean_profit,
        best.std_error,
        high_cell.mean_profit,
        high_cell.std_error,
        margin_sigma_multiple,
    )
    if not (beats_low and beats_high):
        low_combined = math.hypot(best.std_error, low_cell.std_error)
        high_combined = math.hypot(best.std_error, high_cell.std_error)
        return (
            True,
            (
                f"loyalty-inert: best constant loyalty_spend "
                f"{best_loyalty_spend:.3f} (profit {best_profit:.2f}) does not beat "
                f"BOTH endpoints by >{margin_sigma_multiple:g}x std-error: "
                f"vs floor margin {best.mean_profit - low_cell.mean_profit:.2f} "
                f"(threshold {margin_sigma_multiple * low_combined:.2f}) AND "
                f"vs max margin {best.mean_profit - high_cell.mean_profit:.2f} "
                f"(threshold {margin_sigma_multiple * high_combined:.2f}); "
                f"calibration miss - beta_program too low OR cost_per_unit_spend "
                f"too high (a free PPO learner finds no gradient on the lever)"
            ),
            best_loyalty_spend,
            best_profit,
        )

    return (
        False,
        (
            f"non-degenerate: best constant loyalty_spend {best_loyalty_spend:.3f} "
            f"(profit {best_profit:.2f}) is INTERIOR AND beats BOTH endpoints by "
            f">{margin_sigma_multiple:g}x std-error (vs no-program "
            f"{low_cell.mean_profit:.2f}, vs max-spend {high_cell.mean_profit:.2f}) "
            f"-> a real cost-vs-boost tradeoff; a free PPO learner that conditions "
            f"on observable own state (per-region loyalty stock + brand-loyal mix + "
            f"own margin) can extract value beyond the best fixed setting"
        ),
        best_loyalty_spend,
        best_profit,
    )


def loyalty_vs_promotion_orthogonality_verdict(
    *,
    base_loyalty: float,
    base_promotion: float,
    base_profit: float,
    base_std_error: float,
    swap_profits: tuple[tuple[float, float, float, float], ...],
    margin_sigma_multiple: float = MARGIN_SIGMA_MULTIPLE,
) -> LoyaltyVsPromotionOrthogonality:
    """The orthogonality probe verdict: at least one loyalty<->promotion swap must move profit by margin."""
    if not swap_profits:
        return LoyaltyVsPromotionOrthogonality(
            base_loyalty=float(base_loyalty),
            base_promotion=float(base_promotion),
            base_profit=float(base_profit),
            swap_profits=(),
            max_swap_delta=0.0,
            threshold=0.0,
            passed=False,
            reason=(
                "no swaps tested: the orthogonality-vs-promotion diagnostic ran "
                "with zero substitution cells (degenerate FAIL — the diagnostic "
                "cannot conclude levers are orthogonal without at least one swap)"
            ),
        )

    best_swap = max(swap_profits, key=lambda s: abs(s[2] - float(base_profit)))
    swap_loyalty, swap_promotion, swap_profit, swap_se = best_swap
    max_delta = abs(swap_profit - float(base_profit))
    combined_se = math.hypot(float(base_std_error), float(swap_se))
    threshold = float(margin_sigma_multiple) * combined_se

    if max_delta > threshold:
        return LoyaltyVsPromotionOrthogonality(
            base_loyalty=float(base_loyalty),
            base_promotion=float(base_promotion),
            base_profit=float(base_profit),
            swap_profits=tuple(swap_profits),
            max_swap_delta=float(max_delta),
            threshold=float(threshold),
            passed=True,
            reason=(
                f"orthogonal: strict substitution (loyalty {swap_loyalty:.3f}, "
                f"promotion {swap_promotion:.3f}) moved profit by {max_delta:.2f} "
                f"(>{margin_sigma_multiple:g}x combined std-error = {threshold:.2f}) "
                f"vs the base (loyalty {base_loyalty:.3f}, promotion "
                f"{base_promotion:.3f}, profit {base_profit:.2f}); the levers are "
                f"NOT economically substitutable (the boost is moat-anchored and "
                f"the lift is flat — structurally distinct)"
            ),
        )

    return LoyaltyVsPromotionOrthogonality(
        base_loyalty=float(base_loyalty),
        base_promotion=float(base_promotion),
        base_profit=float(base_profit),
        swap_profits=tuple(swap_profits),
        max_swap_delta=float(max_delta),
        threshold=float(threshold),
        passed=False,
        reason=(
            f"degenerate: every strict promotion<->loyalty substitution preserved "
            f"profit within {margin_sigma_multiple:g}x combined std-error "
            f"(max |delta| {max_delta:.2f} <= threshold {threshold:.2f}); the "
            f"two levers are economically isomorphic at the joint optima — "
            f"loyalty has collapsed into 'promotion with extra steps' (the boost "
            f"is no longer moat-anchored)"
        ),
    )
