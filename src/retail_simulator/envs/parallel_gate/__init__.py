"""The parallel self-play learnability-gate package (SuperSuit<->SB3 bridge + gates).

Splits the old single-file ``parallel_gate`` harness into focused modules:
``bridge`` (vector-env construction + bridge smoke), ``stationary`` (stationary and
head-to-head evaluation), ``lever_eval`` (on-vs-off research/SCM/automation/loyalty
gates), ``lever_usage`` (lever-usage and uptake diagnostics), and ``gate`` (the
train-then-gate runner + report + diversity). The package re-exports the frozen public
API plus a few private helpers that other env modules import.
"""

from __future__ import annotations

from retail_simulator.core.reward import weighted_objective

from .bridge import (
    BRIDGE_SMOKE_STEPS,
    DEFAULT_N_LEARNING_AGENTS,
    DIVERSITY_MAX_RATIO,
    DIVERSITY_WITHIN_FRACTION,
    LSTM_HIDDEN_SIZE,
    BridgeSmokeResult,
    build_parallel_vec_env,
    competition_config,
    run_bridge_smoke,
)

# Private: reused by envs/league/runner.
from .bridge import _make_curve_callback as _make_curve_callback
from .bridge import _make_shared_ppo as _make_shared_ppo
from .gate import (
    DiversityReport,
    build_parallel_report,
    diversity_report,
    run_parallel_gate,
)
from .lever_eval import (
    automation_force_default_action,
    evaluate_automation_on_vs_off,
    evaluate_loyalty_on_vs_off,
    evaluate_research_on_vs_off,
    evaluate_scm_on_vs_off,
    loyalty_force_default_action,
    research_force_off_action,
    scm_force_default_action,
)
from .lever_usage import (
    AUTOMATION_UPTAKE_TAU,
    LOYALTY_USAGE_MARGIN_MEAN,
    LOYALTY_USAGE_MARGIN_STD,
    USAGE_MARGIN_MEAN,
    USAGE_MARGIN_STD,
    AutomationUptake,
    LoyaltyLeverUsage,
    SCMLeverUsage,
    measure_automation_uptake,
    measure_loyalty_lever_usage,
    measure_scm_lever_usage,
)

# Private: reused by envs/curriculum + envs/hotseat + envs/league/evaluation.
from .stationary import _components_from_info as _components_from_info
from .stationary import _episode_stationary_score as _episode_stationary_score
from .stationary import (
    evaluate_head_to_head,
    evaluate_random_stationary,
    evaluate_shared_policy_stationary,
)

__all__ = [
    "AUTOMATION_UPTAKE_TAU",
    "BRIDGE_SMOKE_STEPS",
    "DEFAULT_N_LEARNING_AGENTS",
    "DIVERSITY_MAX_RATIO",
    "DIVERSITY_WITHIN_FRACTION",
    "LOYALTY_USAGE_MARGIN_MEAN",
    "LOYALTY_USAGE_MARGIN_STD",
    "LSTM_HIDDEN_SIZE",
    "USAGE_MARGIN_MEAN",
    "USAGE_MARGIN_STD",
    "AutomationUptake",
    "BridgeSmokeResult",
    "DiversityReport",
    "LoyaltyLeverUsage",
    "SCMLeverUsage",
    "automation_force_default_action",
    "build_parallel_report",
    "build_parallel_vec_env",
    "competition_config",
    "diversity_report",
    "evaluate_automation_on_vs_off",
    "evaluate_head_to_head",
    "evaluate_loyalty_on_vs_off",
    "evaluate_random_stationary",
    "evaluate_research_on_vs_off",
    "evaluate_scm_on_vs_off",
    "evaluate_shared_policy_stationary",
    "loyalty_force_default_action",
    "measure_automation_uptake",
    "measure_loyalty_lever_usage",
    "measure_scm_lever_usage",
    "research_force_off_action",
    "run_bridge_smoke",
    "run_parallel_gate",
    "scm_force_default_action",
    "weighted_objective",
]
