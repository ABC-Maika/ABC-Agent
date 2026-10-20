"""Flat subpackage for the analytical pricing sweep.

The public module path ``retail_simulator.envs.sweep`` is preserved here so that
all existing importers (cli.py, curriculum.py, parallel_gate.py) continue to work
without any changes to their import statements.
"""

from retail_simulator.envs.sweep.axes import (
    research_gate_segments,
    research_opponent_config,
)
from retail_simulator.envs.sweep.axes_extended import (
    automation_gate_config,
    automation_gate_segments,
    scm_gate_config,
    scm_gate_segments,
)
from retail_simulator.envs.sweep.axes_loyalty import (
    loyalty_gate_config,
    loyalty_gate_segments,
)
from retail_simulator.envs.sweep.report import (
    assortment_axis_csv_rows,
    automation_axis_csv_rows,
    expansion_timing_csv_rows,
    format_sweep_table,
    loyalty_axis_csv_rows,
    promotion_axis_csv_rows,
    research_axis_csv_rows,
    scm_axis_csv_rows,
    sweep_csv_rows,
)
from retail_simulator.envs.sweep.rows import RESEARCH_OPPONENT_ARCHETYPE
from retail_simulator.envs.sweep.run import run_sweep
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

__all__ = [
    "RESEARCH_OPPONENT_ARCHETYPE",
    "assortment_axis_csv_rows",
    "assortment_verdict_for_axis",
    "automation_axis_csv_rows",
    "automation_gate_config",
    "automation_gate_segments",
    "automation_verdict_for_timing",
    "expansion_timing_csv_rows",
    "expansion_verdict_for_timing",
    "format_sweep_table",
    "loyalty_axis_csv_rows",
    "loyalty_gate_config",
    "loyalty_gate_segments",
    "loyalty_verdict_for_spend",
    "loyalty_vs_promotion_orthogonality_verdict",
    "promotion_axis_csv_rows",
    "promotion_verdict_for_axis",
    "research_axis_csv_rows",
    "research_gate_segments",
    "research_opponent_config",
    "research_verdict_for_fidelity",
    "run_sweep",
    "scm_axis_csv_rows",
    "scm_gate_config",
    "scm_gate_segments",
    "scm_verdict_for_service",
    "sweep_csv_rows",
    "verdict_for_sweep",
]
