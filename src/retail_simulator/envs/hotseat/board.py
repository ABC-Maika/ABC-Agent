"""Hot-seat game board: human input prompting, lever validation, and the two-panel
scoreboard renderer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from itertools import zip_longest
from typing import TYPE_CHECKING, Any

import numpy as np

from retail_simulator.core.encoding import StructuredAction, encode_action
from retail_simulator.core.schema import LEVER_REGISTRY, LeverKind
from retail_simulator.envs.hotseat.model import (
    BOARD_RIVAL_FIELDS,
    PLAYER_LEVERS,
    SELF_LAST_PROFIT_FIELD,
    SELF_MARKET_SHARE_FIELD,
    RunningImpression,
    categorize_rival_field,
    field_confidence_sigil,
    obs_field,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as _npt


# Registry-default lever values; empty input holds every lever here ("hold course").
_REGISTRY_DEFAULTS: dict[str, float] = {spec.name: spec.default for spec in LEVER_REGISTRY}


# --- Human input → flat action ---


def hold_course_action() -> _npt.NDArray[np.float32]:
    """Flat action holding every lever at its registry default (the hold-course baseline)."""
    return encode_action(StructuredAction(levers=dict(_REGISTRY_DEFAULTS)))


def _parse_lever_input(raw: str, *, lever: str, default: float) -> tuple[bool, float, str]:
    """Validate one prompted lever value; returns (ok, value, error_message).

    Empty input accepts the default; otherwise the value is bound-checked against the
    lever's schema range (continuous) or integer choice set (expansion).
    """
    text = raw.strip()
    if text == "":
        return True, default, ""
    try:
        value = float(text)
    except ValueError:
        return False, default, f"  ! {lever}: '{raw.strip()}' is not a number — try again."

    spec = next(s for s in LEVER_REGISTRY if s.name == lever)
    if spec.kind is LeverKind.CONTINUOUS:
        assert spec.low is not None and spec.high is not None
        if not (spec.low <= value <= spec.high):
            return (
                False,
                default,
                f"  ! {lever}: {value:g} is outside [{spec.low:g}, {spec.high:g}] — try again.",
            )
        return True, value, ""
    # Discrete lever (expansion): integer choice in [0, n).
    assert spec.n is not None
    choice = int(value)
    if choice != value or not (0 <= choice < spec.n):
        return (
            False,
            default,
            f"  ! {lever}: choice must be an integer in [0, {spec.n}) — try again.",
        )
    return True, float(choice), ""


def prompt_human_action(
    obs: _npt.NDArray[np.float32],
    impression: RunningImpression,
    info: Mapping[str, Any],
    *,
    tick: int,
    n_ticks: int,
    n_regions: int,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    prior_profit: float | None = None,
) -> _npt.NDArray[np.float32]:
    """Render the board, prompt the nine player levers, and build the flat action.

    ``n_regions`` is this world's real region count (the seam's expansion mask is
    fixed-capacity). ``input_fn``/``output_fn`` are injected so tests can script a full
    game without a terminal.
    """
    output_fn(
        render_board_with_delta(
            obs, impression, tick=tick, n_ticks=n_ticks, prior_profit=prior_profit
        )
    )
    return _collect_levers(info, n_regions=n_regions, input_fn=input_fn, output_fn=output_fn)


def _collect_levers(
    info: Mapping[str, Any],
    *,
    n_regions: int,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
) -> _npt.NDArray[np.float32]:
    """Prompt and validate the nine player levers (board already rendered); build the
    flat action.

    Empty input holds the default; bad input re-prompts that lever. Expansion is checked
    twice: trimmed to ``n_regions`` first (the seam's mask is fixed-capacity), then
    against the cash gate. Non-exposed levers stay at their defaults.
    """
    expansion_mask = list(info["action_mask"]["expansion"])[:n_regions]
    chosen: dict[str, float] = dict(_REGISTRY_DEFAULTS)
    for lever in PLAYER_LEVERS:
        default = _REGISTRY_DEFAULTS[lever]
        while True:
            raw = input_fn(f"  {lever} [{default:g}]: ")
            ok, value, error = _parse_lever_input(raw, lever=lever, default=default)
            if not ok:
                output_fn(error)
                continue
            if lever == "expansion":
                choice = int(value)
                if choice >= len(expansion_mask):
                    output_fn(
                        f"  ! expansion: region {choice} does not exist in this world "
                        f"(it has {len(expansion_mask)} region(s)) — pick an allowed "
                        "choice."
                    )
                    continue
                if not expansion_mask[choice]:
                    output_fn(
                        f"  ! expansion: choice {choice} is cash-gated (the seam forbids "
                        "it this tick) — pick an allowed choice."
                    )
                    continue
            chosen[lever] = value
            break
    return encode_action(StructuredAction(levers=chosen))


# --- Board renderer ---

_BOARD_WIDTH: int = 78
# Own-panel label column width (fits "staff happiness", the longest label).
_OWN_LABEL_WIDTH: int = 16


def _fmt_delta(delta: float) -> str:
    """A compact up/down/flat trend marker for the profit turn-over-turn delta."""
    if delta > 0:
        return f"^ +{delta:,.0f}"
    if delta < 0:
        return f"v {delta:,.0f}"
    return "= flat"


def _rival_presence_label(obs: _npt.NDArray[np.float32]) -> str:
    """Per-region rival presence derived from perceived per-region share > 0 (there is no
    competitor-presence obs field).
    """
    share_0 = obs_field(obs, "competitor_market_share")
    share_1 = obs_field(obs, "competitor_market_share_region_1")
    regions = [str(r) for r, share in enumerate((share_0, share_1)) if share > 0.0]
    return "+".join(f"R{r}" for r in regions) if regions else "(none seen)"


def render_board(
    obs: _npt.NDArray[np.float32],
    impression: RunningImpression,
    *,
    tick: int,
    n_ticks: int,
) -> str:
    """Two-panel ~80-col board with no profit trend — see :func:`render_board_with_delta`."""
    return render_board_with_delta(obs, impression, tick=tick, n_ticks=n_ticks, prior_profit=None)


def render_board_with_delta(
    obs: _npt.NDArray[np.float32],
    impression: RunningImpression,
    *,
    tick: int,
    n_ticks: int,
    prior_profit: float | None,
) -> str:
    """Two-panel ~80-col board: left is the own snapshot (exact obs), right the rival
    impression.

    Left reads own obs (share, profit with trend, cash, stores, fidelity, and the
    wage/warehouse readouts). Right reads only the perceived-rival running impression:
    per-field posture buckets with confidence sigils, plus presence. Never reads
    true_competitor_view and never prints the archetype label.
    """
    own_share = obs_field(obs, SELF_MARKET_SHARE_FIELD)
    own_profit = obs_field(obs, SELF_LAST_PROFIT_FIELD)
    own_cash = obs_field(obs, "cash")
    own_stores = obs_field(obs, "stores")
    own_fidelity = obs_field(obs, "research_fidelity")
    own_happiness = obs_field(obs, "employee_happiness")
    own_wage_level = obs_field(obs, "wage_level")
    own_warehouse_used = obs_field(obs, "warehouse_utilization")
    own_lost_sales = obs_field(obs, "last_stockout_rate")
    trend = "" if prior_profit is None else f"  ({_fmt_delta(own_profit - prior_profit)})"

    means = impression.means()
    rival_rows: list[tuple[str, str]] = []
    for label in BOARD_RIVAL_FIELDS:
        sigil = field_confidence_sigil(label, impression.n_ticks)
        bucket = categorize_rival_field(label, means[label]).value
        rival_rows.append((label, f"{sigil} {bucket}"))

    left = [
        "YOU (retailer_0)",
        f"  {'market share':<{_OWN_LABEL_WIDTH}}: {own_share:6.1%}",
        f"  {'last profit':<{_OWN_LABEL_WIDTH}}: {own_profit:>10,.0f}{trend}",
        f"  {'cash':<{_OWN_LABEL_WIDTH}}: {own_cash:>10,.0f}",
        f"  {'stores':<{_OWN_LABEL_WIDTH}}: {int(own_stores):>10d}",
        f"  {'fidelity':<{_OWN_LABEL_WIDTH}}: {own_fidelity:>10.0%}",
        f"  {'staff happiness':<{_OWN_LABEL_WIDTH}}: {own_happiness:>10.0%}",
        f"  {'staff pay':<{_OWN_LABEL_WIDTH}}: {own_wage_level:>10,.0f}",
        f"  {'warehouse used':<{_OWN_LABEL_WIDTH}}: {own_warehouse_used:>10.0%}",
        f"  {'lost sales':<{_OWN_LABEL_WIDTH}}: {own_lost_sales:>10.0%}",
    ]
    right = [
        "RIVAL (read of perceived signal)",
        *[f"  {label:<10}: {posture}" for label, posture in rival_rows],
        f"  presence  : {_rival_presence_label(obs)}",
    ]

    half = _BOARD_WIDTH // 2
    header = f"== TICK {tick}/{n_ticks} ".ljust(_BOARD_WIDTH, "=")
    body_lines = []
    for left_line, right_line in zip_longest(left, right, fillvalue=""):
        body_lines.append(f"{left_line:<{half}}{right_line}")
    return "\n".join([header, *body_lines, "=" * _BOARD_WIDTH])
