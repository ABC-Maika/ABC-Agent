"""Hot-seat human-vs-archetype game: observation decoder, rival-impression fold,
posture/confidence mapping, and the interactive layer (board, prompt, driver, reveal,
debrief). The rival is an NPC seat, so the human only ever sees its perceived (noised)
state; env imports are lazy so module load stays cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import numpy.typing as npt

from retail_simulator.core.schema import OBSERVATION_SCHEMA

# --- Observation decoder ---
OBS_INDEX: dict[str, int] = {f.name: i for i, f in enumerate(OBSERVATION_SCHEMA)}


def obs_field(obs: npt.NDArray[np.float32], name: str) -> float:
    """Read one named scalar from a flat observation vector; KeyError on an unknown field name."""
    return float(obs[OBS_INDEX[name]])


# Rival scoreboard fields shown on the board: human-facing label -> perceived obs field
# name ("marketing" reads the seated competitor_awareness stock; there is no
# competitor_marketing field).
BOARD_RIVAL_FIELDS: dict[str, str] = {
    "price": "competitor_price",
    "marketing": "competitor_awareness",
    "promotion": "competitor_promotion",
    "share": "competitor_market_share",
}

# Own (exact, never noised) scoreboard fields, read by name through OBS_INDEX.
SELF_LAST_PROFIT_FIELD: str = "last_profit"
SELF_MARKET_SHARE_FIELD: str = "market_share"


# --- Running-impression fold --------------------------------------------------------


@dataclass
class RunningImpression:
    """Cumulative per-field mean of the perceived (noised) rival scoreboard over one episode.

    Fold post-step observations only: the reset obs is seeded with true values and would
    leak. update() reads only the perceived fields in BOARD_RIVAL_FIELDS, never
    un-noised state.
    """

    # Per-field sums and the shared fold count.
    _sums: dict[str, float] = field(default_factory=lambda: dict.fromkeys(BOARD_RIVAL_FIELDS, 0.0))
    _n: int = 0

    @property
    def n_ticks(self) -> int:
        """Number of post-step observations folded since construction/reset."""
        return self._n

    def update(self, obs: npt.NDArray[np.float32]) -> None:
        """Fold one post-step observation's perceived rival values into the running means.

        Never call with the reset observation (it is seeded with true values — a leak).
        """
        for label, field_name in BOARD_RIVAL_FIELDS.items():
            self._sums[label] += obs_field(obs, field_name)
        self._n += 1

    def means(self) -> dict[str, float]:
        """Per-field running means keyed by human-facing label; all-nan before the first fold."""
        if self._n == 0:
            return dict.fromkeys(BOARD_RIVAL_FIELDS, float("nan"))
        return {label: total / self._n for label, total in self._sums.items()}


# --- Categorical mapping + confidence -----------------------------------------------


class PostureBucket(Enum):
    """A coarse categorical posture label for one rival scoreboard field."""

    LOW = "LOW"
    MED = "MED"
    HIGH = "HIGH"


@dataclass(frozen=True)
class FieldThresholds:
    """LOW|MED|HIGH cut points for one rival field: LOW at <= low_max, HIGH at >= high_min,
    MED between.
    """

    low_max: float
    high_min: float


# Per-field LOW|MED|HIGH cut points, tuned on real perceived rollout obs so the three
# archetypes land in distinct buckets on each field.
RIVAL_FIELD_THRESHOLDS: dict[str, FieldThresholds] = {
    "price": FieldThresholds(low_max=0.918, high_min=1.255),
    "marketing": FieldThresholds(low_max=0.728, high_min=1.819),
    "promotion": FieldThresholds(low_max=0.184, high_min=0.452),
    "share": FieldThresholds(low_max=0.551, high_min=0.756),
}


def categorize_rival_field(label: str, value: float) -> PostureBucket:
    """Map a running-mean rival scalar to LOW | MED | HIGH via RIVAL_FIELD_THRESHOLDS;
    nan maps to MED.
    """
    cuts = RIVAL_FIELD_THRESHOLDS[label]
    if np.isnan(value):
        return PostureBucket.MED
    if value <= cuts.low_max:
        return PostureBucket.LOW
    if value >= cuts.high_min:
        return PostureBucket.HIGH
    return PostureBucket.MED


# Confidence sigils by fold count: forming below 2 folds, tentative through 4, settled at 5+.
CONFIDENCE_FORMING: str = "??"
CONFIDENCE_TENTATIVE: str = "~"
CONFIDENCE_SETTLED: str = ">>"


def confidence_sigil(n_ticks: int) -> str:
    """Map a fold count to a confidence sigil: ?? (n<2), ~ (2<=n<5), >> (n>=5).

    Field-agnostic schedule; negative n_ticks reads as forming (display-only).
    """
    if n_ticks < 2:
        return CONFIDENCE_FORMING
    if n_ticks < 5:
        return CONFIDENCE_TENTATIVE
    return CONFIDENCE_SETTLED


# Fields whose bucket reads are less reliable; their confidence caps at tentative.
SOFT_RIVAL_FIELDS: frozenset[str] = frozenset({"promotion", "share"})


def field_confidence_sigil(field: str, n_ticks: int) -> str:
    """Per-field confidence sigil: SOFT_RIVAL_FIELDS cap at tentative, others follow
    confidence_sigil.

    Raises KeyError for an unknown field name.
    """
    if field not in BOARD_RIVAL_FIELDS:
        raise KeyError(field)
    sigil = confidence_sigil(n_ticks)
    if field in SOFT_RIVAL_FIELDS and sigil == CONFIDENCE_SETTLED:
        return CONFIDENCE_TENTATIVE
    return sigil


# --- Interactive game layer (board, prompt, driver, debrief) ---
# env/pettingzoo imports are lazy (inside the driver) so importing this module stays cheap.

import json as _json  # noqa: E402
from collections.abc import Callable, Mapping  # noqa: E402
from itertools import zip_longest  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import TYPE_CHECKING, Any  # noqa: E402

from retail_simulator.core.encoding import (  # noqa: E402
    StructuredAction,
    decode_action,
    encode_action,
)
from retail_simulator.core.schema import LEVER_REGISTRY, LeverKind  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as _npt


PLAYER_LEVERS: tuple[str, ...] = (
    "price_index",
    "marketing",
    "assortment",
    "service_level",
    "expansion",
    "promotion",
    "research",
    "wage_spend",
    "warehouse_invest",
)

# Registry-default lever values; empty input holds every lever here ("hold course").
_REGISTRY_DEFAULTS: dict[str, float] = {spec.name: spec.default for spec in LEVER_REGISTRY}

# Playstyle labels as plain strings so the archetype module is imported lazily, in the
# driver. Counter-cycle: LEAN beats AGGRESSOR beats PATIENT beats LEAN.
PLAYSTYLES: tuple[str, ...] = ("lean", "aggressor", "patient")

COUNTERED_BY: dict[str, str] = {
    "aggressor": "lean",
    "patient": "aggressor",
    "lean": "patient",
}

# A one-line "why" for each counter edge — the teaching feedback shown at the reveal.
_COUNTER_RATIONALE: dict[str, str] = {
    "aggressor": (
        "LEAN counters AGGRESSOR: the aggressor over-extends on cash-hungry growth "
        "(expansion + heavy marketing/promotion), and the lean defender's fat margin "
        "out-survives the burn."
    ),
    "patient": (
        "AGGRESSOR counters PATIENT: the patient re-investor builds too slowly, so the aggressor's early aggressive expansion takes share the patient never recovers."
    ),
    "lean": (
        "PATIENT counters LEAN: the lean defender deliberately under-invests, and the "
        "patient re-investor compounds past it over the full horizon."
    ),
}


# --- Per-turn record (the debrief row) ----------------------------------------------


@dataclass
class TurnRecord:
    """One tick's logged state for the debrief; deterministic, no wall-clock.

    Holds the chosen levers, the tick's raw and mean perceived-rival readings, the
    human's own outcome, and the expansion mask faced (so illegal-move rejections
    reproduce).
    """

    tick: int
    human_levers: dict[str, float]
    perceived_rival_raw: dict[str, float]
    perceived_rival_mean: dict[str, float]
    own_market_share: float
    own_last_profit: float
    expansion_mask: list[bool]

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON view (lists/floats only) for the ``.jsonl`` writer."""
        return {
            "tick": self.tick,
            "human_levers": {k: float(v) for k, v in self.human_levers.items()},
            "perceived_rival_raw": {k: float(v) for k, v in self.perceived_rival_raw.items()},
            "perceived_rival_mean": {k: float(v) for k, v in self.perceived_rival_mean.items()},
            "own_market_share": float(self.own_market_share),
            "own_last_profit": float(self.own_last_profit),
            "expansion_mask": [bool(b) for b in self.expansion_mask],
        }


@dataclass
class GameResult:
    """Full record of one finished hot-seat game; the debrief writer's input.

    Deterministic (no timestamp). ``won`` is decided on the cumulative weighted_objective
    scores; final cash is a secondary record only.
    """

    seed: int
    n_ticks: int
    archetype_actual: str
    turns: list[TurnRecord]
    guess: str
    counter_guess: str
    correct: bool
    won: bool
    # Verdict metric: cumulative weighted_objective per seat over the episode.
    own_final_score: float
    rival_final_score: float
    # Secondary record (not the verdict): final seated cash per seat.
    own_final_cash: float
    rival_final_cash: float

    def to_dict(self) -> dict[str, Any]:
        """The single JSON object appended as one ``.jsonl`` line. NO wall-clock."""
        return {
            "seed": self.seed,
            "n_ticks": self.n_ticks,
            "archetype_actual": self.archetype_actual,
            "turns": [t.to_dict() for t in self.turns],
            "final": {
                "guess": self.guess,
                "counter_guess": self.counter_guess,
                "archetype_actual": self.archetype_actual,
                "correct": self.correct,
                "won": self.won,
                # Verdict metric and secondary cash record.
                "own_final_score": float(self.own_final_score),
                "rival_final_score": float(self.rival_final_score),
                "own_final_cash": float(self.own_final_cash),
                "rival_final_cash": float(self.rival_final_cash),
            },
        }


# --- The perceivable-rival NPC adapter ----------------------------------------------


class _ArchetypeNPCPolicy:
    """Wrap an archetype predict closure as an NPC-seat policy (open-loop fixed posture).

    The rival must be an NPC seat so the seam noises its state into the perceived
    competitor obs block (an all-learning duopoly would perceive an all-zero rival).
    Deterministic: no rng draw.
    """

    def __init__(self, predict: Callable[..., "_npt.NDArray[np.float32]"]) -> None:
        self._predict = predict

    def act(self, state: object, rng: object, seat_index: int = 1) -> "_npt.NDArray[np.float32]":
        """Return the archetype's fixed open-loop action (ignores state/rng/seat)."""
        del state, rng, seat_index
        return self._predict(None, episode_start=False)


# --- Human input → flat action ------------------------------------------------------


def hold_course_action() -> "_npt.NDArray[np.float32]":
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
    obs: "_npt.NDArray[np.float32]",
    impression: RunningImpression,
    info: Mapping[str, Any],
    *,
    tick: int,
    n_ticks: int,
    n_regions: int,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    prior_profit: float | None = None,
) -> "_npt.NDArray[np.float32]":
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
) -> "_npt.NDArray[np.float32]":
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


# --- Board renderer ------------------------------------------------------------------

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


def _rival_presence_label(obs: "_npt.NDArray[np.float32]") -> str:
    """Per-region rival presence derived from perceived per-region share > 0 (there is no
    competitor-presence obs field).
    """
    share_0 = obs_field(obs, "competitor_market_share")
    share_1 = obs_field(obs, "competitor_market_share_region_1")
    regions = [str(r) for r, share in enumerate((share_0, share_1)) if share > 0.0]
    return "+".join(f"R{r}" for r in regions) if regions else "(none seen)"


def render_board(
    obs: "_npt.NDArray[np.float32]",
    impression: RunningImpression,
    *,
    tick: int,
    n_ticks: int,
) -> str:
    """Two-panel ~80-col board with no profit trend — see :func:`render_board_with_delta`."""
    return render_board_with_delta(obs, impression, tick=tick, n_ticks=n_ticks, prior_profit=None)


def render_board_with_delta(
    obs: "_npt.NDArray[np.float32]",
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


# --- Turn-loop driver ----------------------------------------------------------------


def _pick_hidden_archetype(seed: int) -> str:
    """Pick the HIDDEN opponent playstyle from a SEEDED RNG (reproducible per seed)."""
    rng = np.random.default_rng(seed)
    return str(rng.choice(PLAYSTYLES))


def print_rps_primer(output_fn: Callable[[str], None]) -> None:
    """Teach the three playstyles + the abstract counter-cycle (NO opponent reference).

    The first-time primer the CLI shows before the game starts: it teaches what-beats-what
    ABSTRACTLY (no reference to the current opponent), so the human learns the cycle
    without being told which archetype they face.
    """
    output_fn(
        "\n".join(
            [
                "=" * _BOARD_WIDTH,
                "HOT-SEAT — know your opponents (rock / paper / scissors)".center(_BOARD_WIDTH),
                "=" * _BOARD_WIDTH,
                "Three playstyles, each beaten by exactly one other:",
                "",
                "  LEAN      defend margin: high price, minimal spend, no growth.",
                "  AGGRESSOR max growth: cheap price, heavy marketing/promotion, expand fast.",
                "  PATIENT   measured re-investor: moderate everything, build steadily.",
                "",
                "  LEAN beats AGGRESSOR   (fat margin out-survives the burn)",
                "  AGGRESSOR beats PATIENT (early aggressive expansion takes share)",
                "  PATIENT beats LEAN      (compounds past the under-investor)",
                "",
                "Watch the RIVAL panel, infer the playstyle, and play its counter.",
                "=" * _BOARD_WIDTH,
            ]
        )
    )


def play_hotseat_game(
    *,
    config: Any,
    seed: int,
    n_ticks: int,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    archetype: str | None = None,
    action_fn: Callable[[Mapping[str, Any]], "_npt.NDArray[np.float32]"] | None = None,
) -> GameResult:
    """Run one hot-seat game: human at seat 0, the archetype NPC at seat 1, for n_ticks.

    ``archetype`` forces the playstyle (default: a seeded pick); ``action_fn`` scripts
    seat-0's action from its info instead of prompting.
    """
    from retail_simulator.core.reward import weighted_objective
    from retail_simulator.envs.parallel_env import RetailParallelEnv
    from retail_simulator.harness.parallel_gate import _components_from_info
    from retail_simulator.harness.spike_archetypes import calibrated_archetypes

    chosen_archetype = archetype if archetype is not None else _pick_hidden_archetype(seed)
    if chosen_archetype not in PLAYSTYLES:
        raise ValueError(f"unknown archetype {chosen_archetype!r}; known: {PLAYSTYLES}")
    predict = calibrated_archetypes()[chosen_archetype]

    # Seat 0 is the human (learning); seat 1 is the NPC rival. The placeholder archetype
    # only shapes the seat plan; the policy is replaced with the chosen archetype.
    env = RetailParallelEnv(config, n_learning_agents=1, npc_archetypes=("discounter",))
    env._world._npc_policies[1] = _ArchetypeNPCPolicy(predict)  # noqa: SLF001

    observations, infos = env.reset(seed=seed)
    # The seam's expansion mask is fixed-capacity (N_REGIONS_MAX wide); read this world's
    # real region count once and trim the mask to it everywhere.
    reset_state = env._state  # noqa: SLF001
    assert reset_state is not None
    n_regions = len(reset_state.regions)
    impression = RunningImpression()
    turns: list[TurnRecord] = []
    prior_profit: float | None = None
    reward_cfg = config.reward
    own_score = 0.0
    rival_score = 0.0

    for tick in range(1, n_ticks + 1):
        obs = observations["retailer_0"]
        info = infos["retailer_0"]
        if action_fn is None:
            # Interactive path: render the board and prompt the human's levers.
            action = prompt_human_action(
                obs,
                impression,
                info,
                tick=tick,
                n_ticks=n_ticks,
                n_regions=n_regions,
                input_fn=input_fn,
                output_fn=output_fn,
                prior_profit=prior_profit,
            )
        else:
            # Scripted path: action_fn supplies seat-0's action from its info; no board.
            action = action_fn(info)
        prior_profit = obs_field(obs, SELF_LAST_PROFIT_FIELD)

        observations, _rewards, _term, _trunc, infos = env.step({"retailer_0": action})

        # Score both seats on this tick's stationary weighted_objective: seat 0 from its
        # public info, seat 1 from the seam's raw stash.
        own_score += weighted_objective(_components_from_info(infos["retailer_0"]), reward_cfg)
        raw_infos = env._last_raw_infos  # noqa: SLF001
        assert raw_infos is not None  # set by env.step we just called
        rival_score += weighted_objective(_components_from_info(raw_infos[1]), reward_cfg)

        # Fold the post-step obs only (never the reset obs — it is seeded with true values).
        post_obs = observations["retailer_0"]
        impression.update(post_obs)
        turns.append(
            _capture_turn(
                tick, action, post_obs, impression, info["action_mask"]["expansion"][:n_regions]
            )
        )

    guess, counter_guess = _prompt_end_game_guess(input_fn=input_fn, output_fn=output_fn)
    # Final seated cash is a secondary record only; the verdict is the cumulative score
    # accumulated above.
    final_state = env._state
    assert final_state is not None  # n_ticks >= 1 ⇒ at least one step ran after reset
    own_cash = float(final_state.retailers[0].cash)
    rival_cash = float(final_state.retailers[1].cash)
    won = own_score > rival_score
    correct = guess == chosen_archetype
    _print_reveal(
        chosen_archetype,
        guess=guess,
        counter_guess=counter_guess,
        correct=correct,
        won=won,
        own_score=own_score,
        rival_score=rival_score,
        own_cash=own_cash,
        rival_cash=rival_cash,
        output_fn=output_fn,
    )
    return GameResult(
        seed=seed,
        n_ticks=n_ticks,
        archetype_actual=chosen_archetype,
        turns=turns,
        guess=guess,
        counter_guess=counter_guess,
        correct=correct,
        won=won,
        own_final_score=own_score,
        rival_final_score=rival_score,
        own_final_cash=own_cash,
        rival_final_cash=rival_cash,
    )


def _capture_turn(
    tick: int,
    action: "_npt.NDArray[np.float32]",
    post_obs: "_npt.NDArray[np.float32]",
    impression: RunningImpression,
    expansion_mask: Any,
) -> TurnRecord:
    """Build the per-tick debrief record from the action + post-step obs + impression."""
    levers = decode_action(action).levers
    human_levers = {lever: float(levers[lever]) for lever in PLAYER_LEVERS}
    raw = {
        label: obs_field(post_obs, field_name) for label, field_name in BOARD_RIVAL_FIELDS.items()
    }
    return TurnRecord(
        tick=tick,
        human_levers=human_levers,
        perceived_rival_raw=raw,
        perceived_rival_mean=impression.means(),
        own_market_share=obs_field(post_obs, SELF_MARKET_SHARE_FIELD),
        own_last_profit=obs_field(post_obs, SELF_LAST_PROFIT_FIELD),
        expansion_mask=[bool(b) for b in expansion_mask],
    )


def _prompt_playstyle(
    prompt: str, *, input_fn: Callable[[str], str], output_fn: Callable[[str], None]
) -> str:
    """Prompt for one playstyle label until a valid one (or empty → first) is given."""
    while True:
        raw = input_fn(prompt).strip().lower()
        if raw == "":
            return PLAYSTYLES[0]
        if raw in PLAYSTYLES:
            return raw
        output_fn(f"  ! pick one of {', '.join(PLAYSTYLES)} — try again.")


def _prompt_end_game_guess(
    *, input_fn: Callable[[str], str], output_fn: Callable[[str], None]
) -> tuple[str, str]:
    """Prompt the human for (opponent-playstyle guess, the playstyle they think counters it)."""
    output_fn("\nGame over. Time to call it:")
    guess = _prompt_playstyle(
        f"  Which playstyle did you face? ({'/'.join(PLAYSTYLES)}): ",
        input_fn=input_fn,
        output_fn=output_fn,
    )
    counter_guess = _prompt_playstyle(
        f"  Which playstyle counters it? ({'/'.join(PLAYSTYLES)}): ",
        input_fn=input_fn,
        output_fn=output_fn,
    )
    return guess, counter_guess


def _print_reveal(
    archetype_actual: str,
    *,
    guess: str,
    counter_guess: str,
    correct: bool,
    won: bool,
    own_score: float,
    rival_score: float,
    own_cash: float,
    rival_cash: float,
    output_fn: Callable[[str], None],
) -> None:
    """Reveal the archetype, win/loss, and counter rationale.

    The score line reports the cumulative weighted_objective (the verdict metric); final
    cash is a secondary line.
    """
    true_counter = COUNTERED_BY[archetype_actual]
    output_fn(
        "\n".join(
            [
                "=" * _BOARD_WIDTH,
                "REVEAL".center(_BOARD_WIDTH),
                "=" * _BOARD_WIDTH,
                f"  You faced       : {archetype_actual.upper()}",
                f"  Your guess      : {guess.upper()}  ({'correct' if correct else 'wrong'})",
                f"  It is countered : {true_counter.upper()}  "
                f"(you said {counter_guess.upper()}: "
                f"{'right' if counter_guess == true_counter else 'not quite'})",
                f"  Score           : you {own_score:,.1f}  vs  rival {rival_score:,.1f}",
                f"  (final cash      : you {own_cash:,.0f}  vs  rival {rival_cash:,.0f})",
                f"  Result          : {'YOU WIN' if won else 'YOU LOSE'}",
                "",
                f"  {_COUNTER_RATIONALE[archetype_actual]}",
                "=" * _BOARD_WIDTH,
            ]
        )
    )


# --- Debrief writer ------------------------------------------------------------------


def append_debrief(result: GameResult, path: Path) -> None:
    """Append the game record as one .jsonl line (deterministic — no wall-clock).

    Re-running the same seed with the same inputs writes a byte-identical line
    (sort_keys keeps key order stable).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _json.dumps(result.to_dict(), sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
