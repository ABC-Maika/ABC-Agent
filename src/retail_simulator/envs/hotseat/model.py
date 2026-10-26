"""Hot-seat game model: observation decoder, rival-impression fold, posture/confidence
mapping, per-turn records, and the NPC adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from retail_simulator.core.schema import OBSERVATION_SCHEMA

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as _npt


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


# --- Running-impression fold ---


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


# --- Categorical mapping + confidence ---


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


# --- Game constants ---

# Player levers in display order.
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


# --- Per-turn record ---


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


# --- The perceivable-rival NPC adapter ---


class _ArchetypeNPCPolicy:
    """Wrap an archetype predict closure as an NPC-seat policy (open-loop fixed posture).

    The rival must be an NPC seat so the seam noises its state into the perceived
    competitor obs block (an all-learning duopoly would perceive an all-zero rival).
    Deterministic: no rng draw.
    """

    def __init__(self, predict: Callable[..., _npt.NDArray[np.float32]]) -> None:
        self._predict = predict

    def act(self, state: object, rng: object, seat_index: int = 1) -> _npt.NDArray[np.float32]:
        """Return the archetype's fixed open-loop action (ignores state/rng/seat)."""
        del state, rng, seat_index
        return self._predict(None, episode_start=False)
