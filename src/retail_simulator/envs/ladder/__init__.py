"""Margin-based competition ladder: pairwise matches ranked by field margin with bootstrap CIs."""

from __future__ import annotations

from .model import (
    ARCHETYPE_SPEC_PREFIX,
    CALIBRATED_ENV_LABEL,
    DEFAULT_ENV_LABEL,
    LADDER_ARCHETYPE_LABELS,
    LADDER_SPIKE_ARCHETYPE_LABELS,
    RANDOM_SPEC_PREFIX,
    LadderEntry,
    LadderResult,
    PairwiseMargin,
    ParticipantKind,
    ParticipantSpec,
    TournamentSummary,
    parse_participant_spec,
)
from .policy import (
    _materialize_predicts as _materialize_predicts,  # private re-export: arena_match, phase4_rl_confirm
)
from .runner import run_ladder
from .stats import (
    _bootstrap_margin_ci as _bootstrap_margin_ci,  # private re-export: spike, phase4_rl_confirm
)

__all__ = [
    "ARCHETYPE_SPEC_PREFIX",
    "CALIBRATED_ENV_LABEL",
    "DEFAULT_ENV_LABEL",
    "LADDER_ARCHETYPE_LABELS",
    "LADDER_SPIKE_ARCHETYPE_LABELS",
    "RANDOM_SPEC_PREFIX",
    "LadderEntry",
    "LadderResult",
    "PairwiseMargin",
    "ParticipantKind",
    "ParticipantSpec",
    "TournamentSummary",
    "parse_participant_spec",
    "run_ladder",
]
