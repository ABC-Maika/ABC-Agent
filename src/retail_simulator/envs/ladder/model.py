"""Data models, constants, and spec parsing for the margin-based competition ladder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from retail_simulator.envs.geo_archetypes import GEO_ARCHETYPE_LABELS
from retail_simulator.envs.spike_archetypes import AGGRESSOR, LEAN, PATIENT

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
]

DEFAULT_ENV_LABEL = "default (competition)"
CALIBRATED_ENV_LABEL = "redesign (phase-4 calibrated)"

# Spec grammar: a bare string is a checkpoint zip path; these prefixes select
# a scripted or random participant instead.
ARCHETYPE_SPEC_PREFIX = "archetype:"
RANDOM_SPEC_PREFIX = "random:"

# The open-loop spike family's labels, kept as their own constant so callers
# needing only that family don't subtract geo labels from the union below.
LADDER_SPIKE_ARCHETYPE_LABELS = (LEAN, AGGRESSOR, PATIENT)

# The names ``archetype:<name>`` accepts: the spike family plus the closed-loop
# geo archetypes (imported, not re-listed, so the two can't drift). Callers that
# mean the spike family specifically use LADDER_SPIKE_ARCHETYPE_LABELS.
LADDER_ARCHETYPE_LABELS = LADDER_SPIKE_ARCHETYPE_LABELS + GEO_ARCHETYPE_LABELS

ParticipantKind = Literal["checkpoint", "archetype", "random"]


@dataclass(frozen=True)
class PairwiseMargin:
    """One pair's score margin, bootstrap CI, and optional per-component breakdown."""

    label_a: str
    label_b: str
    per_seed_margins: tuple[float, ...]
    mean_margin: float
    margin_se: float
    ci_low: float | None = None
    ci_high: float | None = None
    component_mean_margins: dict[str, float] | None = None
    component_ci: dict[str, tuple[float, float]] | None = None
    component_per_seed_margins: dict[str, tuple[float, ...]] | None = None


@dataclass(frozen=True)
class LadderEntry:
    """One ranked participant: field margin, bootstrap CI, and dense rank.

    ``checkpoint_path`` carries the spec string verbatim for non-checkpoint
    participants. ``component_field_margins`` mirrors ``field_margin`` per
    component name, ``None`` when any pair lacks component data. The trailing
    defaults keep serialized entries byte-identical.
    """

    label: str
    checkpoint_path: str
    field_margin: float
    ci_low: float
    ci_high: float
    rank: int
    kind: ParticipantKind = "checkpoint"
    component_field_margins: dict[str, float] | None = None


@dataclass(frozen=True)
class TournamentSummary:
    """Dominance, cycle, and Condorcet read of the field's own score margins.

    ``dominance`` maps each label to the labels it beats (mean_margin > 0);
    ``n_cyclic_triples`` counts cyclic 3-subsets of the C(n, 3) total.
    ``decisive_cycle`` is True iff the FIRST cyclic triple's edges all have
    bootstrap CIs excluding 0; ``decisive_triples`` lists every such triple in
    cycle order, smallest label first.
    ``checkpoint_inclusive_decisive_triples`` counts entries touching a
    ``"checkpoint"`` participant. The trailing defaults keep serialized
    artifacts byte-identical.
    """

    dominance: dict[str, tuple[str, ...]]
    n_cyclic_triples: int
    total_triples: int
    has_condorcet_winner: bool
    condorcet_winner: str | None
    decisive_cycle: bool
    decisive_triples: tuple[tuple[str, str, str], ...] = ()
    checkpoint_inclusive_decisive_triples: int = 0


@dataclass(frozen=True)
class LadderResult:
    """A ladder run: ranked entries, pairwise margins, and tournament summary."""

    ladder_id: str
    n_seeds: int
    seed_set: tuple[int, ...]
    n_episodes: int
    entries: tuple[LadderEntry, ...]
    pairwise: tuple[PairwiseMargin, ...]
    kendall_tau_split_half: float | None
    bootstrap_resamples: int
    bootstrap_seed: int
    config_label: str = DEFAULT_ENV_LABEL
    tournament: TournamentSummary | None = None
    home_regions: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ParticipantSpec:
    """A parsed participant spec: kind, canonical source string, and default label.

    ``source`` is the canonical spelling — random specs re-serialize as
    ``f"random:{seed}"`` so equivalent inputs hash to the same ladder_id.
    ``seed`` is the parsed integer for ``kind == "random"`` (else ``None``).
    """

    kind: ParticipantKind
    source: str
    default_label: str
    seed: int | None = None


def parse_participant_spec(spec: str) -> ParticipantSpec:
    """Parse a participant spec into kind and default label (no filesystem checks).

    ``"archetype:<name>"`` must name one of LADDER_ARCHETYPE_LABELS (the default
    label is the name); ``"random:<seed>"`` takes a non-negative integer seed and
    re-serializes to ``f"random:{seed}"``; anything else is a checkpoint zip path.
    Raises ValueError on an unknown archetype or a malformed/negative seed.
    """
    if spec.startswith(ARCHETYPE_SPEC_PREFIX):
        name = spec[len(ARCHETYPE_SPEC_PREFIX) :]
        if name not in LADDER_ARCHETYPE_LABELS:
            raise ValueError(
                f"unknown archetype {name!r} in {spec!r}; must be one of "
                f"{LADDER_ARCHETYPE_LABELS}"
            )
        return ParticipantSpec(kind="archetype", source=spec, default_label=name)
    if spec.startswith(RANDOM_SPEC_PREFIX):
        seed_str = spec[len(RANDOM_SPEC_PREFIX) :]
        try:
            seed = int(seed_str)
        except ValueError as exc:
            raise ValueError(
                f"random participant seed must be an integer; got {seed_str!r} in {spec!r}"
            ) from exc
        if seed < 0:
            raise ValueError(f"random participant seed must be >= 0; got {seed} in {spec!r}")
        return ParticipantSpec(
            kind="random",
            source=f"{RANDOM_SPEC_PREFIX}{seed}",
            default_label=f"random_{seed}",
            seed=seed,
        )
    path = Path(spec)
    parent_name = path.parent.name
    default_label = f"{parent_name}/{path.stem}" if parent_name else path.stem
    return ParticipantSpec(kind="checkpoint", source=spec, default_label=default_label)
