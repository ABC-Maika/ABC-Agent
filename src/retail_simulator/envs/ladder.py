"""Margin-based competition ladder.

Runs pairwise head-to-head matches between participants (checkpoint zips,
scripted archetypes, random baselines) and ranks them by field margin with
bootstrap CIs, a split-half Kendall tau diagnostic, and a dominance/cycle
summary.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np

from retail_simulator.core.config import CoreConfig
from retail_simulator.envs.geo_archetypes import GEO_ARCHETYPE_LABELS, geo_archetypes
from retail_simulator.envs.spike_archetypes import (
    AGGRESSOR,
    LEAN,
    PATIENT,
    ArchetypePredict,
    calibrated_archetypes,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ARCHETYPE_SPEC_PREFIX",
    "CALIBRATED_ENV_LABEL",
    "DEFAULT_ENV_LABEL",
    "LADDER_ARCHETYPE_LABELS",
    "LADDER_SPIKE_ARCHETYPE_LABELS",
    "LadderEntry",
    "LadderResult",
    "PairwiseMargin",
    "ParticipantKind",
    "ParticipantSpec",
    "RANDOM_SPEC_PREFIX",
    "TournamentSummary",
    "parse_participant_spec",
    "run_ladder",
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


def _kendall_tau(rank_a: Sequence[int], rank_b: Sequence[int]) -> float:
    """Kendall tau between two total orders (ties count as neither)."""
    if len(rank_a) != len(rank_b):
        raise ValueError(
            f"_kendall_tau requires equal-length rankings; "
            f"got len(rank_a)={len(rank_a)}, len(rank_b)={len(rank_b)}"
        )
    n = len(rank_a)
    if n < 2:
        # No pairs to compare; 1.0 keeps the diagnostic defined at the edge.
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_order = rank_a[i] - rank_a[j]
            b_order = rank_b[i] - rank_b[j]
            product = a_order * b_order
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            # product == 0: a tie, counted as neither (τ-a convention).
    total_pairs = n * (n - 1) // 2
    return float(concordant - discordant) / float(total_pairs)


def _field_margin_from_pair_means(
    label: str,
    others: Sequence[str],
    pair_mean: dict[tuple[str, str], float],
) -> float:
    """Mean of ``label``'s margin over every other participant in ``others``.

    ``pair_mean`` stores A-over-B per sorted pair; the margin is negated when
    ``label`` is the pair's second member.
    """
    margins: list[float] = []
    for other in others:
        if label < other:
            margins.append(pair_mean[(label, other)])
        else:
            margins.append(-pair_mean[(other, label)])
    return float(np.mean(margins))


def _bootstrap_margin_ci(
    label: str,
    others: Sequence[str],
    pair_per_seed: dict[tuple[str, str], np.ndarray],
    *,
    n_seeds: int,
    resampled_seed_indices: np.ndarray,
) -> tuple[float, float]:
    """Bootstrap 95% CI on ``label``'s field margin, resampling the seed axis.

    Each row of ``resampled_seed_indices`` (n_resamples, n_seeds) is applied to
    every pair consistently, so each iteration is one coherent surrogate seed
    set; the CI is the 2.5th/97.5th percentile of the resampled means.
    """
    # Orient each pairing label-over-other into a (n_others, n_seeds) matrix.
    oriented_rows: list[np.ndarray] = []
    for other in others:
        if label < other:
            oriented_rows.append(pair_per_seed[(label, other)])
        else:
            oriented_rows.append(-pair_per_seed[(other, label)])
    oriented = np.asarray(oriented_rows, dtype=np.float64)
    # Mean over the others axis is the field margin per seed.
    field_per_seed = oriented.mean(axis=0)  # shape (n_seeds,)
    # Average field-per-seed over each resampled seed row.
    boot_field = field_per_seed[resampled_seed_indices].mean(axis=1)  # (n_resamples,)
    ci_low = float(np.percentile(boot_field, 2.5))
    ci_high = float(np.percentile(boot_field, 97.5))
    return ci_low, ci_high


def _detect_policy_kind(checkpoint_path: str | Path) -> Literal["mlp", "lstm"]:
    """Peek a checkpoint zip and return "mlp" or "lstm" from its serialized policy."""
    try:
        with zipfile.ZipFile(checkpoint_path) as zf:
            with zf.open("data") as data_member:
                data = json.load(data_member)
        serialized = data["policy_class"][":serialized:"]
        decoded = base64.b64decode(serialized)
    except (
        zipfile.BadZipFile,
        KeyError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        # Peek failed: default to "mlp" and warn so a demoted LSTM stays visible.
        logger.warning(
            "policy-kind auto-detect peek failed for %s (%s: %s); defaulting to 'mlp'",
            checkpoint_path,
            type(exc).__name__,
            exc,
        )
        return "mlp"
    if b"recurrent" in decoded or b"sb3_contrib" in decoded:
        return "lstm"
    return "mlp"


def _make_random_predict(seed: int, action_space: Any) -> Callable[..., Any]:
    """Build a predict closure that re-seeds a deep-copied action space each episode."""
    space = copy.deepcopy(action_space)
    episode_index = 0
    space.seed(seed)

    def predict(_obs: object, *, episode_start: bool = False) -> Any:
        nonlocal episode_index
        if episode_start:
            space.seed(seed + 1_000_003 * episode_index)
            episode_index += 1
        return np.asarray(space.sample(), dtype=np.float32)

    return predict


def _materialize_predicts(
    spec_by_label: dict[str, str],
    *,
    action_space: Any | None = None,
    config: CoreConfig | None = None,
) -> dict[str, Callable[..., Any]]:
    """Build each participant's predict closure once, routed by its spec's kind.

    Checkpoints wrap a one-member FrozenPolicyPool (lazy SB3 import). Spike
    archetype names share one calibrated_archetypes() build; geo names share one
    geo_archetypes() build and need ``config``; random baselines need
    ``action_space``. The map is reused across every pair and seed, so each
    checkpoint loads once rather than per (pair, seed).
    """
    # Lazy import: keep module load cheap and sb3-free.
    from retail_simulator.envs.league import (  # noqa: PLC0415
        FrozenPolicyPool,
        PoolMember,
    )

    predicts: dict[str, Callable[..., Any]] = {}
    spike_archetype_map: dict[str, ArchetypePredict] | None = None
    geo_archetype_map: dict[str, Callable[..., Any]] | None = None
    for label, spec in spec_by_label.items():
        parsed = parse_participant_spec(spec)
        if parsed.kind == "archetype":
            if parsed.default_label in GEO_ARCHETYPE_LABELS:
                if geo_archetype_map is None:
                    if config is None:
                        raise ValueError(
                            f"participant {label!r} ({spec!r}) is a geo archetype but "
                            "no config was provided to materialize it (no region table "
                            "to build geo_archetypes() from)"
                        )
                    # One geo_archetypes() call per invocation, shared across geo
                    # participants; config.demand.regions is read only here, so a
                    # field with no geo participant never needs it.
                    region_table = config.demand.regions
                    geo_archetype_map = geo_archetypes(len(region_table), region_table)
                predicts[label] = geo_archetype_map[parsed.default_label]
            else:
                if spike_archetype_map is None:
                    # One calibrated_archetypes() call per invocation, shared
                    # across spike participants.
                    spike_archetype_map = calibrated_archetypes()
                predicts[label] = spike_archetype_map[parsed.default_label]
        elif parsed.kind == "random":
            if action_space is None:
                raise ValueError(
                    f"participant {label!r} ({spec!r}) is a random baseline but no "
                    "action_space was provided to materialize it"
                )
            # parse_participant_spec sets seed for "random" and only "random".
            assert parsed.seed is not None
            predicts[label] = _make_random_predict(parsed.seed, action_space)
        elif parsed.kind == "checkpoint":
            # materialize resolves checkpoint_path against pool_dir, so seat
            # pool_dir at the zip's parent and pass the bare filename.
            path = Path(spec)
            policy_kind = _detect_policy_kind(path)
            member = PoolMember(
                member_id=f"checkpoint:ladder_{label}",
                kind="checkpoint",
                archetype_name=f"frozen_policy:ladder_{label}",
                checkpoint_path=Path(path.name),
                round_index=0,
                policy_kind=policy_kind,
            )
            pool = FrozenPolicyPool(path.parent, members=(member,))
            predicts[label] = pool.materialize(member, env=None)
        else:
            # Unreachable; an explicit raise so a future kind can't fall through
            # into the checkpoint branch above.
            raise ValueError(f"participant {label!r} has unhandled kind {parsed.kind!r}")
    return predicts


def _default_pairwise_per_seed(
    sorted_labels: Sequence[str],
    spec_by_label: dict[str, str],
    *,
    seed_set: Sequence[int],
    n_episodes: int,
    config: CoreConfig | None = None,
    home_regions: tuple[int, ...] | None = None,
) -> tuple[
    dict[tuple[str, str], list[float]],
    dict[tuple[str, str], dict[str, tuple[float, ...]] | None],
]:
    """Default rollouts: build one shared env, materialize once, then loop pairs × seeds.

    ``config`` selects the env (``None`` falls back to
    ``parallel_gate.competition_config()``). The env is built before
    materializing so a materialize failure still reaches the ``finally`` and
    closes it; head-to-head re-seeds the world per call, so one shared env
    across every pair and seed is deterministic. Returns
    ``(per_seed_by_pair, component_per_seed_by_pair)``; a pair's component dict
    is ``None`` when its rollout primitive never filled the component sink.
    ``home_regions`` forwards to the env constructor (``None`` = default).
    """
    # Lazy imports: keep module load cheap and sb3-free.
    from retail_simulator.envs.parallel_env import RetailParallelEnv  # noqa: PLC0415
    from retail_simulator.envs.league import (  # noqa: PLC0415
        _position_controlled_head_to_head,
    )
    from retail_simulator.envs.parallel_gate import competition_config  # noqa: PLC0415

    effective_config = config if config is not None else competition_config()

    per_seed_by_pair: dict[tuple[str, str], list[float]] = {}
    component_per_seed_by_pair: dict[tuple[str, str], dict[str, tuple[float, ...]] | None] = {}
    env = RetailParallelEnv(
        effective_config, n_learning_agents=2, seed=seed_set[0], home_regions=home_regions
    )
    try:
        predicts = _materialize_predicts(
            spec_by_label,
            action_space=env.action_space(env.possible_agents[0]),
            # Always forward the effective config: it is only read (`.demand.regions`)
            # when a geo archetype participant is present, so this is harmless for
            # every other field.
            config=effective_config,
        )
        for i, label_a in enumerate(sorted_labels):
            for label_b in sorted_labels[i + 1 :]:
                per_seed: list[float] = []
                # Per-component per-seed margins for this pair; flips to None the
                # moment a seed's sink comes back empty (all-or-nothing for the pair).
                component_per_seed: dict[str, list[float]] | None = {}
                for seed in seed_set:
                    component_sink: dict[str, list[tuple[float, float]]] = {}
                    scores_a, scores_b = _position_controlled_head_to_head(
                        env,
                        effective_config.reward,
                        predict_a=predicts[label_a],
                        predict_b=predicts[label_b],
                        n_episodes=n_episodes,
                        seed=seed,
                        component_sink=component_sink,
                    )
                    per_seed.append(float(np.mean(scores_a)) - float(np.mean(scores_b)))
                    if component_per_seed is not None:
                        if component_sink:
                            for name, per_episode in component_sink.items():
                                margin = float(np.mean([a - b for a, b in per_episode]))
                                component_per_seed.setdefault(name, []).append(margin)
                        else:
                            component_per_seed = None
                per_seed_by_pair[(label_a, label_b)] = per_seed
                component_per_seed_by_pair[(label_a, label_b)] = (
                    {name: tuple(values) for name, values in component_per_seed.items()}
                    if component_per_seed is not None
                    else None
                )
    finally:
        env.close()
    return per_seed_by_pair, component_per_seed_by_pair


def _content_hash_ladder_id(
    participants: Sequence[tuple[str, str]],
    *,
    n_seeds: int,
    n_episodes: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    config_label: str = DEFAULT_ENV_LABEL,
    home_regions: tuple[int, ...] | None = None,
) -> str:
    """Derive a stable ladder_id from the field and run settings (config label / home regions only when non-default)."""
    fingerprint: dict[str, Any] = {
        "participants": [list(p) for p in participants],
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
    }
    if config_label != DEFAULT_ENV_LABEL:
        fingerprint["config_label"] = config_label
    if home_regions is not None:
        fingerprint["home_regions"] = list(home_regions)
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()
    return f"ladder-{digest[:12]}"


class _ScoreMarginRow(NamedTuple):
    """One pair's score margin + bootstrap CI as concrete floats.

    Kept separate from PairwiseMargin (whose CIs are ``float | None``) so
    _build_tournament_summary never has to narrow an Optional.
    """

    label_a: str
    label_b: str
    per_seed_margins: tuple[float, ...]
    mean_margin: float
    margin_se: float
    ci_low: float
    ci_high: float


def _build_tournament_summary(
    score_margin_rows: Sequence[_ScoreMarginRow],
    labels: Sequence[str],
    kind_by_label: Mapping[str, ParticipantKind],
) -> TournamentSummary:
    """Build the dominance/cycle/Condorcet summary from this ladder's score margins.

    Reuses spike._measure_tournament (dominance, 3-cycle count, Condorcet,
    first-triple decisiveness) over PairMargin records from score_margin_rows,
    and spike._triple_cycles to list EVERY decisive cyclic triple from the same
    beats/CI facts (no bootstrap re-run). kind_by_label counts the decisive
    triples touching a "checkpoint" participant.
    """
    from retail_simulator.envs.spike import (  # noqa: PLC0415
        PairMargin,
        _measure_tournament,
        _triple_cycles,
    )

    pair_margins = tuple(
        PairMargin(
            label_a=row.label_a,
            label_b=row.label_b,
            per_seed_margins=row.per_seed_margins,
            mean_margin=row.mean_margin,
            margin_se=row.margin_se,
            ci_low=row.ci_low,
            ci_high=row.ci_high,
        )
        for row in score_margin_rows
    )
    (
        dominance,
        n_cyclic_triples,
        has_condorcet_winner,
        condorcet_winner,
        _cycle_direction,
        decisive_cycle,
    ) = _measure_tournament(pair_margins, tuple(labels))
    triples = list(combinations(labels, 3))
    total_triples = len(triples)

    # Recompute the same per-pair beats/CI-excludes-zero facts _measure_tournament
    # uses internally, so every cyclic triple's decisiveness can be read, not just
    # the first one `decisive_cycle` mirrors.
    beats: dict[tuple[str, str], bool] = {}
    ci_excludes_zero: dict[frozenset[str], bool] = {}
    for pm in pair_margins:
        beats[(pm.label_a, pm.label_b)] = pm.mean_margin > 0.0
        beats[(pm.label_b, pm.label_a)] = pm.mean_margin < 0.0
        ci_excludes_zero[frozenset((pm.label_a, pm.label_b))] = (
            pm.ci_low > 0.0 if pm.mean_margin > 0.0 else pm.ci_high < 0.0
        )

    decisive_triples: list[tuple[str, str, str]] = []
    for triple in triples:
        order = _triple_cycles(triple, beats)
        if order is None:
            continue
        # Smallest-label-first is inherited: `labels` arrive sorted, so `triple[0]`
        # is the lexicographically smallest, and `_triple_cycles` walks from it.
        assert order[0] == triple[0]
        edges = (
            frozenset((order[0], order[1])),
            frozenset((order[1], order[2])),
            frozenset((order[2], order[0])),
        )
        if all(ci_excludes_zero[edge] for edge in edges):
            decisive_triples.append(order)

    checkpoint_inclusive_decisive_triples = sum(
        1
        for triple in decisive_triples
        if any(kind_by_label[label] == "checkpoint" for label in triple)
    )

    return TournamentSummary(
        dominance=dominance,
        n_cyclic_triples=n_cyclic_triples,
        total_triples=total_triples,
        has_condorcet_winner=has_condorcet_winner,
        condorcet_winner=condorcet_winner,
        decisive_cycle=decisive_cycle,
        decisive_triples=tuple(decisive_triples),
        checkpoint_inclusive_decisive_triples=checkpoint_inclusive_decisive_triples,
    )


def run_ladder(
    participants: Sequence[tuple[str, str]],
    *,
    n_seeds: int = 30,
    n_episodes: int = 1,
    ladder_id: str | None = None,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 0,
    head_to_head: Callable[..., tuple[list[float], list[float]]] | None = None,
    config: CoreConfig | None = None,
    config_label: str = DEFAULT_ENV_LABEL,
    home_regions: tuple[int, ...] | None = None,
) -> LadderResult:
    """Run the full ladder: validate inputs, roll out every pair over n_seeds, rank by field margin.

    ``head_to_head`` may inject a test seam for the pairwise rollouts;
    ``config``/``config_label`` and ``home_regions`` select the env and are
    validated up front.
    """
    if len(participants) < 2:
        raise ValueError(f"run_ladder requires at least 2 participants; got {len(participants)}")
    labels = [label for _spec, label in participants]
    if len(set(labels)) != len(labels):
        raise ValueError(f"run_ladder participant labels must be unique; got {labels}")
    if n_seeds < 1:
        raise ValueError(f"run_ladder requires n_seeds >= 1; got {n_seeds}")
    if n_episodes < 1:
        raise ValueError(f"run_ladder requires n_episodes >= 1; got {n_episodes}")
    # config and config_label are inseparable: the id/artifact path follow the
    # label, so a mismatch would mint the wrong id or clobber the default env's artifact.
    if config is not None and config_label == DEFAULT_ENV_LABEL:
        raise ValueError(
            "run_ladder: a non-default config requires a distinct config_label — "
            "the ladder_id and artifact path follow the LABEL, so a non-default "
            "config left at the default label would silently mint the DEFAULT "
            "env's id and overwrite its artifact"
        )
    if config is None and config_label != DEFAULT_ENV_LABEL:
        raise ValueError(
            f"run_ladder: config_label={config_label!r} was given but config is "
            "None (falls back to competition_config()) — the ladder_id/artifact "
            "path would then claim an env that never actually ran; pass the "
            "matching config"
        )
    if config is not None:
        # The checks above only distinguish default from non-default; compare the
        # label against config's region table instead. calibrated_spike_config's
        # table is CoreConfig's own 2-region default, so every other non-default
        # env differs, and full equality (not just length) also catches a
        # same-length coincidence. One check catches both mismatch directions.
        # Lazy import: harness.spike pulls the same heavy rollout stack as league.
        from retail_simulator.envs.spike import calibrated_spike_config  # noqa: PLC0415

        is_calibrated_shaped = config.demand.regions == calibrated_spike_config().demand.regions
        labeled_calibrated = config_label == CALIBRATED_ENV_LABEL
        if is_calibrated_shaped != labeled_calibrated:
            raise ValueError(
                f"run_ladder: config_label={config_label!r} does not match its "
                f"config's region table (config is calibrated-shaped: "
                f"{is_calibrated_shaped}, label claims calibrated: "
                f"{labeled_calibrated}) — a (config, config_label) pair naming "
                "the wrong env would mint an artifact that lies about which "
                "world ran"
            )
    if home_regions is not None:
        # Fail fast, before any rollout (mirrors the config/config_label checks
        # above); lazy import keeps this out of module-load time.
        from retail_simulator.envs.parallel_gate import competition_config  # noqa: PLC0415

        effective_n_regions = len(
            (config if config is not None else competition_config()).demand.regions
        )
        # Mirror envs/parallel_env's _build_seat_plan validation: the shared env
        # always seats exactly 2 learning agents and 0 NPCs, so only exactly 2
        # integer region indexes (bools excluded) can ever reach a real seat plan.
        if len(home_regions) != 2:
            raise ValueError(
                f"run_ladder: home_regions must have exactly 2 entries (the "
                "ladder's shared env always seats 2 learning agents, no "
                f"NPCs); got {len(home_regions)}"
            )
        for seat_index, home in enumerate(home_regions):
            if (
                not isinstance(home, (int, np.integer))
                or isinstance(home, bool)
                or not (0 <= home < effective_n_regions)
            ):
                raise ValueError(
                    f"run_ladder: home_regions[{seat_index}]={home!r} (type "
                    f"{type(home).__name__}) is invalid; must be an integer "
                    f"region index satisfying 0 <= r < {effective_n_regions}"
                )

    # Parse every spec up front: a malformed spec must raise before any rollout
    # runs, and normalization (e.g. "random:03" -> "random:3") gives the
    # content-hash id and every downstream consumer one canonical spelling.
    parsed_by_label: dict[str, ParticipantSpec] = {
        label: parse_participant_spec(spec) for spec, label in participants
    }
    kind_by_label: dict[str, ParticipantKind] = {
        label: parsed.kind for label, parsed in parsed_by_label.items()
    }
    normalized_participants = [(parsed_by_label[label].source, label) for label in labels]

    resolved_ladder_id = (
        ladder_id
        if ladder_id is not None
        else _content_hash_ladder_id(
            normalized_participants,
            n_seeds=n_seeds,
            n_episodes=n_episodes,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            config_label=config_label,
            home_regions=home_regions,
        )
    )
    seed_set = tuple(range(n_seeds))
    spec_by_label = {label: parsed.source for label, parsed in parsed_by_label.items()}

    # Every unordered pair in (label_a < label_b) order — stable regardless of input ordering.
    sorted_labels = sorted(labels)

    # Gather each pair's per-seed margins. The default path materializes every
    # participant once and reuses predicts across all pairs/seeds; the injected
    # head_to_head seam keeps its per-(pair, seed) call shape and never supplies
    # per-component margins (every pair gets None there).
    component_per_seed_by_pair: dict[tuple[str, str], dict[str, tuple[float, ...]] | None]
    if head_to_head is None:
        per_seed_by_pair, component_per_seed_by_pair = _default_pairwise_per_seed(
            sorted_labels,
            spec_by_label,
            seed_set=seed_set,
            n_episodes=n_episodes,
            config=config,
            home_regions=home_regions,
        )
    else:
        per_seed_by_pair = {}
        component_per_seed_by_pair = {}
        for i, label_a in enumerate(sorted_labels):
            for label_b in sorted_labels[i + 1 :]:
                per_seed_list: list[float] = []
                for seed in seed_set:
                    scores_a, scores_b = head_to_head(
                        spec_by_label[label_a],
                        spec_by_label[label_b],
                        n_episodes=n_episodes,
                        seed=seed,
                    )
                    per_seed_list.append(float(np.mean(scores_a)) - float(np.mean(scores_b)))
                per_seed_by_pair[(label_a, label_b)] = per_seed_list
                component_per_seed_by_pair[(label_a, label_b)] = None

    pair_mean: dict[tuple[str, str], float] = {}
    pair_margin_se: dict[tuple[str, str], float] = {}
    pair_per_seed: dict[tuple[str, str], np.ndarray] = {}
    for i, label_a in enumerate(sorted_labels):
        for label_b in sorted_labels[i + 1 :]:
            per_seed = per_seed_by_pair[(label_a, label_b)]
            per_seed_arr = np.asarray(per_seed, dtype=np.float64)
            mean_margin = float(per_seed_arr.mean())
            # Population sd / sqrt(n_seeds): the standard error under the fixed-seed-set contract.
            margin_se = float(np.std(per_seed_arr, ddof=0)) / sqrt(n_seeds)
            pair_mean[(label_a, label_b)] = mean_margin
            pair_margin_se[(label_a, label_b)] = margin_se
            pair_per_seed[(label_a, label_b)] = per_seed_arr

    # Field-margin per participant (mean margin vs the rest; self excluded).
    field_margin_by_label: dict[str, float] = {}
    for label in sorted_labels:
        others = [other for other in sorted_labels if other != label]
        field_margin_by_label[label] = _field_margin_from_pair_means(label, others, pair_mean)

    # Bootstrap: draw the seed-INDEX resamples once (an RNG independent of rollout
    # seeds) and apply them consistently to every participant, so one iteration is
    # one coherent surrogate seed set, reused for every per-pair CI below.
    rng = np.random.default_rng(bootstrap_seed)
    resampled_seed_indices = rng.integers(0, n_seeds, size=(bootstrap_resamples, n_seeds))
    ci_by_label: dict[str, tuple[float, float]] = {}
    for label in sorted_labels:
        others = [other for other in sorted_labels if other != label]
        ci_by_label[label] = _bootstrap_margin_ci(
            label,
            others,
            pair_per_seed,
            n_seeds=n_seeds,
            resampled_seed_indices=resampled_seed_indices,
        )

    # Per-pair score CI (the same draw as above) + per-component mean/CI/per-seed
    # breakdown, plus the raw rows _build_tournament_summary needs.
    # component_pair_mean feeds component_field_margins via the same
    # _field_margin_from_pair_means helper the score field-margin uses.
    pairwise: list[PairwiseMargin] = []
    component_pair_mean: dict[str, dict[tuple[str, str], float]] = {}
    all_pairs_have_components = True
    score_margin_rows: list[_ScoreMarginRow] = []
    for i, label_a in enumerate(sorted_labels):
        for label_b in sorted_labels[i + 1 :]:
            pair_key = (label_a, label_b)
            mean_margin = pair_mean[pair_key]
            margin_se = pair_margin_se[pair_key]
            per_seed_tuple = tuple(float(v) for v in pair_per_seed[pair_key])
            ci_low, ci_high = _bootstrap_margin_ci(
                label_a,
                [label_b],
                pair_per_seed,
                n_seeds=n_seeds,
                resampled_seed_indices=resampled_seed_indices,
            )
            score_margin_rows.append(
                _ScoreMarginRow(
                    label_a=label_a,
                    label_b=label_b,
                    per_seed_margins=per_seed_tuple,
                    mean_margin=mean_margin,
                    margin_se=margin_se,
                    ci_low=ci_low,
                    ci_high=ci_high,
                )
            )

            components = component_per_seed_by_pair[pair_key]
            component_mean_margins: dict[str, float] | None = None
            component_ci: dict[str, tuple[float, float]] | None = None
            component_per_seed_margins: dict[str, tuple[float, ...]] | None = None
            if components is None:
                all_pairs_have_components = False
            else:
                component_mean_margins = {}
                component_ci = {}
                # Defensive copy: PairwiseMargin is frozen/public, so callers must not
                # mutate run_ladder's internal dict through the returned entry.
                component_per_seed_margins = dict(components)
                for name, values in components.items():
                    arr = np.asarray(values, dtype=np.float64)
                    component_mean_margins[name] = float(arr.mean())
                    component_pair_mean.setdefault(name, {})[pair_key] = float(arr.mean())
                    c_ci_low, c_ci_high = _bootstrap_margin_ci(
                        label_a,
                        [label_b],
                        {pair_key: arr},
                        n_seeds=n_seeds,
                        resampled_seed_indices=resampled_seed_indices,
                    )
                    component_ci[name] = (c_ci_low, c_ci_high)

            pairwise.append(
                PairwiseMargin(
                    label_a=label_a,
                    label_b=label_b,
                    per_seed_margins=per_seed_tuple,
                    mean_margin=mean_margin,
                    margin_se=margin_se,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    component_mean_margins=component_mean_margins,
                    component_ci=component_ci,
                    component_per_seed_margins=component_per_seed_margins,
                )
            )

    # Rank field-margin DESC; tie-break by label for a deterministic total order.
    ranked_labels = sorted(sorted_labels, key=lambda lbl: (-field_margin_by_label[lbl], lbl))
    entries: list[LadderEntry] = []
    for rank_index, label in enumerate(ranked_labels, start=1):
        ci_low, ci_high = ci_by_label[label]
        component_field_margins: dict[str, float] | None = None
        if all_pairs_have_components:
            others = [other for other in sorted_labels if other != label]
            component_field_margins = {
                name: _field_margin_from_pair_means(label, others, component_pair_mean[name])
                for name in component_pair_mean
            }
        entries.append(
            LadderEntry(
                label=label,
                checkpoint_path=spec_by_label[label],
                field_margin=field_margin_by_label[label],
                ci_low=ci_low,
                ci_high=ci_high,
                rank=rank_index,
                kind=kind_by_label[label],
                component_field_margins=component_field_margins,
            )
        )

    kendall_tau_split_half = _split_half_kendall_tau(sorted_labels, pair_per_seed, n_seeds=n_seeds)
    # The tournament summary is always computed from this ladder's own score
    # margins, on both the default and injected-seam paths.
    tournament = _build_tournament_summary(score_margin_rows, sorted_labels, kind_by_label)

    return LadderResult(
        ladder_id=resolved_ladder_id,
        n_seeds=n_seeds,
        seed_set=seed_set,
        n_episodes=n_episodes,
        entries=tuple(entries),
        pairwise=tuple(pairwise),
        kendall_tau_split_half=kendall_tau_split_half,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        config_label=config_label,
        tournament=tournament,
        home_regions=home_regions,
    )


def _rank_labels_on_seed_subset(
    labels: Sequence[str],
    pair_per_seed: dict[tuple[str, str], np.ndarray],
    seed_indices: np.ndarray,
) -> list[str]:
    """Rank labels by field-margin computed over only seed_indices.

    Used by the split-half diagnostic (evens / odds seed subsets).
    """
    field_by_label: dict[str, float] = {}
    for label in labels:
        others = [other for other in labels if other != label]
        oriented_rows: list[np.ndarray] = []
        for other in others:
            if label < other:
                oriented_rows.append(pair_per_seed[(label, other)][seed_indices])
            else:
                oriented_rows.append(-pair_per_seed[(other, label)][seed_indices])
        oriented = np.asarray(oriented_rows, dtype=np.float64)
        field_by_label[label] = float(oriented.mean())
    return sorted(labels, key=lambda lbl: (-field_by_label[lbl], lbl))


def _split_half_kendall_tau(
    labels: Sequence[str],
    pair_per_seed: dict[tuple[str, str], np.ndarray],
    *,
    n_seeds: int,
) -> float | None:
    """Kendall tau between the field rankings over even and odd seeds (None below 4 seeds)."""
    if n_seeds < 4:
        return None
    even_indices = np.arange(0, n_seeds, 2)
    odd_indices = np.arange(1, n_seeds, 2)
    even_order = _rank_labels_on_seed_subset(labels, pair_per_seed, even_indices)
    odd_order = _rank_labels_on_seed_subset(labels, pair_per_seed, odd_indices)
    # Map each label to its rank in each half, then τ over those ranks in a fixed label order.
    even_rank = {label: pos for pos, label in enumerate(even_order)}
    odd_rank = {label: pos for pos, label in enumerate(odd_order)}
    fixed = sorted(labels)
    rank_a = [even_rank[label] for label in fixed]
    rank_b = [odd_rank[label] for label in fixed]
    return _kendall_tau(rank_a, rank_b)
