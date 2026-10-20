"""Pairwise rollout orchestration and tournament summary construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np

from retail_simulator.core.config import CoreConfig

from .model import ParticipantKind, TournamentSummary
from .policy import _materialize_predicts
from .stats import _ScoreMarginRow


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
    from retail_simulator.envs.league import (
        _position_controlled_head_to_head,
    )
    from retail_simulator.envs.parallel_env import RetailParallelEnv
    from retail_simulator.envs.parallel_gate import competition_config

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
    from retail_simulator.envs.spike import (
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
