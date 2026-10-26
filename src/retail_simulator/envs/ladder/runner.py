"""The run_ladder entry point: validate, roll out, rank, and return LadderResult."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from math import sqrt

import numpy as np

_log = logging.getLogger(__name__)

from retail_simulator.core.config import CoreConfig

from .model import (
    CALIBRATED_ENV_LABEL,
    DEFAULT_ENV_LABEL,
    LadderEntry,
    LadderResult,
    PairwiseMargin,
    ParticipantKind,
    ParticipantSpec,
    parse_participant_spec,
)
from .stats import (
    _bootstrap_margin_ci,
    _content_hash_ladder_id,
    _field_margin_from_pair_means,
    _ScoreMarginRow,
    _split_half_kendall_tau,
)
from .tournament import _build_tournament_summary, _default_pairwise_per_seed


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
        from retail_simulator.envs.spike import calibrated_spike_config

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
        from retail_simulator.envs.parallel_gate import competition_config

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
    n_pairs = len(sorted_labels) * (len(sorted_labels) - 1) // 2
    _log.info(
        "ladder starting: %d participant(s), %d pair(s) over %d seed(s) x %d episode(s) [%s]",
        len(labels),
        n_pairs,
        n_seeds,
        n_episodes,
        config_label,
    )

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
    pair_index = 0
    for i, label_a in enumerate(sorted_labels):
        for label_b in sorted_labels[i + 1 :]:
            pair_index += 1
            per_seed = per_seed_by_pair[(label_a, label_b)]
            per_seed_arr = np.asarray(per_seed, dtype=np.float64)
            mean_margin = float(per_seed_arr.mean())
            # Population sd / sqrt(n_seeds): the standard error under the fixed-seed-set contract.
            margin_se = float(np.std(per_seed_arr, ddof=0)) / sqrt(n_seeds)
            _log.info(
                "ladder pair %d/%d: %s vs %s — mean margin %+.2f",
                pair_index,
                n_pairs,
                label_a,
                label_b,
                mean_margin,
            )
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

    ladder_order = ", ".join(
        f"{label} ({field_margin_by_label[label]:+.2f})" for label in ranked_labels
    )
    _log.info("ladder done — ranking: %s", ladder_order)

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
