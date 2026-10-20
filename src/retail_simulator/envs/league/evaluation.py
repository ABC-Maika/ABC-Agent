"""League evaluation: round-end head-to-head folds and improvement diagnostics.

Round-end evaluation feeds the win-rate matrix; the snapshot-improvement
report compares round N vs round 1; the generalization diagnostic scores
curriculum-PPO vs fixed-PPO on held-out scenarios. Heavy imports stay lazy.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from retail_simulator.logging_setup import human_steps

from .winrate import WinRateEntry

_log = logging.getLogger(__name__)

if TYPE_CHECKING:  # type-only; never pulled in at runtime import time
    from retail_simulator.core.config import CoreConfig
    from retail_simulator.envs.curriculum import ScenarioDistribution

    from .pool import FrozenPolicyPool, PoolMember
    from .training import LeagueTrainingResult


@dataclass(frozen=True)
class PoolSnapshotImprovementReport:
    """Head-to-head report comparing the latest snapshot against the round-1 snapshot."""

    round_n_mean: float
    round_1_mean: float
    round_n_std: float
    round_1_std: float
    margin: float
    threshold: float
    beats_round_1: bool
    n_episodes: int
    per_round_means: tuple[tuple[int, float], ...]


# The four head-to-head-scored reward components, in RewardComponents
# field order; the scoreboard sink-filling code below iterates this tuple.
HEAD_TO_HEAD_COMPONENTS: tuple[str, ...] = ("profit", "revenue", "market_share", "loyalty")

# The subset of the above whose per-episode reduction is a per-tick MEAN
# (share/EMA are per-tick quantities; summing them across ticks is not
# meaningful). A named constant so a new component can't inherit the wrong
# reduction by accident.
PER_TICK_MEAN_COMPONENTS: tuple[str, ...] = ("market_share", "loyalty")


def _head_to_head_episode_scores(
    env: Any,
    cfg: Any,
    *,
    seed: int,
    predict_for_agent: dict[str, Callable[..., Any]],
    component_sink: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Run one full episode; return each agent's total weighted objective."""
    from retail_simulator.envs.learnability import DEFAULT_MAX_EPISODE_STEPS
    from retail_simulator.envs.parallel_gate import (
        _components_from_info,
        weighted_objective,
    )

    observations, _infos = env.reset(seed=seed)
    totals: dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
    if component_sink is not None:
        for aid in env.possible_agents:
            component_sink[aid] = {
                "profit": 0.0,
                "revenue": 0.0,
                "market_share": 0.0,
                "loyalty": 0.0,
            }
    is_first_step = True
    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
        actions = {
            aid: predict_for_agent[aid](observations[aid], episode_start=is_first_step)
            for aid in env.agents
        }
        observations, _rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            totals[aid] += weighted_objective(_components_from_info(infos[aid]), cfg)
            if component_sink is not None:
                # A second, separate reconstruction from the same info, kept
                # apart so the score-computation line above stays untouched.
                components = _components_from_info(infos[aid])
                component_sink[aid]["profit"] += components.profit
                component_sink[aid]["revenue"] += components.revenue
                component_sink[aid]["market_share"] += components.market_share
                component_sink[aid]["loyalty"] += components.loyalty
        is_first_step = False
    if component_sink is not None:
        for aid in env.possible_agents:
            component_sink[aid]["n_ticks"] = float(DEFAULT_MAX_EPISODE_STEPS)
    return totals


def _position_controlled_head_to_head(
    env: Any,
    cfg: Any,
    *,
    predict_a: Callable[..., Any],
    predict_b: Callable[..., Any],
    n_episodes: int,
    seed: int,
    env_factory: Callable[[], Any] | None = None,
    component_sink: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[list[float], list[float]]:
    """Position-controlled head-to-head over n_episodes; return per-episode seat means."""
    scores_a: list[float] = []
    scores_b: list[float] = []
    # Agent IDs are stable across env constructions (seats 0..N-1), so the
    # passed env's agent list is a valid stand-in. When env is None (the
    # reuse_env=False path), peek at one freshly-built env and close it just
    # to learn the set.
    if env is not None:
        possible_agents = list(env.possible_agents)
    else:
        if env_factory is None:  # pragma: no cover - defensive; not reachable from callers
            raise ValueError(
                "_position_controlled_head_to_head: either env or env_factory must be provided"
            )
        peek_env = env_factory()
        try:
            possible_agents = list(peek_env.possible_agents)
        finally:
            peek_env.close()
    for episode in range(n_episodes):
        episode_seed = seed + episode
        a_seat_vals: list[float] = []
        b_seat_vals: list[float] = []
        # One (a_seat, component sink) pair per seat rotation this episode,
        # collected so the per-component means mirror scores_a/scores_b.
        rotation_component_sinks: list[tuple[str, dict[str, dict[str, float]]]] = []
        for a_seat in possible_agents:
            predict_for_agent: dict[str, Callable[..., Any]] = {
                aid: (predict_a if aid == a_seat else predict_b) for aid in possible_agents
            }
            inner_component_sink: dict[str, dict[str, float]] | None = (
                {} if component_sink is not None else None
            )
            if env_factory is None:
                totals = _head_to_head_episode_scores(
                    env,
                    cfg,
                    seed=episode_seed,
                    predict_for_agent=predict_for_agent,
                    component_sink=inner_component_sink,
                )
            else:
                # Fresh-env path: build, reset+step, close.
                episode_env = env_factory()
                try:
                    totals = _head_to_head_episode_scores(
                        episode_env,
                        cfg,
                        seed=episode_seed,
                        predict_for_agent=predict_for_agent,
                        component_sink=inner_component_sink,
                    )
                finally:
                    episode_env.close()
            a_seat_vals.append(totals[a_seat])
            b_seat_vals.extend(v for aid, v in totals.items() if aid != a_seat)
            if inner_component_sink is not None:
                rotation_component_sinks.append((a_seat, inner_component_sink))
        scores_a.append(float(np.mean(a_seat_vals)))
        scores_b.append(float(np.mean(b_seat_vals)))
        if component_sink is not None:
            for name in HEAD_TO_HEAD_COMPONENTS:
                component_a_vals: list[float] = []
                component_b_vals: list[float] = []
                for rotation_a_seat, inner_sink in rotation_component_sinks:
                    for aid, agent_totals in inner_sink.items():
                        value = (
                            agent_totals[name] / agent_totals["n_ticks"]
                            if name in PER_TICK_MEAN_COMPONENTS
                            else agent_totals[name]
                        )
                        if aid == rotation_a_seat:
                            component_a_vals.append(value)
                        else:
                            component_b_vals.append(value)
                a_side_mean = float(np.mean(component_a_vals))
                b_side_mean = float(np.mean(component_b_vals))
                component_sink.setdefault(name, []).append((a_side_mean, b_side_mean))
    return scores_a, scores_b


_PER_ROUND_CURVE_EPISODES: int = 3


def evaluate_pool_snapshot_improvement(
    pool: FrozenPolicyPool,
    training_result: LeagueTrainingResult,
    *,
    n_eval_episodes: int = 20,
    config: CoreConfig | None = None,
    seed: int = 0,
    reuse_env: bool = True,
    home_regions: tuple[int, ...] | None = None,
) -> PoolSnapshotImprovementReport:
    """Compare the round-N snapshot against round 1 head-to-head and report the margin."""
    # Lazy import: RetailParallelEnv pulls pettingzoo, which this module
    # otherwise avoids until call time.
    from retail_simulator.envs.learnability import GATE_SIGMA_MULTIPLE
    from retail_simulator.envs.parallel_env import RetailParallelEnv
    from retail_simulator.envs.parallel_gate import competition_config

    checkpoints = training_result.checkpoints
    if len(checkpoints) < 2:
        raise ValueError(
            "evaluate_pool_snapshot_improvement requires at least 2 checkpoints "
            "(round-1 and round-N) to compare; "
            f"got {len(checkpoints)} checkpoint(s) in training_result"
        )

    # Round 1 is the first trained checkpoint (trained against the round-0
    # random-init pool); round N (max round_index) is the latest snapshot.
    by_round = {member.round_index: member for member in checkpoints}
    if 1 not in by_round:
        raise ValueError(
            "evaluate_pool_snapshot_improvement requires a checkpoint at round_index=1 "
            f"(the first trained snapshot); got round_indexes {sorted(by_round)}"
        )
    round_1_member = by_round[1]
    round_n_member = checkpoints[-1]  # checkpoints are appended in round order

    effective_config = config if config is not None else competition_config()
    cfg = effective_config.reward

    predict_round_n = pool.materialize(round_n_member, env=None)
    predict_round_1 = pool.materialize(round_1_member, env=None)

    # Either build the eval env once and reuse it across all episodes
    # (default), or build a fresh env per inner episode (the reuse_env=False
    # path). The two paths produce field-equal reports.
    def _build_eval_env() -> Any:
        return RetailParallelEnv(
            effective_config, n_learning_agents=2, seed=seed, home_regions=home_regions
        )

    if reuse_env:
        shared_env: Any | None = _build_eval_env()
        primitive_env_factory: Callable[[], Any] | None = None
    else:
        shared_env = None
        primitive_env_factory = _build_eval_env
    try:
        round_n_scores, round_1_scores = _position_controlled_head_to_head(
            shared_env,
            cfg,
            predict_a=predict_round_n,
            predict_b=predict_round_1,
            n_episodes=n_eval_episodes,
            seed=seed,
            env_factory=primitive_env_factory,
        )

        # Per-round means curve: for each checkpoint k (round-index order),
        # a cheap mini-eval of round-k vs round-1 (round-1 vs round-0 when
        # k == 1). Each entry is round-k's mean over _PER_ROUND_CURVE_EPISODES.
        # The pool's load cache reads each checkpoint zip at most once.
        per_round: list[tuple[int, float]] = []
        for member in checkpoints:
            predict_k = pool.materialize(member, env=None)
            if member.round_index == 1:
                # Round 1's peer is round 0 (the random-init snapshot).
                round_0 = by_round.get(0)
                if round_0 is None:
                    # A 1-round result may lack round 0 (a hand-built pool);
                    # compare round 1 against itself so the curve is defined.
                    peer_predict = predict_k
                else:
                    peer_predict = pool.materialize(round_0, env=None)
            else:
                peer_predict = predict_round_1
            k_scores, _peer_scores = _position_controlled_head_to_head(
                shared_env,
                cfg,
                predict_a=predict_k,
                predict_b=peer_predict,
                n_episodes=_PER_ROUND_CURVE_EPISODES,
                seed=seed + 10_000 + member.round_index,
                env_factory=primitive_env_factory,
            )
            per_round.append((member.round_index, float(np.mean(k_scores))))
    finally:
        if shared_env is not None:
            shared_env.close()

    round_n_mean = float(np.mean(round_n_scores))
    round_1_mean = float(np.mean(round_1_scores))
    round_n_std = float(np.std(round_n_scores))
    round_1_std = float(np.std(round_1_scores))
    margin = round_n_mean - round_1_mean
    threshold = float(GATE_SIGMA_MULTIPLE) * round_1_std
    beats_round_1 = margin >= threshold

    return PoolSnapshotImprovementReport(
        round_n_mean=round_n_mean,
        round_1_mean=round_1_mean,
        round_n_std=round_n_std,
        round_1_std=round_1_std,
        margin=margin,
        threshold=threshold,
        beats_round_1=beats_round_1,
        n_episodes=n_eval_episodes,
        per_round_means=tuple(per_round),
    )


# --- Round-end head-to-head evaluation ----------------------------------------


def _run_round_end_evaluation(
    pool: FrozenPolicyPool,
    learner_member: PoolMember,
    *,
    n_episodes: int = 5,
    config: CoreConfig | None = None,
    seed: int = 0,
    home_regions: tuple[int, ...] | None = None,
) -> dict[str, WinRateEntry]:
    """Round-end head-to-head of the new checkpoint vs every other pool member."""
    # Lazy imports: RetailParallelEnv pulls pettingzoo, kept out of module load.
    from retail_simulator.envs.learnability import DEFAULT_MAX_EPISODE_STEPS
    from retail_simulator.envs.parallel_env import RetailParallelEnv
    from retail_simulator.envs.parallel_gate import (
        _components_from_info,
        competition_config,
        weighted_objective,
    )

    if learner_member.kind != "checkpoint":
        raise ValueError(
            f"_run_round_end_evaluation: learner_member must be a checkpoint, "
            f"got kind={learner_member.kind!r} (the learner is the current round's "
            f"trained policy; scripted members can only be opponents)"
        )

    opponents = tuple(m for m in pool.members() if m.member_id != learner_member.member_id)
    if not opponents:
        return {}

    effective_config = config if config is not None else competition_config()
    reward_cfg = effective_config.reward

    learner_predict = pool.materialize(learner_member, env=None)
    rng = np.random.default_rng(seed)
    results: dict[str, WinRateEntry] = {}

    for opp_idx, opp in enumerate(opponents):
        wins = 0
        if opp.kind in ("checkpoint", "archetype"):
            # Checkpoint-vs-checkpoint (or vs an archetype): position-
            # controlled head-to-head, reusing the shared primitive so the
            # scoring substrate is identical to evaluate_pool_snapshot_improvement.
            opp_predict = pool.materialize(opp, env=None)
            env = RetailParallelEnv(
                effective_config, n_learning_agents=2, seed=seed, home_regions=home_regions
            )
            try:
                # Per-opponent base seed offset so concurrent eval loops
                # don't collide on the same episode_seed.
                pair_seed = seed + opp_idx * 10_000
                learner_scores, opp_scores = _position_controlled_head_to_head(
                    env,
                    reward_cfg,
                    predict_a=learner_predict,
                    predict_b=opp_predict,
                    n_episodes=n_episodes,
                    seed=pair_seed,
                )
            finally:
                env.close()
            # One episode -> one win/loss outcome (mean per-seat scores;
            # ties via the seeded rng coin flip).
            for learner_score, opp_score in zip(learner_scores, opp_scores, strict=True):
                if learner_score > opp_score:
                    wins += 1
                elif learner_score == opp_score:  # noqa: SIM102
                    if rng.random() < 0.5:
                        wins += 1
        else:
            # Scripted opponent: a FRESH single-learner env per episode,
            # because each episode's seed differs and the env is bound to its
            # constructor seed.
            for ep_idx in range(n_episodes):
                ep_seed = seed + opp_idx * 10_000 + ep_idx * 1_000
                env_sl = RetailParallelEnv(
                    effective_config,
                    n_learning_agents=1,
                    npc_archetypes=(opp.archetype_name,),
                    seed=ep_seed,
                    home_regions=home_regions,
                )
                try:
                    observations, _infos = env_sl.reset(seed=ep_seed)
                    learner_total = 0.0
                    is_first_step = True
                    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
                        learner_obs = observations["retailer_0"]
                        learner_action = learner_predict(learner_obs, episode_start=is_first_step)
                        joint = {"retailer_0": learner_action}
                        observations, _rewards, _term, _trunc, infos = env_sl.step(joint)
                        learner_total += weighted_objective(
                            _components_from_info(infos["retailer_0"]), reward_cfg
                        )
                        is_first_step = False
                finally:
                    env_sl.close()
                # Fallback win signal: the env does not surface the NPC's
                # per-seat objective, so treat the scripted opponent as a
                # zero-objective baseline - learner wins iff its score > 0.
                # Non-degenerate enough for PFSP's weight curve; exact-zero
                # ties resolve via the seeded coin flip.
                if learner_total > 0.0:
                    wins += 1
                elif learner_total == 0.0:  # noqa: SIM102
                    if rng.random() < 0.5:
                        wins += 1
        results[opp.member_id] = WinRateEntry(wins=wins, total=n_episodes)

    return results


@dataclass(frozen=True)
class RealPpoGeneralizationDiagnostic:
    """Per-scenario means for curriculum-PPO vs fixed-PPO after a real training run."""

    mean_curriculum_score: float
    mean_fixed_score: float
    per_scenario_curriculum_means: tuple[float, ...]
    per_scenario_fixed_means: tuple[float, ...]
    margin: float
    n_scenarios: int
    n_episodes_per_scenario: int
    train_steps: int
    per_scenario_curriculum_std_rewards: tuple[float, ...] = ()
    per_scenario_fixed_std_rewards: tuple[float, ...] = ()


def _default_holdout_scenarios() -> tuple[CoreConfig, ...]:
    """Build the default held-out configs (mirror-B and mirror-C variants)."""
    from retail_simulator.envs.curriculum import (
        _VARIANT_B_REGION_0_MIX,
        _VARIANT_B_REGION_1_MIX,
        _VARIANT_C_REGION_0_MIX,
        _VARIANT_C_REGION_1_MIX,
        phase_1_4_six_segment_config,
    )

    # Mirror variant-B (region-1 mix in region 0; region-0 mix in region 1).
    mirror_b = phase_1_4_six_segment_config(
        region_0_mix=_VARIANT_B_REGION_1_MIX,
        region_1_mix=_VARIANT_B_REGION_0_MIX,
    )
    # Mirror variant-C (the symmetric mirroring of the bargain-leaning regime).
    mirror_c = phase_1_4_six_segment_config(
        region_0_mix=_VARIANT_C_REGION_1_MIX,
        region_1_mix=_VARIANT_C_REGION_0_MIX,
    )
    return (mirror_b, mirror_c)


# Holdout scenarios the CLI's league generalization-diagnostic command
# defaults to; built lazily on first access to keep imports cheap.
_DEFAULT_HOLDOUT_SCENARIOS: tuple[CoreConfig, ...] | None = None


def default_holdout_scenarios() -> tuple[CoreConfig, ...]:
    """Cached accessor for the seed held-out scenarios.

    Populated on first call so the scenario helpers (and pettingzoo) are not
    pulled at module import; subsequent calls return the SAME tuple instance.
    """
    global _DEFAULT_HOLDOUT_SCENARIOS
    if _DEFAULT_HOLDOUT_SCENARIOS is None:
        _DEFAULT_HOLDOUT_SCENARIOS = _default_holdout_scenarios()
    return _DEFAULT_HOLDOUT_SCENARIOS


def run_real_ppo_generalization_diagnostic(
    curriculum: ScenarioDistribution,
    holdout_scenarios: tuple[CoreConfig, ...],
    train_steps: int,
    seeds: tuple[int, int],
    *,
    n_eval_episodes_per_scenario: int = 5,
    fixed_config: CoreConfig | None = None,
    n_learning_agents: int = 2,
) -> RealPpoGeneralizationDiagnostic:
    """Train curriculum-PPO and fixed-PPO, then score both on the held-out scenarios."""
    if not holdout_scenarios:
        raise ValueError(
            "run_real_ppo_generalization_diagnostic requires at least one held-out "
            "scenario; the diagnostic measures per-scenario means"
        )
    if train_steps < 1:
        raise ValueError(
            f"train_steps must be >= 1, got {train_steps}; "
            "at least one timestep is needed to train either model"
        )
    if n_eval_episodes_per_scenario < 1:
        raise ValueError(
            f"n_eval_episodes_per_scenario must be >= 1, got {n_eval_episodes_per_scenario}; "
            "at least one evaluation episode per scenario is needed for a mean"
        )

    # Lazy heavy imports keep module import cheap.
    from retail_simulator.envs.curriculum import evaluate_generalization
    from retail_simulator.envs.parallel_gate import (
        _make_curve_callback,
        _make_shared_ppo,
        build_parallel_vec_env,
        competition_config,
    )

    curriculum_seed, core_seed = seeds

    # --- Train the curriculum-PPO -------------------------------------------
    # build_parallel_vec_env wraps the parallel env in the curriculum wrapper
    # (training-time only; the schema is unchanged, so the bridged Box is
    # identical to the fixed run).
    vec_curric = build_parallel_vec_env(
        n_learning_agents=n_learning_agents,
        seed=core_seed,
        curriculum=curriculum,
        curriculum_seed=curriculum_seed,
    )
    _log.info(
        "generalization diagnostic: training curriculum-PPO for %s steps",
        human_steps(train_steps),
    )
    try:
        curriculum_model = _make_shared_ppo(vec_curric, seed=core_seed)
        curriculum_curve: list[tuple[int, float]] = []
        curriculum_model.learn(
            total_timesteps=train_steps,
            callback=_make_curve_callback(
                curriculum_curve,
                log_every=max(train_steps // 20, 1),
                label="curriculum PPO",
                logger=_log,
            ),
            progress_bar=False,
        )
    finally:
        vec_curric.close()

    # --- Train the fixed-PPO -------------------------------------------------
    fixed_config_effective = fixed_config if fixed_config is not None else competition_config()
    vec_fixed = build_parallel_vec_env(
        config=fixed_config_effective,
        n_learning_agents=n_learning_agents,
        seed=core_seed,
    )
    _log.info(
        "generalization diagnostic: training fixed-PPO for %s steps",
        human_steps(train_steps),
    )
    try:
        fixed_model = _make_shared_ppo(vec_fixed, seed=core_seed)
        fixed_curve: list[tuple[int, float]] = []
        fixed_model.learn(
            total_timesteps=train_steps,
            callback=_make_curve_callback(
                fixed_curve,
                log_every=max(train_steps // 20, 1),
                label="fixed PPO",
                logger=_log,
            ),
            progress_bar=False,
        )
    finally:
        vec_fixed.close()

    holdout_tuple = tuple(holdout_scenarios)
    report = evaluate_generalization(
        curriculum_model,
        fixed_model,
        holdout_tuple,
        episodes_per_scenario=n_eval_episodes_per_scenario,
        n_learning_agents=n_learning_agents,
        seed=core_seed,
    )

    return RealPpoGeneralizationDiagnostic(
        mean_curriculum_score=report.mean_curriculum_score,
        mean_fixed_score=report.mean_fixed_score,
        per_scenario_curriculum_means=report.per_scenario_curriculum_means,
        per_scenario_fixed_means=report.per_scenario_fixed_means,
        margin=report.margin,
        n_scenarios=report.n_scenarios,
        n_episodes_per_scenario=report.n_episodes_per_scenario,
        train_steps=int(train_steps),
    )


__all__ = [
    "HEAD_TO_HEAD_COMPONENTS",
    "PER_TICK_MEAN_COMPONENTS",
    "PoolSnapshotImprovementReport",
    "RealPpoGeneralizationDiagnostic",
    "default_holdout_scenarios",
    "evaluate_pool_snapshot_improvement",
    "run_real_ppo_generalization_diagnostic",
]
