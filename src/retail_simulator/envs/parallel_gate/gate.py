"""The multi-agent self-play learnability gate: train a shared policy, then gate it.

Selects the training substrate + on-vs-off (or head-to-head) evaluation, builds the
param-sharing bridge, trains PPO, and summarizes the result in a GateReport. Heavy
RL/env imports stay lazy so ``import retail_simulator.envs.parallel_gate`` is cheap.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

import numpy as np

from retail_simulator.envs.learnability import (
    DEFAULT_EVAL_EPISODES,
    FULL_TOTAL_STEPS,
    GATE_SIGMA_MULTIPLE,
    GateReport,
    _curve_trends_up,
)
from retail_simulator.envs.sweep import (
    RESEARCH_OPPONENT_ARCHETYPE,
    automation_gate_config,
    loyalty_gate_config,
    research_gate_segments,
    scm_gate_config,
)

from .bridge import (
    DEFAULT_N_LEARNING_AGENTS,
    DIVERSITY_MAX_RATIO,
    DIVERSITY_WITHIN_FRACTION,
    _make_curve_callback,
    _make_shared_ppo,
    build_parallel_vec_env,
    competition_config,
)
from .lever_eval import (
    evaluate_automation_on_vs_off,
    evaluate_loyalty_on_vs_off,
    evaluate_research_on_vs_off,
    evaluate_scm_on_vs_off,
)
from .stationary import evaluate_head_to_head

if TYPE_CHECKING:  # type-only; never pulled in at runtime import time
    from retail_simulator.envs.curriculum import ScenarioDistribution


def run_parallel_gate(
    *,
    total_steps: int = FULL_TOTAL_STEPS,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    seed: int = 42,
    num_vec_envs: int = 1,
    num_cpus: int = 1,
    reward_mode: str = "weighted",
    research_gate: bool = False,
    research_opponent: str = RESEARCH_OPPONENT_ARCHETYPE,
    scm_gate: bool = False,
    automation_gate: bool = False,
    loyalty_gate: bool = False,
    curriculum: ScenarioDistribution | None = None,
    curriculum_seed: int = 0,
    policy_kind: Literal["mlp", "lstm"] = "mlp",
) -> GateReport:
    """Train a shared policy, then run the chosen on-vs-off (or head-to-head) gate."""
    # The four gates are mutually exclusive (each selects a different training
    # substrate + on-vs-off eval). The validation surfaces an explicit error so a typo
    # (e.g. setting two flags at once) fails fast at the boundary.
    if sum(int(g) for g in (research_gate, scm_gate, automation_gate, loyalty_gate)) > 1:
        raise ValueError(
            "research_gate, scm_gate, automation_gate, and loyalty_gate are mutually "
            "exclusive — each selects a different training substrate and a different "
            "on-vs-off eval (research vs the wandering opponent; SCM vs the "
            "Discounter in the SCM scenario; automation vs the Discounter in the "
            "automation scenario; loyalty vs the Discounter in the loyalty scenario)"
        )

    config = competition_config(reward_mode=reward_mode)
    if research_gate:
        config = replace(config, segments=research_gate_segments())
    elif scm_gate:
        config = scm_gate_config(config)
    elif automation_gate:
        config = automation_gate_config(config)
    elif loyalty_gate:
        config = loyalty_gate_config(config)
    npc_archetypes: tuple[str, ...]
    if research_gate:
        npc_archetypes = (research_opponent,)
    elif scm_gate or automation_gate or loyalty_gate:
        npc_archetypes = ("discounter",)
    else:
        npc_archetypes = ()

    vec = build_parallel_vec_env(
        config,
        n_learning_agents=n_learning_agents,
        npc_archetypes=npc_archetypes,
        num_vec_envs=num_vec_envs,
        num_cpus=num_cpus,
        seed=seed,
        curriculum=curriculum,
        curriculum_seed=curriculum_seed,
    )
    curve: list[tuple[int, float]] = []
    callback = _make_curve_callback(
        curve, log_every=max(total_steps // 20, 1), label="parallel gate"
    )
    model = _make_shared_ppo(vec, seed=seed, policy_kind=policy_kind)
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=False)
    vec.close()

    if research_gate:
        on_scores, off_scores = evaluate_research_on_vs_off(
            model,
            config,
            n_learning_agents=n_learning_agents,
            npc_archetypes=npc_archetypes,
            episodes=eval_episodes,
            seed=seed,
        )
        return build_parallel_report(
            trained_scores=on_scores,
            random_scores=off_scores,
            learning_curve=list(curve),
            total_steps=total_steps,
            eval_episodes=eval_episodes,
        )

    if scm_gate:
        on_scores, off_scores = evaluate_scm_on_vs_off(
            model,
            config,
            n_learning_agents=n_learning_agents,
            npc_archetypes=npc_archetypes,
            episodes=eval_episodes,
            seed=seed,
        )
        return build_parallel_report(
            trained_scores=on_scores,
            random_scores=off_scores,
            learning_curve=list(curve),
            total_steps=total_steps,
            eval_episodes=eval_episodes,
        )

    if automation_gate:
        on_scores, off_scores = evaluate_automation_on_vs_off(
            model,
            config,
            n_learning_agents=n_learning_agents,
            npc_archetypes=npc_archetypes,
            episodes=eval_episodes,
            seed=seed,
        )
        return build_parallel_report(
            trained_scores=on_scores,
            random_scores=off_scores,
            learning_curve=list(curve),
            total_steps=total_steps,
            eval_episodes=eval_episodes,
        )

    if loyalty_gate:
        on_scores, off_scores = evaluate_loyalty_on_vs_off(
            model,
            config,
            n_learning_agents=n_learning_agents,
            npc_archetypes=npc_archetypes,
            episodes=eval_episodes,
            seed=seed,
        )
        return build_parallel_report(
            trained_scores=on_scores,
            random_scores=off_scores,
            learning_curve=list(curve),
            total_steps=total_steps,
            eval_episodes=eval_episodes,
        )

    trained_scores, random_scores = evaluate_head_to_head(
        model, config, n_learning_agents=n_learning_agents, episodes=eval_episodes, seed=seed
    )

    return build_parallel_report(
        trained_scores=trained_scores,
        random_scores=random_scores,
        learning_curve=list(curve),
        total_steps=total_steps,
        eval_episodes=eval_episodes,
    )


def build_parallel_report(
    *,
    trained_scores: list[float],
    random_scores: list[float],
    learning_curve: list[tuple[int, float]],
    total_steps: int,
    eval_episodes: int,
) -> GateReport:
    """Summarize trained vs random scores + the learning curve into a GateReport."""
    trained_mean = float(np.mean(trained_scores))
    random_mean = float(np.mean(random_scores))
    random_std = float(np.std(random_scores))
    margin = trained_mean - random_mean
    threshold = GATE_SIGMA_MULTIPLE * random_std
    beats_random = margin >= threshold
    trends_up = _curve_trends_up(learning_curve)
    return GateReport(
        ppo_mean_reward=trained_mean,
        random_mean_reward=random_mean,
        random_std_reward=random_std,
        margin=margin,
        threshold=threshold,
        beats_random=beats_random,
        curve_trends_up=trends_up,
        total_steps=total_steps,
        eval_episodes=eval_episodes,
        learning_curve=tuple((int(s), float(r)) for s, r in learning_curve),
    )


@dataclass(frozen=True)
class DiversityReport:
    """Aggregate roster-diversity stats over per-policy scores."""

    best_score: float
    min_score: float
    fraction_within: float
    n_within: int
    max_min_ratio: float
    meets_soft_goal: bool


def diversity_report(policy_scores: dict[str, float]) -> DiversityReport:
    """Roster-diversity stats (within-fraction, max/min ratio) over per-policy scores."""
    if not policy_scores:
        raise ValueError("policy_scores must not be empty")

    scores = list(policy_scores.values())
    best = max(scores)
    worst = min(scores)

    # Fraction within 80% of best. With a non-positive best (every policy <= 0 — a
    # degenerate/failed roster) no policy "competes": the within-fraction is 0.
    if best > 0.0:
        threshold = DIVERSITY_WITHIN_FRACTION * best
        n_within = sum(1 for s in scores if s >= threshold)
    else:
        n_within = 0
    fraction_within = n_within / len(scores)

    if worst > 0.0:
        max_min_ratio = best / worst
    else:
        max_min_ratio = float("inf") if best > 0.0 else 0.0

    meets_soft_goal = n_within >= 2 and max_min_ratio <= DIVERSITY_MAX_RATIO

    return DiversityReport(
        best_score=float(best),
        min_score=float(worst),
        fraction_within=float(fraction_within),
        n_within=int(n_within),
        max_min_ratio=float(max_min_ratio),
        meets_soft_goal=bool(meets_soft_goal),
    )
