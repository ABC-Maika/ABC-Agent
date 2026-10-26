"""Stationary and head-to-head evaluation of shared policies on the parallel env."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from retail_simulator.core.config import CoreConfig, RewardConfig
from retail_simulator.core.reward import RewardComponents, weighted_objective
from retail_simulator.envs.learnability import DEFAULT_EVAL_EPISODES, DEFAULT_MAX_EPISODE_STEPS

from .bridge import DEFAULT_N_LEARNING_AGENTS

if TYPE_CHECKING:  # type-only; never pulled in at runtime import time
    from retail_simulator.envs.parallel_env import RetailParallelEnv


def _episode_stationary_score(
    env: RetailParallelEnv,
    cfg: RewardConfig,
    *,
    seed: int,
    predict: Any,
) -> float:
    """Mean per-seat weighted objective of one stationary episode for a predict fn."""
    observations, _infos = env.reset(seed=seed)
    per_seat_totals: dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
    horizon = DEFAULT_MAX_EPISODE_STEPS
    for _ in range(horizon):
        actions = {aid: predict(observations[aid]) for aid in env.agents}
        observations, _rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            components = _components_from_info(infos[aid])
            per_seat_totals[aid] += weighted_objective(components, cfg)
    return float(np.mean(list(per_seat_totals.values())))


def _components_from_info(info: dict[str, Any]) -> RewardComponents:
    """Build the RewardComponents from a per-seat step-info ``components`` payload."""
    raw = info["components"]
    return RewardComponents(
        profit=float(raw["profit"]),
        revenue=float(raw["revenue"]),
        market_share=float(raw["market_share"]),
        loyalty=float(raw["loyalty"]),
    )


def evaluate_shared_policy_stationary(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> list[float]:
    """Stationary mean episode scores of a trained shared policy over N episodes."""
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    cfg = config.reward
    env = RetailParallelEnv(config, n_learning_agents=n_learning_agents, seed=seed)

    def predict(obs: Any) -> Any:
        action, _state = model.predict(obs, deterministic=True)
        return action

    scores = [
        _episode_stationary_score(env, cfg, seed=seed + episode, predict=predict)
        for episode in range(episodes)
    ]
    env.close()
    return scores


def evaluate_random_stationary(
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> list[float]:
    """Stationary mean episode scores of a random policy over N episodes."""
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    cfg = config.reward
    env = RetailParallelEnv(config, n_learning_agents=n_learning_agents, seed=seed)
    action_space = env.action_space(env.possible_agents[0])
    action_space.seed(seed)

    def predict(_obs: Any) -> Any:
        return action_space.sample()

    scores = [
        _episode_stationary_score(env, cfg, seed=seed + episode, predict=predict)
        for episode in range(episodes)
    ]
    env.close()
    return scores


def _mixed_game_episode_scores(
    env: RetailParallelEnv,
    cfg: RewardConfig,
    *,
    seed: int,
    model: Any,
    trained_agent: str,
    action_space: Any,
) -> dict[str, float]:
    """One episode where ONLY ``trained_agent`` uses the trained policy; rest are random.

    The trained seat plays its deterministic ``predict``; every OTHER learning seat samples
    its action space (the random opponent). Returns ``{agent_id: Σ_t weighted_objective}``
    — the stationary score of EACH seat in this single shared game. This is the head-to-head
    primitive: trained and random compete in the SAME market, so the score difference is a
    real competitive-skill signal (unlike comparing two separate all-trained / all-random
    games on the absolute objective — see :func:`evaluate_head_to_head`).
    """
    observations, _infos = env.reset(seed=seed)
    totals: dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
        actions = {
            aid: (
                model.predict(observations[aid], deterministic=True)[0]
                if aid == trained_agent
                else action_space.sample()
            )
            for aid in env.agents
        }
        observations, _rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            totals[aid] += weighted_objective(_components_from_info(infos[aid]), cfg)
    return totals


def evaluate_head_to_head(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Head-to-head mean episode scores: the trained seat vs random seats per market."""
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    cfg = config.reward
    env = RetailParallelEnv(config, n_learning_agents=n_learning_agents, seed=seed)
    action_space = env.action_space(env.possible_agents[0])

    trained_scores: list[float] = []
    random_scores: list[float] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        action_space.seed(episode_seed + 5_000)
        trained_seat_vals: list[float] = []
        random_seat_vals: list[float] = []
        # The trained policy occupies each learning seat once (random fills the rest).
        for trained_agent in env.possible_agents:
            totals = _mixed_game_episode_scores(
                env,
                cfg,
                seed=episode_seed,
                model=model,
                trained_agent=trained_agent,
                action_space=action_space,
            )
            trained_seat_vals.append(totals[trained_agent])
            random_seat_vals.extend(v for aid, v in totals.items() if aid != trained_agent)
        trained_scores.append(float(np.mean(trained_seat_vals)))
        random_scores.append(float(np.mean(random_seat_vals)))
    env.close()
    return trained_scores, random_scores
