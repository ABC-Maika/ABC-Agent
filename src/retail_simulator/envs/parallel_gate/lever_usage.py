"""Lever-usage / uptake diagnostics: how strongly a policy drives each lever in a rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from retail_simulator.core.config import CoreConfig
from retail_simulator.envs.learnability import DEFAULT_EVAL_EPISODES, DEFAULT_MAX_EPISODE_STEPS

from .bridge import (
    _AUTOMATION_LEVER_NAME,
    _LOYALTY_LEVER_NAME,
    _SCM_LEVER_NAME,
    DEFAULT_N_LEARNING_AGENTS,
)
from .lever_eval import (
    _automation_offset_and_width,
    _loyalty_offset_and_default,
    _scm_offset_and_default,
)


@dataclass(frozen=True)
class SCMLeverUsage:
    """Aggregated service_level lever-usage stats across a rollout."""

    mean: float
    std: float
    uses_lever: bool
    default: float
    n_samples: int


USAGE_MARGIN_MEAN: float = 0.1
USAGE_MARGIN_STD: float = 0.05


def measure_scm_lever_usage(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = ("discounter",),
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> SCMLeverUsage:
    """Measure how strongly a policy drives the service_level lever from its default."""
    from retail_simulator.envs.gym_env import decode
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )
    _scm_offset, scm_default = _scm_offset_and_default(env._schema)

    service_levels: list[float] = []
    for episode in range(episodes):
        observations, _infos = env.reset(seed=seed + episode)
        for _ in range(DEFAULT_MAX_EPISODE_STEPS):
            actions: dict[str, Any] = {}
            for aid in env.agents:
                base_action, _state = model.predict(observations[aid], deterministic=True)
                actions[aid] = base_action
                # The decoded service_level (post-clip, the SAME value the seam reads
                # for the fill-rate cap + per-retailer cogs_fraction) — the lever's
                # USE signal, not the raw network output.
                decoded = decode(np.asarray(base_action, dtype=np.float32), env._schema).levers
                service_levels.append(float(decoded.get(_SCM_LEVER_NAME, scm_default)))
            observations, _rewards, _term, _trunc, _infos = env.step(actions)
    env.close()

    arr = np.asarray(service_levels, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else float(scm_default)
    std = float(arr.std()) if arr.size > 1 else 0.0
    uses_lever = abs(mean - scm_default) > USAGE_MARGIN_MEAN and std > USAGE_MARGIN_STD
    return SCMLeverUsage(
        mean=mean,
        std=std,
        uses_lever=uses_lever,
        default=scm_default,
        n_samples=int(arr.size),
    )


AUTOMATION_UPTAKE_TAU: float = 0.5


@dataclass(frozen=True)
class AutomationUptake:
    """Aggregated automation tier-uptake + tier-distribution stats across a rollout."""

    uptake_fraction: float
    tier_distribution: tuple[float, ...]
    uses_lever: bool
    n_tiers: int
    n_samples: int
    n_episodes: int


def measure_automation_uptake(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = ("discounter",),
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
    uptake_tau: float = AUTOMATION_UPTAKE_TAU,
) -> AutomationUptake:
    """Measure automation tier uptake + tier distribution for a policy across a rollout."""
    from retail_simulator.envs.gym_env import decode
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )
    _automation_offset, n_tiers = _automation_offset_and_width(env._schema)

    decoded_tiers: list[int] = []
    # Per-episode end-of-episode seated tier (across learning seats). A seat-episode
    # counts as "reached tier >= 1" if its FINAL seated automation_tier is >= 1
    # (the monotonicity guarantee means this is true iff any upgrade ever fired).
    seat_episode_outcomes: list[bool] = []
    for episode in range(episodes):
        observations, _infos = env.reset(seed=seed + episode)
        episode_tiers: list[int] = []
        final_seats: dict[str, int] = {aid: 0 for aid in env.possible_agents}
        for _ in range(DEFAULT_MAX_EPISODE_STEPS):
            actions: dict[str, Any] = {}
            for aid in env.agents:
                base_action, _state = model.predict(observations[aid], deterministic=True)
                actions[aid] = base_action
                decoded = decode(np.asarray(base_action, dtype=np.float32), env._schema).levers
                # The DECODED target tier — diagnostic over the rollout for the
                # tier distribution channel.
                episode_tiers.append(int(decoded.get(_AUTOMATION_LEVER_NAME, 0.0)))
            observations, _rewards, _term, _trunc, infos = env.step(actions)
            # Track each learning seat's CURRENT seated tier (monotone non-decreasing
            # — only the last value matters for the end-of-episode check; we overwrite
            # rather than accumulate so we always carry the latest seated value).
            for aid in env.agents:
                final_seats[aid] = int(infos[aid].get("automation_tier", 0))
        decoded_tiers.extend(episode_tiers)
        # One outcome per learning seat × episode: True iff the final seated tier is
        # >= 1 (i.e. some upgrade fired by end of episode under monotonicity).
        for aid in env.possible_agents:
            seat_episode_outcomes.append(final_seats[aid] >= 1)
    env.close()

    arr = np.asarray(decoded_tiers, dtype=np.int64)
    n_samples = int(arr.size)
    uptake_fraction = (
        float(sum(seat_episode_outcomes) / len(seat_episode_outcomes))
        if seat_episode_outcomes
        else 0.0
    )
    if n_samples > 0:
        counts = np.bincount(arr, minlength=n_tiers)
        tier_distribution = tuple(float(c) / n_samples for c in counts[:n_tiers])
    else:
        tier_distribution = tuple(0.0 for _ in range(n_tiers))
    uses_lever = uptake_fraction > uptake_tau
    return AutomationUptake(
        uptake_fraction=uptake_fraction,
        tier_distribution=tier_distribution,
        uses_lever=uses_lever,
        n_tiers=n_tiers,
        n_samples=n_samples,
        n_episodes=int(episodes),
    )


LOYALTY_USAGE_MARGIN_MEAN: float = 0.1
LOYALTY_USAGE_MARGIN_STD: float = 0.05


@dataclass(frozen=True)
class LoyaltyLeverUsage:
    """Aggregated loyalty_spend lever-usage stats across a rollout."""

    mean: float
    std: float
    uses_lever: bool
    default: float
    n_samples: int


def measure_loyalty_lever_usage(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = ("discounter",),
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
    usage_margin_mean: float = LOYALTY_USAGE_MARGIN_MEAN,
    usage_margin_std: float = LOYALTY_USAGE_MARGIN_STD,
) -> LoyaltyLeverUsage:
    """Measure how strongly a policy drives the loyalty_spend lever across a rollout."""
    from retail_simulator.envs.gym_env import decode
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )
    _loyalty_offset, loyalty_default = _loyalty_offset_and_default(env._schema)

    loyalty_spends: list[float] = []
    for episode in range(episodes):
        observations, _infos = env.reset(seed=seed + episode)
        for _ in range(DEFAULT_MAX_EPISODE_STEPS):
            actions: dict[str, Any] = {}
            for aid in env.agents:
                base_action, _state = model.predict(observations[aid], deterministic=True)
                actions[aid] = base_action
                # The decoded loyalty_spend (post-clip, the SAME value the seam
                # reads for the same-tick brand-loyal utility boost + the
                # per-tick opex line) — the lever's USE signal, not the raw
                # network output.
                decoded = decode(np.asarray(base_action, dtype=np.float32), env._schema).levers
                loyalty_spends.append(float(decoded.get(_LOYALTY_LEVER_NAME, loyalty_default)))
            observations, _rewards, _term, _trunc, _infos = env.step(actions)
    env.close()

    arr = np.asarray(loyalty_spends, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else float(loyalty_default)
    std = float(arr.std()) if arr.size > 1 else 0.0
    uses_lever = mean > usage_margin_mean and std > usage_margin_std
    return LoyaltyLeverUsage(
        mean=mean,
        std=std,
        uses_lever=uses_lever,
        default=loyalty_default,
        n_samples=int(arr.size),
    )
