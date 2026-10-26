"""On-vs-off learnability gates for the research, SCM, automation, and loyalty levers.

Each gate compares a seat playing its lever ON against every OTHER seat forced to the
lever's off/default value, rotating the ON role across all learning seats in one market.
Heavy env import stays lazy so ``import retail_simulator.envs.parallel_gate`` is cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import CoreConfig, RewardConfig
from retail_simulator.core.reward import weighted_objective
from retail_simulator.envs.learnability import DEFAULT_EVAL_EPISODES, DEFAULT_MAX_EPISODE_STEPS
from retail_simulator.envs.sweep import RESEARCH_OPPONENT_ARCHETYPE

from .bridge import (
    _AUTOMATION_LEVER_NAME,
    _LOYALTY_LEVER_NAME,
    _RESEARCH_LEVER_NAME,
    _SCM_LEVER_NAME,
    DEFAULT_N_LEARNING_AGENTS,
)
from .stationary import _components_from_info

if TYPE_CHECKING:  # type-only; never pulled in at runtime import time
    from retail_simulator.envs.parallel_env import RetailParallelEnv


def _research_offset(schema: Any) -> int:
    """The flat-action offset of the ``research`` lever (read from the schema layout).

    The force-off mechanism pins exactly this dim to 0. Reading the offset from
    ``flatten_action_space_layout`` (the layout authority) rather than hardcoding 6
    keeps the force-off correct if the action layout ever changes.
    """
    from retail_simulator.core.encoding import flatten_action_space_layout

    layout = flatten_action_space_layout(schema)
    for slot in layout.levers:
        if slot.spec.name == _RESEARCH_LEVER_NAME:
            return slot.offset
    raise ValueError(f"no {_RESEARCH_LEVER_NAME!r} lever in the action schema")


def research_force_off_action(
    action: npt.NDArray[np.float32], research_offset: int
) -> npt.NDArray[np.float32]:
    """Copy an action and pin the research lever's dim to 0 (research forced off)."""
    forced = np.array(action, dtype=np.float32).copy()
    forced[research_offset] = 0.0
    return forced


def scm_force_default_action(
    action: npt.NDArray[np.float32],
    scm_offset: int,
    scm_default: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Copy an action and pin the service_level lever's dim to its registry default."""
    forced = np.array(action, dtype=np.float32).copy()
    forced[scm_offset] = scm_default
    return forced


def _scm_offset_and_default(schema: Any) -> tuple[int, float]:
    """The flat offset + the REGISTRY DEFAULT of the ``service_level`` lever.

    Reading both the offset AND the registry default from
    ``flatten_action_space_layout`` (the layout authority) rather than hardcoding 7
    and 1.0 keeps the force-default correct if the action layout or the registry
    default ever change.
    """
    from retail_simulator.core.encoding import flatten_action_space_layout

    layout = flatten_action_space_layout(schema)
    for slot in layout.levers:
        if slot.spec.name == _SCM_LEVER_NAME:
            return slot.offset, float(slot.spec.default)
    raise ValueError(f"no {_SCM_LEVER_NAME!r} lever in the action schema")


def _research_on_off_episode_scores(
    env: RetailParallelEnv,
    cfg: RewardConfig,
    *,
    seed: int,
    model: Any,
    research_on_agent: str,
    research_offset: int,
) -> dict[str, float]:
    """Per-seat totals for one episode where only research_on_agent has research on."""
    observations, _infos = env.reset(seed=seed)
    totals: dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
        actions: dict[str, Any] = {}
        for aid in env.agents:
            base_action, _state = model.predict(observations[aid], deterministic=True)
            if aid == research_on_agent:
                actions[aid] = base_action
            else:
                actions[aid] = research_force_off_action(base_action, research_offset)
        observations, _rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            totals[aid] += weighted_objective(_components_from_info(infos[aid]), cfg)
    return totals


def evaluate_research_on_vs_off(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = (RESEARCH_OPPONENT_ARCHETYPE,),
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Mean episode scores: research-on seats vs research-off (forced) seats per run."""
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    cfg = config.reward
    if n_learning_agents < 2:
        raise ValueError(
            f"evaluate_research_on_vs_off requires n_learning_agents >= 2 (got "
            f"{n_learning_agents}): the position-controlled rotation puts every OTHER "
            "seat in the research-off role, which is empty (⇒ silent NaN mean) at "
            "n=1. For a true single-agent research gate, run two separate rollouts "
            "(same trained model, action unchanged vs research_force_off_action'd) "
            "and compare — that path is the orchestrator's runner, not this function."
        )
    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )
    research_offset = _research_offset(env._schema)  # the env's action schema

    research_on_scores: list[float] = []
    research_off_scores: list[float] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        on_seat_vals: list[float] = []
        off_seat_vals: list[float] = []
        # The research-ON role occupies each learning seat once (rest forced off).
        for research_on_agent in env.possible_agents:
            totals = _research_on_off_episode_scores(
                env,
                cfg,
                seed=episode_seed,
                model=model,
                research_on_agent=research_on_agent,
                research_offset=research_offset,
            )
            on_seat_vals.append(totals[research_on_agent])
            off_seat_vals.extend(v for aid, v in totals.items() if aid != research_on_agent)
        research_on_scores.append(float(np.mean(on_seat_vals)))
        research_off_scores.append(float(np.mean(off_seat_vals)))
    env.close()
    return research_on_scores, research_off_scores


def _scm_on_off_episode_scores(
    env: RetailParallelEnv,
    cfg: RewardConfig,
    *,
    seed: int,
    model: Any,
    scm_on_agent: str,
    scm_offset: int,
    scm_default: float,
) -> dict[str, float]:
    """Per-seat totals for one episode where only scm_on_agent has SCM on (rest default)."""
    observations, _infos = env.reset(seed=seed)
    totals: dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
        actions: dict[str, Any] = {}
        for aid in env.agents:
            base_action, _state = model.predict(observations[aid], deterministic=True)
            if aid == scm_on_agent:
                actions[aid] = base_action
            else:
                actions[aid] = scm_force_default_action(base_action, scm_offset, scm_default)
        observations, _rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            totals[aid] += weighted_objective(_components_from_info(infos[aid]), cfg)
    return totals


def evaluate_scm_on_vs_off(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = ("discounter",),
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Mean episode scores: SCM-on seats vs SCM-off (forced default) seats per run."""
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    if n_learning_agents < 2:
        raise ValueError(
            f"""evaluate_scm_on_vs_off requires n_learning_agents >= 2 (got {n_learning_agents}): the position-controlled rotation puts every OTHER seat in the scm-off role, which is empty (⇒ silent NaN mean) at n=1. The ideal substrate is single-agent vs Discounter; the current implementation uses n=2 + Discounter as a close equivalent (SCM is first-order competitor-independent). A literal n=1 path would compare two separate rollouts (same trained model, action unchanged vs scm_force_default_action'd) — that is a separate follow-up runner, not this function."""
        )
    cfg = config.reward
    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )
    scm_offset, scm_default = _scm_offset_and_default(env._schema)

    scm_on_scores: list[float] = []
    scm_off_scores: list[float] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        on_seat_vals: list[float] = []
        off_seat_vals: list[float] = []
        # The SCM-ON role occupies each learning seat once (rest forced to default).
        for scm_on_agent in env.possible_agents:
            totals = _scm_on_off_episode_scores(
                env,
                cfg,
                seed=episode_seed,
                model=model,
                scm_on_agent=scm_on_agent,
                scm_offset=scm_offset,
                scm_default=scm_default,
            )
            on_seat_vals.append(totals[scm_on_agent])
            off_seat_vals.extend(v for aid, v in totals.items() if aid != scm_on_agent)
        scm_on_scores.append(float(np.mean(on_seat_vals)))
        scm_off_scores.append(float(np.mean(off_seat_vals)))
    env.close()
    return scm_on_scores, scm_off_scores


def automation_force_default_action(
    action: npt.NDArray[np.float32], automation_offset: int, n_tiers: int
) -> npt.NDArray[np.float32]:
    """Copy an action and pin the whole automation block to tier 0 (forced default)."""
    forced = np.array(action, dtype=np.float32).copy()
    # Zero the whole 3-wide block, then write a one-hot at the tier-0 slot — the
    # argmax-decoded value is 0 regardless of the surrounding action's magnitude.
    forced[automation_offset : automation_offset + n_tiers] = 0.0
    forced[automation_offset] = 1.0
    return forced


def _automation_offset_and_width(schema: Any) -> tuple[int, int]:
    """The flat offset + the WIDTH of the ``automation`` lever (read from the schema).

    Reading the offset AND the width from ``flatten_action_space_layout`` (the layout
    authority) rather than hardcoding 8 and 3 keeps the force-default correct if the
    action layout ever changes. The default automation lever's width equals
    :data:`retail_simulator.core.schema.N_AUTOMATION_TIERS`.
    """
    from retail_simulator.core.encoding import flatten_action_space_layout

    layout = flatten_action_space_layout(schema)
    for slot in layout.levers:
        if slot.spec.name == _AUTOMATION_LEVER_NAME:
            return slot.offset, slot.width
    raise ValueError(f"no {_AUTOMATION_LEVER_NAME!r} lever in the action schema")


def _automation_on_off_episode_scores(
    env: RetailParallelEnv,
    cfg: RewardConfig,
    *,
    seed: int,
    model: Any,
    automation_on_agent: str,
    automation_offset: int,
    n_tiers: int,
) -> dict[str, float]:
    """Per-seat totals for one episode where only automation_on_agent has automation on."""
    observations, _infos = env.reset(seed=seed)
    totals: dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
        actions: dict[str, Any] = {}
        for aid in env.agents:
            base_action, _state = model.predict(observations[aid], deterministic=True)
            if aid == automation_on_agent:
                actions[aid] = base_action
            else:
                actions[aid] = automation_force_default_action(
                    base_action, automation_offset, n_tiers
                )
        observations, _rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            totals[aid] += weighted_objective(_components_from_info(infos[aid]), cfg)
    return totals


def evaluate_automation_on_vs_off(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = ("discounter",),
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Mean episode scores: automation-on seats vs automation-off (tier 0) seats per run."""
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    if n_learning_agents < 2:
        raise ValueError(
            f"""evaluate_automation_on_vs_off requires n_learning_agents >= 2 (got {n_learning_agents}): the position-controlled rotation puts every OTHER seat in the automation-off role, which is empty (⇒ silent NaN mean) at n=1. The ideal substrate is single-agent vs Discounter; the current implementation uses n=2 + Discounter as a close equivalent (automation is first-order competitor-independent, — identical to SCM's structure). A literal n=1 path would compare two separate rollouts (same trained model, action unchanged vs automation_force_default_action'd) — that is a separate follow-up runner, not this function."""
        )
    cfg = config.reward
    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )
    automation_offset, n_tiers = _automation_offset_and_width(env._schema)

    automation_on_scores: list[float] = []
    automation_off_scores: list[float] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        on_seat_vals: list[float] = []
        off_seat_vals: list[float] = []
        # The automation-ON role occupies each learning seat once (rest forced to tier 0).
        for automation_on_agent in env.possible_agents:
            totals = _automation_on_off_episode_scores(
                env,
                cfg,
                seed=episode_seed,
                model=model,
                automation_on_agent=automation_on_agent,
                automation_offset=automation_offset,
                n_tiers=n_tiers,
            )
            on_seat_vals.append(totals[automation_on_agent])
            off_seat_vals.extend(v for aid, v in totals.items() if aid != automation_on_agent)
        automation_on_scores.append(float(np.mean(on_seat_vals)))
        automation_off_scores.append(float(np.mean(off_seat_vals)))
    env.close()
    return automation_on_scores, automation_off_scores


def loyalty_force_default_action(
    action: npt.NDArray[np.float32],
    loyalty_offset: int,
    loyalty_default: float = 0.0,
) -> npt.NDArray[np.float32]:
    """Copy an action and pin the loyalty_spend lever's dim to its default (0.0)."""
    forced = np.array(action, dtype=np.float32).copy()
    forced[loyalty_offset] = loyalty_default
    return forced


def _loyalty_offset_and_default(schema: Any) -> tuple[int, float]:
    """The flat offset + the REGISTRY DEFAULT of the ``loyalty_spend`` lever.

    Reading both the offset AND the registry default from
    ``flatten_action_space_layout`` (the layout authority) rather than hardcoding
    11 and 0.0 keeps the force-default correct if the action layout or the
    registry default ever change.
    """
    from retail_simulator.core.encoding import flatten_action_space_layout

    layout = flatten_action_space_layout(schema)
    for slot in layout.levers:
        if slot.spec.name == _LOYALTY_LEVER_NAME:
            return slot.offset, float(slot.spec.default)
    raise ValueError(f"no {_LOYALTY_LEVER_NAME!r} lever in the action schema")


def _loyalty_on_off_episode_scores(
    env: RetailParallelEnv,
    cfg: RewardConfig,
    *,
    seed: int,
    model: Any,
    loyalty_on_agent: str,
    loyalty_offset: int,
    loyalty_default: float,
) -> dict[str, float]:
    """Per-seat totals for one episode where only loyalty_on_agent has loyalty on."""
    observations, _infos = env.reset(seed=seed)
    totals: dict[str, float] = {aid: 0.0 for aid in env.possible_agents}
    for _ in range(DEFAULT_MAX_EPISODE_STEPS):
        actions: dict[str, Any] = {}
        for aid in env.agents:
            base_action, _state = model.predict(observations[aid], deterministic=True)
            if aid == loyalty_on_agent:
                actions[aid] = base_action
            else:
                actions[aid] = loyalty_force_default_action(
                    base_action, loyalty_offset, loyalty_default
                )
        observations, _rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            totals[aid] += weighted_objective(_components_from_info(infos[aid]), cfg)
    return totals


def evaluate_loyalty_on_vs_off(
    model: Any,
    config: CoreConfig,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = ("discounter",),
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> tuple[list[float], list[float]]:
    """Mean episode scores: loyalty-on seats vs loyalty-off (0.0) seats per run."""
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    if n_learning_agents < 2:
        raise ValueError(
            f"""evaluate_loyalty_on_vs_off requires n_learning_agents >= 2 (got {n_learning_agents}): the position-controlled rotation puts every OTHER seat in the loyalty-off role, which is empty (⇒ silent NaN mean) at n=1. The ideal substrate is single-agent vs Discounter; the current implementation uses n=2 + Discounter as a close equivalent (loyalty is first-order competitor-independent, — identical to SCM / automation). A literal n=1 path would compare two separate rollouts (same trained model, action unchanged vs loyalty_force_default_action'd) — that is a separate follow-up runner, not this function."""
        )
    cfg = config.reward
    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )
    loyalty_offset, loyalty_default = _loyalty_offset_and_default(env._schema)

    loyalty_on_scores: list[float] = []
    loyalty_off_scores: list[float] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        on_seat_vals: list[float] = []
        off_seat_vals: list[float] = []
        # The loyalty-ON role occupies each learning seat once (rest forced to 0.0).
        for loyalty_on_agent in env.possible_agents:
            totals = _loyalty_on_off_episode_scores(
                env,
                cfg,
                seed=episode_seed,
                model=model,
                loyalty_on_agent=loyalty_on_agent,
                loyalty_offset=loyalty_offset,
                loyalty_default=loyalty_default,
            )
            on_seat_vals.append(totals[loyalty_on_agent])
            off_seat_vals.extend(v for aid, v in totals.items() if aid != loyalty_on_agent)
        loyalty_on_scores.append(float(np.mean(on_seat_vals)))
        loyalty_off_scores.append(float(np.mean(off_seat_vals)))
    env.close()
    return loyalty_on_scores, loyalty_off_scores
