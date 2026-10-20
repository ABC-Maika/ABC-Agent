"""``RetailEnv`` — the Gymnasium single-agent view over the pure ``World`` seam.

A thin adapter: constructs one ``World``, threads the latest ``WorldState``
between calls, translates the agent's flat-Box action via
``envs.action_wrapper``, and presents the Gymnasium 5-tuple. All world logic
lives in ``core``.

Episode boundaries are not modeled here: ``terminated`` is always ``False``
(continuing world); apply ``TimeLimit`` externally via
gym.make("RetailSim-v0", max_episode_steps=...). The info dict carries DX keys
(awareness, marketing_spend, assortment, promotion, stockpile, ...), each
derived from the same action the seam read.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.envs.registration import register

from retail_simulator.core.config import CoreConfig
from retail_simulator.core.schema import ActionSchema, default_action_schema
from retail_simulator.core.world import AGENT_INDEX, World, WorldState, compute_expansion_mask
from retail_simulator.envs.action_wrapper import decode, mask_expansion_logits, to_core_action
from retail_simulator.envs.info import enrich_agent_info
from retail_simulator.envs.spaces import build_action_space, build_observation_space

# The registered Gymnasium environment id.
ENV_ID: str = "RetailSim-v0"


class RetailEnv(gym.Env[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    """Single-agent Gymnasium view: the learning agent vs a discounter NPC.

    ``reset`` builds and persists the initial ``WorldState``; ``step`` advances the
    seam by one tick with the agent seated at index 0.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        config: CoreConfig | None = None,
        *,
        seed: int | None = None,
        mask_expansion: bool = False,
    ) -> None:
        """Build the adapter around a ``World`` (config defaults to ``CoreConfig.default()``)."""
        super().__init__()
        self._config = config if config is not None else CoreConfig.default()
        self._world = World(config=self._config, seed=seed)
        # Use the default action schema so spaces/translation stay in lockstep with the seam.
        self._schema: ActionSchema = default_action_schema()
        self._mask_expansion = mask_expansion

        self.observation_space = build_observation_space(self._config)
        self.action_space = build_action_space(self._schema)

        # Current WorldState, owned by the env (the seam is stateless); None until reset().
        self._state: WorldState | None = None
        # Episode-step counter (core has no episode notion); 0 right after reset.
        self._episode_step: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        """Reset the world and return the initial ``(observation, info)``.

        Re-seeds the world's RNG when ``seed`` is given (same seed, same trajectory).
        """
        super().reset(seed=seed)
        self._state = self._world.reset(seed)
        self._episode_step = 0
        obs, info = self._world.agent_observation(self._state, AGENT_INDEX)
        return obs, self._enrich_info(
            info,
            self._state,
            marketing_spend=0.0,
            assortment=0.0,
            promotion=0.0,
            expansion_action=0.0,
            research_action=0.0,
            # At reset no action has been taken: the SCM lever defaults to 1.0.
            service_level_action=1.0,
            prior_service_score_per_region=None,
            # No automation action at reset: the default tier 0.
            automation_choice=0.0,
            loyalty_spend_action=0.0,
            # No wage/warehouse action at reset: the registry default 0.0.
            wage_spend_action=0.0,
            warehouse_invest_action=0.0,
        )

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Advance one tick with the agent acting; return the Gym 5-tuple.

        ``terminated`` and ``truncated`` are always ``False`` here (apply ``TimeLimit``
        externally).
        """
        if self._state is None:
            raise RuntimeError("step() called before reset(); call reset() first")

        flat_action = to_core_action(action, self._schema)
        if self._mask_expansion:
            # Mask off the PRE-step state — the mask the decision actually faces (not
            # the seam's next-state mask).
            expansion_mask = compute_expansion_mask(self._state, AGENT_INDEX, self._config)
            flat_action = mask_expansion_logits(flat_action, expansion_mask, self._schema)
        # Capture the prior per-region service_score to recover this tick's stockout rate.
        prior_service_score = self._state.retailers[AGENT_INDEX].service_score_per_region
        # Capture the prior seated tier to recover the differential capex charged this tick.
        prior_automation_tier = int(self._state.retailers[AGENT_INDEX].automation_tier)
        result = self._world.step(self._state, {AGENT_INDEX: flat_action})
        self._state = result.next_state
        self._episode_step += 1

        # Decode the action once; the info keys below are all "the action taken this tick".
        levers = decode(flat_action, self._schema).levers
        # Marketing charged this tick = decoded spend * cost_per_unit_spend
        # (the same quantity the seam charges).
        spend_fraction = levers.get("marketing", 0.0)
        marketing_exponent = self._config.marketing.marketing_cost_exponent
        if marketing_exponent == 1.0:
            marketing_spend = spend_fraction * self._config.marketing.cost_per_unit_spend
        else:
            marketing_spend = (
                spend_fraction**marketing_exponent * self._config.marketing.cost_per_unit_spend
            )
        # Assortment breadth set this tick = the decoded action.
        assortment = levers.get("assortment", 0.0)
        # Promotion intensity set this tick = the decoded action.
        promotion = levers.get("promotion", 0.0)
        # Expansion choice decoded this tick (0 = no-op, r = open region r).
        expansion_action = levers.get("expansion", 0.0)
        # Research spend decoded this tick; the resulting fidelity is seated state,
        # read off the next state.
        research_action = levers.get("research", 0.0)
        # Decoded SCM service_level this tick; 1.0 is the default baseline.
        service_level_action = levers.get("service_level", 1.0)
        # Decoded TARGET automation tier this tick; the seated tier after the gate
        # is the seam's authority.
        automation_choice = levers.get("automation", 0.0)
        loyalty_spend_action = levers.get("loyalty_spend", 0.0)
        # Decoded wage/warehouse actions this tick; 0.0 is the registry default.
        wage_spend_action = levers.get("wage_spend", 0.0)
        warehouse_invest_action = levers.get("warehouse_invest", 0.0)

        obs = np.asarray(result.observations[AGENT_INDEX], dtype=np.float32)
        reward = float(result.rewards[AGENT_INDEX])
        terminated = bool(result.terminated[AGENT_INDEX])
        truncated = bool(result.truncated[AGENT_INDEX])
        # base_info already carries the seam's action_mask for next_state.
        info = self._enrich_info(
            result.infos[AGENT_INDEX],
            self._state,
            marketing_spend,
            assortment,
            promotion,
            expansion_action,
            research_action,
            service_level_action=service_level_action,
            prior_service_score_per_region=prior_service_score,
            automation_choice=automation_choice,
            prior_automation_tier=prior_automation_tier,
            loyalty_spend_action=loyalty_spend_action,
            wage_spend_action=wage_spend_action,
            warehouse_invest_action=warehouse_invest_action,
        )
        return obs, reward, terminated, truncated, info

    def _enrich_info(
        self,
        base_info: dict[str, Any],
        state: WorldState,
        marketing_spend: float,
        assortment: float,
        promotion: float,
        expansion_action: float,
        research_action: float,
        *,
        service_level_action: float = 1.0,
        prior_service_score_per_region: tuple[float, ...] | None = None,
        automation_choice: float = 0.0,
        prior_automation_tier: int | None = None,
        loyalty_spend_action: float = 0.0,
        wage_spend_action: float = 0.0,
        warehouse_invest_action: float = 0.0,
    ) -> dict[str, Any]:
        """Return the enriched info dict for the agent seat (see ``envs.info``)."""
        return enrich_agent_info(
            base_info,
            state,
            AGENT_INDEX,
            self._episode_step,
            marketing_spend=marketing_spend,
            assortment=assortment,
            promotion=promotion,
            expansion_action=expansion_action,
            research_action=research_action,
            service_level_action=service_level_action,
            prior_service_score_per_region=prior_service_score_per_region,
            config=self._config,
            automation_choice=automation_choice,
            prior_automation_tier=prior_automation_tier,
            loyalty_spend_action=loyalty_spend_action,
            wage_spend_action=wage_spend_action,
            warehouse_invest_action=warehouse_invest_action,
        )


def register_env() -> None:
    """Register ``RetailSim-v0`` with Gymnasium (idempotent).

    ``max_episode_steps`` is left unset on purpose: truncation is a caller concern
    (``TimeLimit`` via ``gym.make``). Re-registration is skipped so repeated imports
    do not warn.
    """
    if ENV_ID in gym.registry:
        return
    register(id=ENV_ID, entry_point="retail_simulator.envs.gym_env:RetailEnv")
