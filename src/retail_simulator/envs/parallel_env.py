"""``RetailParallelEnv`` — the PettingZoo Parallel multi-agent view over the seam.

The default construction (``n_learning_agents=2``, ``npc_archetypes=()``) is a
2-agent all-learning duopoly; scripted NPCs fill extra seats only when
``npc_archetypes`` is non-empty. All world logic comes from ``core``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from pettingzoo import ParallelEnv

from retail_simulator.core.config import CoreConfig, SeatSpec
from retail_simulator.core.schema import ActionSchema, default_action_schema
from retail_simulator.core.world import World, WorldState, compute_expansion_mask
from retail_simulator.envs.gym_env import decode, mask_expansion_logits, to_core_action
from retail_simulator.envs.info import enrich_agent_info
from retail_simulator.envs.spaces import build_action_space, build_observation_space


def _agent_id(seat_index: int) -> str:
    """The PettingZoo agent id for a learning seat (the stable ordered name)."""
    return f"retailer_{seat_index}"


def _one_hot_presence(home: int, n_regions: int) -> tuple[int, ...]:
    """The one-hot presence tuple for a home region (1 at home, 0 elsewhere)."""
    return tuple(1 if r == home else 0 for r in range(n_regions))


class RetailParallelEnv(ParallelEnv):
    """PettingZoo Parallel multi-agent env: N learning agents over the shared seam.

    Spaces are homogeneous across learning agents (the same per-agent Box the Gym
    env uses). ``reset``/``step`` return per-agent dicts keyed by
    ``possible_agents``; ``infos[agent]`` carries the real ``action_mask``, the
    un-normalized reward ``components``, and DX flat metric keys.
    """

    metadata: ClassVar[dict[str, Any]] = {"name": "retail_sim_parallel_v0", "render_modes": []}

    def __init__(
        self,
        config: CoreConfig | None = None,
        *,
        n_learning_agents: int = 2,
        npc_archetypes: tuple[str, ...] = (),
        home_regions: tuple[int, ...] | None = None,
        seed: int | None = None,
        mask_expansion_seats: tuple[int, ...] = (),
    ) -> None:
        """Build the env; validate seats and home regions, build the seat plan, wire homogeneous spaces."""
        if n_learning_agents < 1:
            raise ValueError(f"n_learning_agents must be >= 1, got {n_learning_agents}")
        for seat in mask_expansion_seats:
            if (
                not isinstance(seat, (int, np.integer))
                or isinstance(seat, bool)
                or not (0 <= seat < n_learning_agents)
            ):
                raise ValueError(
                    f"mask_expansion_seats entry {seat!r} (type {type(seat).__name__}) is "
                    "invalid; must be an integer LEARNING seat index satisfying "
                    f"0 <= seat < n_learning_agents ({n_learning_agents})"
                )

        base_config = config if config is not None else CoreConfig.default()
        region_cfgs = base_config.demand.regions
        n_regions = len(region_cfgs) if region_cfgs else 1
        seat_plan = self._build_seat_plan(
            n_learning_agents, npc_archetypes, n_regions, home_regions
        )
        # Inject the seat plan into the config (additive); every other config field is kept.
        self._config = replace(base_config, seats=seat_plan)
        self._schema: ActionSchema = default_action_schema()
        self._world = World(config=self._config, seed=seed)

        self._n_learning_agents = n_learning_agents
        self.possible_agents: list[str] = [_agent_id(i) for i in range(n_learning_agents)]
        self._seat_of: dict[str, int] = {_agent_id(i): i for i in range(n_learning_agents)}
        self.agents: list[str] = list(self.possible_agents)
        # Learning seats whose decode gets illegal-expansion masking (validated above).
        self._mask_expansion_seats: tuple[int, ...] = mask_expansion_seats

        # Homogeneous per-agent Boxes (same spaces the Gym env builds); expose both
        # method and dict forms for bridge compatibility.
        self._observation_space: gym.spaces.Box = build_observation_space(self._config)
        self._action_space: gym.spaces.Box = build_action_space(self._schema)
        self.observation_spaces: dict[str, gym.spaces.Box] = {
            aid: self._observation_space for aid in self.possible_agents
        }
        self.action_spaces: dict[str, gym.spaces.Box] = {
            aid: self._action_space for aid in self.possible_agents
        }

        # Current WorldState, owned by the env; None until reset().
        self._state: WorldState | None = None
        # Episode-step counter (core has no episode notion); 0 right after reset.
        self._episode_step: int = 0
        # Raw per-seat infos from the last World.step (ALL seats, incl. NPCs) — the
        # seam's un-normalized reward components; hot-seat scoring reads NPC seats
        # from here. None until the first step.
        self._last_raw_infos: dict[int, dict[str, Any]] | None = None

    @staticmethod
    def _build_seat_plan(
        n_learning_agents: int,
        npc_archetypes: tuple[str, ...],
        n_regions: int,
        home_regions: tuple[int, ...] | None = None,
    ) -> tuple[SeatSpec, ...]:
        """Build the ordered seat plan (learning seats first, then NPCs), validating ``home_regions``."""
        n_seats = n_learning_agents + len(npc_archetypes)
        if home_regions is not None:
            if len(home_regions) != n_seats:
                raise ValueError(
                    f"home_regions must have exactly {n_seats} entries (one per seat: "
                    f"{n_learning_agents} learning + {len(npc_archetypes)} NPC), got "
                    f"{len(home_regions)}"
                )
            for seat_index, home in enumerate(home_regions):
                if (
                    not isinstance(home, (int, np.integer))
                    or isinstance(home, bool)
                    or not (0 <= home < n_regions)
                ):
                    raise ValueError(
                        f"home_regions[{seat_index}]={home!r} (type "
                        f"{type(home).__name__}) is invalid; must be an integer region "
                        f"index satisfying 0 <= r < n_regions ({n_regions})"
                    )
            seats: list[SeatSpec] = [
                SeatSpec(is_npc=False, presence=_one_hot_presence(home_regions[i], n_regions))
                for i in range(n_learning_agents)
            ]
            seats.extend(
                SeatSpec(
                    is_npc=True,
                    archetype=name,
                    presence=_one_hot_presence(home_regions[n_learning_agents + i], n_regions),
                )
                for i, name in enumerate(npc_archetypes)
            )
            return tuple(seats)

        cover_with_seat0 = len(npc_archetypes) == 0
        learning: list[SeatSpec] = []
        for i in range(n_learning_agents):
            presence = (1,) * n_regions if (i == 0 and cover_with_seat0) else None
            learning.append(SeatSpec(is_npc=False, presence=presence))
        npc = tuple(SeatSpec(is_npc=True, archetype=name) for name in npc_archetypes)
        return tuple(learning) + npc

    def observation_space(self, agent: str) -> gym.spaces.Box:
        """The homogeneous observation Box for an agent (PettingZoo requires a method)."""
        return self._observation_space

    def action_space(self, agent: str) -> gym.spaces.Box:
        """The homogeneous action Box for an agent (PettingZoo requires a method)."""
        return self._action_space

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, dict[str, Any]]]:
        """Reset the world; return per-learning-agent ``(observations, infos)`` dicts.

        Re-seeds the world's RNG when ``seed`` is given. Per-lever DX keys are 0.0 (no
        action at reset); each ``action_mask`` comes from the seam.
        """
        self._state = self._world.reset(seed)
        self._episode_step = 0
        self.agents = list(self.possible_agents)

        observations: dict[str, npt.NDArray[np.float32]] = {}
        infos: dict[str, dict[str, Any]] = {}
        for aid in self.possible_agents:
            seat = self._seat_of[aid]
            obs, base_info = self._world.agent_observation(self._state, seat)
            observations[aid] = np.asarray(obs, dtype=np.float32)
            infos[aid] = enrich_agent_info(
                base_info,
                self._state,
                seat,
                self._episode_step,
                marketing_spend=0.0,
                assortment=0.0,
                promotion=0.0,
                expansion_action=0.0,
                research_action=0.0,
                # At reset no action has been taken: the SCM lever defaults to 1.0.
                service_level_action=1.0,
                prior_service_score_per_region=None,
                config=self._config,
                # No automation action at reset: tier 0.
                automation_choice=0.0,
                loyalty_spend_action=0.0,
                # No wage/warehouse action at reset: the registry default 0.0.
                wage_spend_action=0.0,
                warehouse_invest_action=0.0,
            )
        return observations, infos

    def step(
        self, actions: dict[str, npt.NDArray[np.float32]]
    ) -> tuple[
        dict[str, npt.NDArray[np.float32]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Advance one tick for all learning agents; return per-agent dicts.

        ``terminated``/``truncated`` are always ``False`` (no episode notion in core).
        """
        if self._state is None:
            raise RuntimeError("step() called before reset(); call reset() first")

        # 1. Build the joint action for the learning seats (the seam fills NPCs);
        #    decode each action once for the per-agent DX info derivations below.
        joint_action: dict[int, npt.NDArray[np.float32]] = {}
        agent_levers: dict[str, dict[str, float]] = {}
        for aid in self.possible_agents:
            seat = self._seat_of[aid]
            flat_action = to_core_action(actions[aid], self._schema)
            if seat in self._mask_expansion_seats:
                # Mask this seat's flat action off the PRE-step state (the mask the
                # decision actually faces), so the seam and the local decode agree on
                # the same (now legal) choice.
                expansion_mask = compute_expansion_mask(self._state, seat, self._config)
                flat_action = mask_expansion_logits(flat_action, expansion_mask, self._schema)
            joint_action[seat] = flat_action
            agent_levers[aid] = decode(flat_action, self._schema).levers

        # Capture each seat's prior per-region service_score to recover the per-tick
        # stockout rate.
        prior_service_scores: dict[str, tuple[float, ...]] = {
            aid: self._state.retailers[self._seat_of[aid]].service_score_per_region
            for aid in self.possible_agents
        }
        # Capture each seat's prior seated automation tier to recover the differential
        # capex charged this tick.
        prior_automation_tiers: dict[str, int] = {
            aid: int(self._state.retailers[self._seat_of[aid]].automation_tier)
            for aid in self.possible_agents
        }

        # 2. Advance the world EXACTLY ONCE.
        result = self._world.step(self._state, joint_action)
        self._state = result.next_state
        self._episode_step += 1
        # Stash the raw per-seat infos (all seats, incl. NPCs) for hot-seat scoring.
        self._last_raw_infos = result.infos

        # 3. Split the seam's per-retailer outputs into per-agent dicts (no recomputation).
        observations: dict[str, npt.NDArray[np.float32]] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}
        for aid in self.possible_agents:
            seat = self._seat_of[aid]
            levers = agent_levers[aid]
            observations[aid] = np.asarray(result.observations[seat], dtype=np.float32)
            rewards[aid] = float(result.rewards[seat])
            terminations[aid] = False
            truncations[aid] = False
            marketing_spend_fraction = levers.get("marketing", 0.0)
            marketing_exponent = self._config.marketing.marketing_cost_exponent
            marketing_spend = (
                marketing_spend_fraction * self._config.marketing.cost_per_unit_spend
                if marketing_exponent == 1.0
                else marketing_spend_fraction**marketing_exponent
                * self._config.marketing.cost_per_unit_spend
            )
            infos[aid] = enrich_agent_info(
                result.infos[seat],
                self._state,
                seat,
                self._episode_step,
                marketing_spend=marketing_spend,
                assortment=levers.get("assortment", 0.0),
                promotion=levers.get("promotion", 0.0),
                expansion_action=levers.get("expansion", 0.0),
                # Decoded research spend this tick; research_fidelity is seated state
                # read by enrich.
                research_action=levers.get("research", 0.0),
                # Decoded SCM service this tick + the prior per-region service_score
                # (captured above).
                service_level_action=levers.get("service_level", 1.0),
                prior_service_score_per_region=prior_service_scores[aid],
                config=self._config,
                # Decoded TARGET automation tier this tick; the seated tier and the
                # differential capex are read by enrich.
                automation_choice=levers.get("automation", 0.0),
                prior_automation_tier=prior_automation_tiers[aid],
                # Decoded loyalty-program intensity this tick; 0.0 is the no-program
                # baseline.
                loyalty_spend_action=levers.get("loyalty_spend", 0.0),
                # Decoded wage/warehouse actions this tick; 0.0 is the registry default.
                wage_spend_action=levers.get("wage_spend", 0.0),
                warehouse_invest_action=levers.get("warehouse_invest", 0.0),
            )

        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        """No-op render (no render modes are supported)."""
        return

    def close(self) -> None:
        """No-op close (the env holds no external resources)."""
        return
