"""League training surface: the PettingZoo parallel wrapper and multi-round types.

LeagueEnvWrapper seats one pool member per episode behind a single exposed
agent; SB3 imports stay lazy (checkpoint materialize and warm-start only).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import numpy as np

# PettingZoo's ParallelEnv is the base class SuperSuit's bridge requires via
# an isinstance check (it does not duck-type); inheriting it makes the wrapper
# a first-class PettingZoo parallel env for the bridge path.
from pettingzoo import ParallelEnv

from .pool import FrozenPolicyPool, PoolMember
from .winrate import OpponentDistribution, WinRateMatrix


class LeagueEnvWrapper(ParallelEnv):  # type: ignore[misc, unused-ignore]
    """Parallel wrapper that turns one pooled league match into a PettingZoo env."""

    metadata: ClassVar[dict[str, Any]] = {"name": "retail_sim_league_parallel_v0", "render_modes": []}

    def __init__(
        self,
        wrapped_env_factory: Callable[[tuple[str, ...]], Any],
        pool: FrozenPolicyPool,
        distribution: OpponentDistribution,
        *,
        league_seed: int = 0,
        league_mode: Literal["self_play", "single_learner"] = "self_play",
        single_learner_env_factory: Callable[[int, tuple[str, ...]], Any] | None = None,
        max_episode_steps: int | None = None,
        episode_seed_base: int | None = None,
    ) -> None:
        """Validate mode/factory pairing and store the league parameters."""
        if league_mode == "single_learner" and single_learner_env_factory is None:
            raise ValueError(
                "single_learner mode requires single_learner_env_factory: "
                "Callable[[int, tuple[str, ...]], Any]; got None"
            )
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError(
                f"max_episode_steps must be None or an int >= 1, got {max_episode_steps}"
            )

        self._wrapped_env_factory = wrapped_env_factory
        self._single_learner_env_factory = single_learner_env_factory
        self._pool = pool
        self._distribution = distribution
        self._league_seed = league_seed
        self._league_mode: Literal["self_play", "single_learner"] = league_mode
        self._max_episode_steps = max_episode_steps
        self._ticks_since_reset: int = 0
        self._episode_seed_seq: np.random.SeedSequence | None = (
            np.random.SeedSequence(episode_seed_base) if episode_seed_base is not None else None
        )
        # The carried league Generator: built once from league_seed, never
        # re-seeded. Only reset() consumes it (the three-RNG isolation
        # contract); step() and the __init__ snapshot draw do not.
        self._league_rng: np.random.Generator = np.random.default_rng(league_seed)

        # Build an initial env from the first opponent so observation_space /
        # action_space are queryable before the first reset. The initial draw
        # uses a snapshot of league_rng, so the first reset() draws fresh.
        snapshot = self._league_rng.bit_generator.state
        initial_member = self._distribution.sample(self._league_rng, self._pool)
        self._league_rng.bit_generator.state = snapshot

        # State carried across step() under single_learner + checkpoint:
        # the frozen predict closure and the hidden seat-1 observation.
        # Both are None in self_play mode or with a scripted opponent (the
        # env's NPC mechanism seats that; no in-process injection needed).
        self._frozen_predict: Callable[..., Any] | None = None
        self._frozen_seat_index: int | None = None
        self._last_obs_seat_1: Any = None
        self._first_step_after_reset: bool = True
        # The hidden seat's PettingZoo id: the inner env builds with two
        # learners named retailer_0 / retailer_1.
        self._hidden_agent_id: str = "retailer_1"

        if self._league_mode == "self_play":
            if initial_member.kind != "scripted":
                raise NotImplementedError(
                    "self_play LeagueEnvWrapper only seats scripted-archetype pool members; "
                    "checkpoint seating requires league_mode='single_learner'. "
                    f"Got member kind={initial_member.kind!r}."
                )
            self._env: Any = self._wrapped_env_factory((initial_member.archetype_name,))
            # The PettingZoo agent interface delegates to the wrapped env
            # under self_play, mirroring CurriculumEnvWrapper.
            self.possible_agents: list[str] = list(self._env.possible_agents)
            self.agents: list[str] = list(self._env.agents)
        else:
            # single_learner: scripted member -> 1 learner + NPC seat;
            # checkpoint/archetype -> 2 learners, the wrapper hides seat 1 and
            # injects its action at step time. Either way the exposed agent is
            # retailer_0 only.
            assert self._single_learner_env_factory is not None  # checked above
            self._env = self._build_single_learner_env(initial_member)
            self.possible_agents = ["retailer_0"]
            self.agents = ["retailer_0"]
            if initial_member.kind in ("checkpoint", "archetype"):
                self._frozen_predict = self._pool.materialize(initial_member, self._env)
                self._frozen_seat_index = 1

        # SuperSuit's MarkovVectorEnv reads par_env.unwrapped.render_mode;
        # expose a (None) render_mode here so the bridge sees a defined
        # attribute.
        self.render_mode: Any = getattr(self._env, "render_mode", None)
        # Track the last sampled member (a DX hook for ``info`` / diagnostics).
        self._last_member: PoolMember = initial_member

    def _build_single_learner_env(self, member: PoolMember) -> Any:
        """Construct the inner env for ``single_learner`` mode for ``member``.

        Scripted members get one learner + an NPC seat; every other member gets two
        learners and the wrapper hides seat 1 (its action is injected at step time
        from :meth:`FrozenPolicyPool.materialize`). Both branches go through the same
        factory seam.
        """
        assert self._single_learner_env_factory is not None
        if member.kind == "scripted":
            return self._single_learner_env_factory(1, (member.archetype_name,))
        return self._single_learner_env_factory(2, ())

    @property
    def unwrapped(self) -> LeagueEnvWrapper:
        """Return self — we are the outermost wrapper.

        SuperSuit reads ``par_env.unwrapped.render_mode``; returning the inner env
        would leak a reference that is rebuilt every ``reset()``, so
        :attr:`render_mode` is exposed on the wrapper instead.
        """
        return self

    @property
    def league_rng(self) -> np.random.Generator:
        """The dedicated league-level RNG (opponent sampling, warm starts)."""
        return self._league_rng

    @property
    def last_member(self) -> PoolMember:
        """The most recently sampled :class:`PoolMember` (set by the last ``reset()``)."""
        return self._last_member

    def observation_space(self, agent: str) -> Any:
        """The homogeneous observation Box (identical across opponents).

        Under ``single_learner`` mode only ``retailer_0``'s space is exposed (the
        hidden seat is not a PettingZoo agent).
        """
        return self._env.observation_space(agent)

    def action_space(self, agent: str) -> Any:
        """The homogeneous action Box (schema v10 — UNCHANGED across opponents)."""
        return self._env.action_space(agent)

    @property
    def observation_spaces(self) -> dict[str, Any]:
        """The per-agent observation Box dict (PettingZoo / SuperSuit attribute surface).

        Under ``single_learner`` mode returns ONLY ``retailer_0``'s entry — the
        hidden seat is NOT a PettingZoo agent.
        """
        inner_spaces = cast(dict[str, Any], self._env.observation_spaces)
        if self._league_mode == "single_learner":
            return {"retailer_0": inner_spaces["retailer_0"]}
        return inner_spaces

    @property
    def action_spaces(self) -> dict[str, Any]:
        """The per-agent action Box dict (PettingZoo / SuperSuit attribute surface).

        Under ``single_learner`` mode returns ONLY ``retailer_0``'s entry — the
        hidden seat is NOT a PettingZoo agent.
        """
        inner_spaces = cast(dict[str, Any], self._env.action_spaces)
        if self._league_mode == "single_learner":
            return {"retailer_0": inner_spaces["retailer_0"]}
        return inner_spaces

    def _resolve_episode_seed(self, seed: int | None) -> int | None:
        """Use the caller's seed if given, else draw one per episode from the seed sequence."""
        if seed is not None or self._episode_seed_seq is None or self._max_episode_steps is None:
            return seed
        child = self._episode_seed_seq.spawn(1)[0]
        return int(child.generate_state(1, dtype=np.uint32)[0])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Draw the next opponent, rebuild the env, and return its ``(observations, infos)``.

        The league draw happens here via the carried ``league_rng`` (only ``reset``
        consumes it). The ``seed`` argument seeds the new env's core PCG64 only —
        ``league_rng`` is not advanced by it. With ``episode_seed_base`` set, an
        incoming ``seed=None`` (what the bridge's auto-reset passes) is replaced by a
        fresh per-episode seed via :meth:`_resolve_episode_seed` so episodes do not
        replay the same world; an explicit seed is never touched.

        Under ``single_learner`` mode the sampled member is materialized, the hidden
        seat's initial observation is captured, and only ``retailer_0``'s slice is
        returned.
        """
        member = self._distribution.sample(self._league_rng, self._pool)
        self._last_member = member
        # Close the prior env (frees its World resources); rebuild from the
        # factory. PettingZoo parallel envs are cheap to construct.
        self._env.close()
        self._first_step_after_reset = True
        self._ticks_since_reset = 0
        effective_seed = self._resolve_episode_seed(seed)

        if self._league_mode == "self_play":
            if member.kind != "scripted":
                raise NotImplementedError(
                    "self_play LeagueEnvWrapper only seats scripted-archetype pool members; "
                    "checkpoint seating requires league_mode='single_learner'. "
                    f"Got member kind={member.kind!r}."
                )
            self._env = self._wrapped_env_factory((member.archetype_name,))
            self.possible_agents = list(self._env.possible_agents)
            self.agents = list(self._env.agents)
            return cast(
                tuple[dict[str, Any], dict[str, Any]],
                self._env.reset(seed=effective_seed, options=options),
            )

        # single_learner: build the inner env for member.kind, then cache the
        # frozen predict closure so step() can inject seat-1's action.
        self._env = self._build_single_learner_env(member)
        if member.kind in ("checkpoint", "archetype"):
            self._frozen_predict = self._pool.materialize(member, self._env)
            self._frozen_seat_index = 1
        else:
            self._frozen_predict = None
            self._frozen_seat_index = None
        self.possible_agents = ["retailer_0"]
        self.agents = ["retailer_0"]

        inner_obs, inner_infos = cast(
            tuple[dict[str, Any], dict[str, Any]],
            self._env.reset(seed=effective_seed, options=options),
        )
        # Capture the hidden seat's initial observation for the next step's
        # predict call; the scripted branch leaves this None.
        self._last_obs_seat_1 = inner_obs.get(self._hidden_agent_id)
        observations = {"retailer_0": inner_obs["retailer_0"]}
        infos = {"retailer_0": inner_infos.get("retailer_0", {})}
        return observations, infos

    def _apply_episode_truncation(self, trunc: dict[str, Any], infos: dict[str, Any]) -> None:
        """Mark all agents truncated once the episode hits ``max_episode_steps``."""
        if self._max_episode_steps is None:
            return
        if self._ticks_since_reset < self._max_episode_steps:
            return
        for agent in trunc:
            trunc[agent] = True
            infos.setdefault(agent, {})["TimeLimit.truncated"] = True

    def step(self, actions: dict[str, Any]) -> tuple[Any, Any, Any, Any, Any]:
        """Advance the wrapped match one tick and apply episode truncation."""
        self._ticks_since_reset += 1
        if self._league_mode == "self_play":
            observations, rewards, term, trunc, infos = self._env.step(actions)
            self.agents = list(self._env.agents)
            self._apply_episode_truncation(trunc, infos)
            return observations, rewards, term, trunc, infos

        # single_learner mode.
        if self._frozen_predict is None:
            # Scripted opponent: the NPC fills its own seat, so the joint
            # action is just retailer_0's.
            joint = {"retailer_0": actions["retailer_0"]}
        else:
            seat_1_action = self._frozen_predict(
                self._last_obs_seat_1, episode_start=self._first_step_after_reset
            )
            joint = {
                "retailer_0": actions["retailer_0"],
                self._hidden_agent_id: seat_1_action,
            }

        inner_obs, inner_rewards, inner_term, inner_trunc, inner_infos = self._env.step(joint)
        # Any successful step() clears the first-step flag so the next step
        # gets episode_start=False; reset() re-arms it. Flipped AFTER the
        # inner step() so a step() that raises does not turn off the LSTM
        # reset mid-episode.
        self._first_step_after_reset = False
        # Update the cached hidden-seat observation for the next step.
        # .get(): the scripted branch (n_learning_agents=1) has no retailer_1
        # key.
        self._last_obs_seat_1 = inner_obs.get(self._hidden_agent_id)
        # Hide retailer_1: agents is the inner env's agents intersected with
        # the exposed single-learner surface.
        self.agents = ["retailer_0"] if "retailer_0" in self._env.agents else []
        observations = {"retailer_0": inner_obs["retailer_0"]}
        rewards = {"retailer_0": inner_rewards["retailer_0"]}
        term = {"retailer_0": inner_term["retailer_0"]}
        trunc = {"retailer_0": inner_trunc["retailer_0"]}
        infos = {"retailer_0": inner_infos.get("retailer_0", {})}
        self._apply_episode_truncation(trunc, infos)
        return observations, rewards, term, trunc, infos

    def close(self) -> None:
        """Close the wrapped env (the league carries no resources of its own)."""
        self._env.close()


# --- Multi-round league training loop ---------------------------------------


@dataclass(frozen=True)
class LeagueTrainingResult:
    """What one league training run produced: pool dir, checkpoints, curves, seeds."""

    pool_dir: Path
    rounds_completed: int
    checkpoints: tuple[PoolMember, ...]
    learning_curves: tuple[tuple[tuple[int, float], ...], ...]
    seeds: tuple[int, int, int]
    train_steps_per_round: int
    win_rate_matrix_snapshot: WinRateMatrix | None = None
    max_episode_steps: int | None = None


def _single_learner_wrapped_factory_guard(_archetypes: tuple[str, ...]) -> Any:
    """Raise if called; the factory slot under single_learner mode must stay unused.

    A stub that silently built an env would mask a refactor that accidentally
    routes a self_play call under single_learner. Module scope so mypy --strict
    has a well-typed reference.
    """
    raise AssertionError(
        "wrapped_env_factory must not be invoked under league_mode='single_learner'; "
        "single_learner_env_factory is the construction seam in that mode"
    )


@dataclass(frozen=True)
class _MatrixAwareDistribution:
    """Per-round proxy that injects the win-rate matrix + learner_id into ``sample``.

    The wrapper's ``sample(rng, pool)`` call site does not pass the matrix kwargs;
    the training loop wraps the caller's distribution with this proxy each round
    so they are forwarded transparently. Uniform/latest-N ignore them; PFSP
    consumes them. Not exported.
    """

    inner: OpponentDistribution
    win_rate_matrix: WinRateMatrix
    learner_id: str | None

    def sample(
        self,
        rng: np.random.Generator,
        pool: FrozenPolicyPool,
        *,
        win_rate_matrix: WinRateMatrix | None = None,
        learner_id: str | None = None,
    ) -> PoolMember:
        # The round loop is the authoritative source for the matrix kwargs:
        # override any caller-supplied values so they can't be elided.
        del win_rate_matrix, learner_id
        return self.inner.sample(
            rng,
            pool,
            win_rate_matrix=self.win_rate_matrix,
            learner_id=self.learner_id,
        )


def _load_warm_start_parameters(
    model: Any, *, source_checkpoint: Path, policy_kind: Literal["mlp", "lstm"]
) -> None:
    """Copy the policy weights from a saved checkpoint into ``model`` (MLP or LSTM)."""
    if policy_kind == "lstm":
        # By the time warm_start gets here the model was already built as
        # LSTM (so sb3-contrib is present); this import is guaranteed to
        # succeed.
        from sb3_contrib import RecurrentPPO

        source_model: Any = RecurrentPPO.load(source_checkpoint, device=model.device)
    else:
        from stable_baselines3 import PPO

        source_model = PPO.load(source_checkpoint, device=model.device)
    model.policy.load_state_dict(source_model.policy.state_dict())
