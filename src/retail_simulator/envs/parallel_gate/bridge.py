"""The SuperSuit<->SB3 bridge: parallel vector-env construction and the bridge smoke.

Heavy RL imports (supersuit, sb3) are done lazily inside the functions so
``import retail_simulator.envs.parallel_gate`` stays cheap and does not hard-require
the ``[train]`` extra.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from retail_simulator.logging_setup import human_steps

_log = logging.getLogger(__name__)

from retail_simulator.core.config import CoreConfig, RewardConfig

if TYPE_CHECKING:  # type-only; never pulled in at runtime import time
    from retail_simulator.envs.curriculum import ScenarioDistribution

_RESEARCH_LEVER_NAME: str = "research"

_SCM_LEVER_NAME: str = "service_level"

_AUTOMATION_LEVER_NAME: str = "automation"

_LOYALTY_LEVER_NAME: str = "loyalty_spend"


# The default self-play competition: a clean all-learning duopoly (F1 default 2),
# scored on the weighted objective. Two homogeneous learning seats exercise the
# param-share bridge + simultaneous actions + the per-seat normalizer.
DEFAULT_N_LEARNING_AGENTS: int = 2
BRIDGE_SMOKE_STEPS: int = 10_000
DIVERSITY_WITHIN_FRACTION: float = 0.80
DIVERSITY_MAX_RATIO: float = 2.0
LSTM_HIDDEN_SIZE: int = 64


def competition_config(*, reward_mode: str = "weighted") -> CoreConfig:
    """Build the default all-learning competition CoreConfig for a reward mode."""
    return CoreConfig(reward=RewardConfig(mode=reward_mode))


def build_parallel_vec_env(
    config: CoreConfig | None = None,
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    npc_archetypes: tuple[str, ...] = (),
    num_vec_envs: int = 1,
    num_cpus: int = 1,
    seed: int | None = None,
    curriculum: ScenarioDistribution | None = None,
    curriculum_seed: int = 0,
    home_regions: tuple[int, ...] | None = None,
    mask_expansion_seats: tuple[int, ...] = (),
) -> Any:
    """Wrap a parallel (or curriculum-wrapped) retail env in a SuperSuit vec env."""
    import supersuit as ss

    from retail_simulator.envs.parallel_env import RetailParallelEnv

    if curriculum is not None and home_regions is not None:
        raise ValueError(
            "build_parallel_vec_env: curriculum and home_regions cannot both be "
            "given — CurriculumEnvWrapper has no seating seam of its own"
        )
    if curriculum is not None and mask_expansion_seats:
        raise ValueError(
            "build_parallel_vec_env: curriculum and mask_expansion_seats cannot both "
            "be given — CurriculumEnvWrapper has no masking seam of its own"
        )

    base_config = config if config is not None else competition_config()
    penv: Any
    if curriculum is not None:
        # Lazy import — keeps ``import retail_simulator.envs.parallel_gate``
        # cheap (CurriculumEnvWrapper pulls envs.parallel_env which pulls
        # pettingzoo). The wrapper's surface mirrors RetailParallelEnv enough for
        # SuperSuit's MarkovVectorEnv (possible_agents, observation_space(agent),
        # action_space(agent), reset(seed, options), step, agents, metadata,
        # unwrapped.render_mode).
        from retail_simulator.envs.curriculum import CurriculumEnvWrapper

        penv = CurriculumEnvWrapper(
            curriculum,
            n_learning_agents=n_learning_agents,
            npc_archetypes=npc_archetypes,
            curriculum_seed=curriculum_seed,
            core_seed=seed,
        )
    else:
        penv = RetailParallelEnv(
            base_config,
            n_learning_agents=n_learning_agents,
            npc_archetypes=npc_archetypes,
            seed=seed,
            home_regions=home_regions,
            mask_expansion_seats=mask_expansion_seats,
        )
    # SuperSuit's MarkovVectorEnv reads ``par_env.unwrapped.render_mode``. The env
    # declares ``render_modes: []`` (no rendering) but does not carry a ``render_mode``
    # attribute; set it to None here (a bridge-only concern — the env itself never
    # renders) so the wrap does not AttributeError. unwrapped already returns self.
    if not hasattr(penv, "render_mode"):
        penv.render_mode = None
    vec = ss.pettingzoo_env_to_vec_env_v1(penv)
    vec = ss.concat_vec_envs_v1(
        vec,
        num_vec_envs,
        num_cpus=num_cpus,
        base_class="stable_baselines3",
    )
    return vec


@dataclass(frozen=True)
class BridgeSmokeResult:
    """Outcome of the end-to-end bridge smoke test."""

    api_test_passed: bool
    learn_steps: int
    n_learning_agents: int


def run_bridge_smoke(
    *,
    n_learning_agents: int = DEFAULT_N_LEARNING_AGENTS,
    learn_steps: int = BRIDGE_SMOKE_STEPS,
    num_cycles: int = 100,
    seed: int = 0,
    policy_kind: Literal["mlp", "lstm"] = "mlp",
) -> BridgeSmokeResult:
    """API-test the parallel env, then build the bridge and train PPO end-to-end."""
    from pettingzoo.test import parallel_api_test

    from retail_simulator.envs.parallel_env import RetailParallelEnv

    parallel_api_test(RetailParallelEnv(n_learning_agents=n_learning_agents), num_cycles=num_cycles)

    # 2. The bridge builds + PPO trains end-to-end (the real plumbing check).
    _log.info(
        "bridge smoke: API test passed; training PPO for %d steps (%d learning agent(s), seed=%d)",
        learn_steps,
        n_learning_agents,
        seed,
    )
    vec = build_parallel_vec_env(n_learning_agents=n_learning_agents, seed=seed)
    model = _make_shared_ppo(vec, seed=seed, policy_kind=policy_kind)
    curve: list[tuple[int, float]] = []
    callback = _make_curve_callback(curve, log_every=max(learn_steps // 5, 1), label="bridge smoke")
    model.learn(total_timesteps=learn_steps, callback=callback, progress_bar=False)
    vec.close()

    return BridgeSmokeResult(
        api_test_passed=True,
        learn_steps=learn_steps,
        n_learning_agents=n_learning_agents,
    )


def _make_shared_ppo(
    vec_env: Any,
    *,
    seed: int,
    policy_kind: Literal["mlp", "lstm"] = "mlp",
    ent_coef: float = 0.0,
    gae_lambda: float = 0.95,
) -> Any:
    """Build a shared-policy PPO (or LSTM RecurrentPPO) on a param-sharing vec env."""
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(seed)
    if policy_kind == "mlp":
        from stable_baselines3 import PPO

        return PPO(
            "MlpPolicy", vec_env, seed=None, verbose=0, ent_coef=ent_coef, gae_lambda=gae_lambda
        )
    if policy_kind == "lstm":
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as exc:  # pragma: no cover - exercised when [recurrent] extra is missing
            raise ImportError(
                "policy_kind='lstm' requires the [recurrent] extra; "
                "install with: pip install 'retail-simulator[recurrent]' "
                "(or: pip install sb3-contrib>=2.3,<3.0)"
            ) from exc
        return RecurrentPPO(
            "MlpLstmPolicy",
            vec_env,
            seed=None,
            verbose=0,
            ent_coef=ent_coef,
            gae_lambda=gae_lambda,
            policy_kwargs={"lstm_hidden_size": LSTM_HIDDEN_SIZE},
        )
    raise ValueError(f"unknown policy_kind {policy_kind!r}; expected 'mlp' or 'lstm'")


def _make_curve_callback(
    sink: list[tuple[int, float]],
    *,
    log_every: int | None = None,
    label: str | None = None,
    logger: logging.Logger | None = None,
) -> Any:
    """SB3 callback recording the shared-policy rolling mean episode reward.

    Mirrors the single-agent gate's curve callback: at each rollout end it appends
    ``(timestep, mean_reward)`` from SB3's ``ep_info_buffer`` (the shared policy's
    episode returns under self-play). ``BaseCallback`` is imported lazily (only the
    ``[train]`` extra has it). The curve drives ``curve_trends_up`` (criterion 2).

    With ``log_every`` set, it also emits a progress line every ``log_every``
    timesteps (timestep / total, rolling episode reward, steps per second) so a
    long ``model.learn`` never runs silently in the terminal. ``label`` prefixes
    the line (e.g. ``"round 3/6"``); the ``parallel_gate`` logger is used by
    default (pass ``logger`` to tag the line with the caller's source), and the
    level follows the entry point's logging setup.
    """
    from stable_baselines3.common import base_class
    from stable_baselines3.common.callbacks import BaseCallback

    class _CurveCallback(BaseCallback):
        def __init__(
            self,
            curve_sink: list[tuple[int, float]],
            *,
            log_every: int | None,
            label: str | None,
            logger: logging.Logger | None = None,
        ) -> None:
            super().__init__()
            self._sink = curve_sink
            self._log_every = log_every
            self._label = label
            self._logger = logger if logger is not None else _log
            self._next_log_at = log_every
            # The model is bound later by SB3 in init_callback(); capture the start
            # there so the first progress line has a real rate.
            self._start_time = time.monotonic()
            self._start_steps = 0
            self._last_log_time: float | None = None
            self._last_log_steps: int = 0

        def init_callback(self, model: base_class.BaseAlgorithm) -> None:
            super().init_callback(model)
            self._start_time = time.monotonic()
            self._start_steps = int(model.num_timesteps)

        def _on_rollout_end(self) -> None:
            buffer = getattr(self.model, "ep_info_buffer", None)
            rewards = [float(ep["r"]) for ep in buffer or [] if "r" in ep]
            if rewards:
                self._sink.append((int(self.num_timesteps), float(np.mean(rewards))))
            if not (self._log_every and self._next_log_at is not None):
                return
            if int(self.num_timesteps) < self._next_log_at:
                return
            # learn() forwards its own locals to the callback, so the denominator is
            # the total_timesteps parameter; fall back to the model's counter.
            total = int(self.locals.get("total_timesteps", 0) or self.model.num_timesteps or 0)
            # A rollout runs to the episode's end, so the counter can overshoot the
            # target; clamp the display so progress never reads past 100%.
            done = int(self.num_timesteps) if total <= 0 else min(int(self.num_timesteps), total)
            prefix = f"{self._label}: " if self._label else ""
            # Rate since the previous log line, or since training start for the first one.
            now = time.monotonic()
            if self._last_log_time is None:
                base_time, base_steps = self._start_time, self._start_steps
            else:
                base_time, base_steps = self._last_log_time, self._last_log_steps
            elapsed = now - base_time
            sps = (int(self.num_timesteps) - base_steps) / elapsed if elapsed > 0 else 0.0
            self._last_log_time = now
            self._last_log_steps = int(self.num_timesteps)
            last = self._sink[-1][1] if self._sink else None
            if last is None or math.isnan(last):
                # The bridged self-play env carries no Monitor episode buffer, so the
                # reward field is simply omitted rather than printing n/a/nan.
                self._logger.info(
                    "%straining %s/%s steps · %.1f sps",
                    prefix,
                    human_steps(done),
                    human_steps(total),
                    sps,
                )
            else:
                self._logger.info(
                    "%straining %s/%s steps · ep_reward %+.1f · %.1f sps",
                    prefix,
                    human_steps(done),
                    human_steps(total),
                    last,
                    sps,
                )
            self._next_log_at += self._log_every

        def _on_step(self) -> bool:
            return True

    return _CurveCallback(sink, log_every=log_every, label=label, logger=logger)
