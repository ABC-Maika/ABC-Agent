"""The SuperSuit<->SB3 bridge: parallel vector-env construction and the bridge smoke.

Heavy RL imports (supersuit, sb3) are done lazily inside the functions so
``import retail_simulator.envs.parallel_gate`` stays cheap and does not hard-require
the ``[train]`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

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
    vec = build_parallel_vec_env(n_learning_agents=n_learning_agents, seed=seed)
    model = _make_shared_ppo(vec, seed=seed, policy_kind=policy_kind)
    model.learn(total_timesteps=learn_steps, progress_bar=False)
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


def _make_curve_callback(sink: list[tuple[int, float]]) -> Any:
    """SB3 callback recording the shared-policy rolling mean episode reward.

    Mirrors the single-agent gate's curve callback: at each rollout end it appends
    ``(timestep, mean_reward)`` from SB3's ``ep_info_buffer`` (the shared policy's
    episode returns under self-play). ``BaseCallback`` is imported lazily (only the
    ``[train]`` extra has it). The curve drives ``curve_trends_up`` (criterion 2).
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class _CurveCallback(BaseCallback):
        def __init__(self, curve_sink: list[tuple[int, float]]) -> None:
            super().__init__()
            self._sink = curve_sink

        def _on_rollout_end(self) -> None:
            buffer = getattr(self.model, "ep_info_buffer", None)
            if buffer:
                mean_reward = float(np.mean([ep["r"] for ep in buffer if "r" in ep]))
                self._sink.append((int(self.num_timesteps), mean_reward))

        def _on_step(self) -> bool:
            return True

    return _CurveCallback(sink)
