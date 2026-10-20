"""6-segment world scenario curriculum + training-time scenario sampler.

Harness layer; no schema bump (v10 UNCHANGED), no new RNG in core, no obs/action delta.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import CoreConfig, DemandConfig, RegionConfig
from retail_simulator.core.state import (
    BARGAIN_SEEKER,
    BRAND_LOYAL,
    CONVENIENCE,
    PREMIUM_HUNTER,
    PRICE_SENSITIVE,
    QUALITY_SEEKING,
    SegmentParams,
    default_segments,
)
from retail_simulator.harness.sweep import (
    automation_gate_config,
    automation_verdict_for_timing,
    run_sweep,
    scm_gate_config,
    scm_verdict_for_service,
)

if TYPE_CHECKING:  # type-only imports — never pulled in at runtime
    from retail_simulator.envs.parallel_env import RetailParallelEnv

# --- 6-segment world ---------------------------------------------------------

PHASE_1_4_PREMIUM_HUNTER_PROFILE: SegmentParams = SegmentParams(
    beta_price=0.3,
    beta_assortment=1.5,
    beta_marketing=1.8,
    beta_promotion=0.1,
    beta_loyalty=0.0,
    beta_under_reference_aversion=1.2,
    beta_reference=0.0,
    beta_service=0.4,
    beta_program=0.0,
)
PHASE_1_4_BARGAIN_SEEKER_PROFILE: SegmentParams = SegmentParams(
    beta_price=1.6,
    beta_assortment=0.3,
    beta_marketing=0.0,
    beta_promotion=2.5,
    beta_loyalty=0.0,
    beta_under_reference_aversion=0.0,
    beta_reference=0.5,
    beta_service=0.0,
    beta_program=0.0,
)


def phase_1_4_six_segments() -> dict[str, SegmentParams]:
    """The 6-segment table: default segments plus the premium-hunter / bargain-seeker profiles."""
    segments = default_segments()
    segments[PREMIUM_HUNTER] = PHASE_1_4_PREMIUM_HUNTER_PROFILE
    segments[BARGAIN_SEEKER] = PHASE_1_4_BARGAIN_SEEKER_PROFILE
    return segments


_VARIANT_A_REGION_0_MIX: dict[str, float] = {
    PRICE_SENSITIVE: 0.30,
    QUALITY_SEEKING: 0.20,
    CONVENIENCE: 0.20,
    BRAND_LOYAL: 0.10,
    PREMIUM_HUNTER: 0.10,
    BARGAIN_SEEKER: 0.10,
}
_VARIANT_A_REGION_1_MIX: dict[str, float] = {
    PRICE_SENSITIVE: 0.15,
    QUALITY_SEEKING: 0.10,
    CONVENIENCE: 0.10,
    BRAND_LOYAL: 0.40,
    PREMIUM_HUNTER: 0.15,
    BARGAIN_SEEKER: 0.10,
}

# Variant B (premium-leaning): region 0 premium-hunter-heavy, region 1 a more even
# brand-loyal/premium-hunter split.
_VARIANT_B_REGION_0_MIX: dict[str, float] = {
    PRICE_SENSITIVE: 0.15,
    QUALITY_SEEKING: 0.20,
    CONVENIENCE: 0.15,
    BRAND_LOYAL: 0.10,
    PREMIUM_HUNTER: 0.30,
    BARGAIN_SEEKER: 0.10,
}
_VARIANT_B_REGION_1_MIX: dict[str, float] = {
    PRICE_SENSITIVE: 0.15,
    QUALITY_SEEKING: 0.10,
    CONVENIENCE: 0.15,
    BRAND_LOYAL: 0.25,
    PREMIUM_HUNTER: 0.25,
    BARGAIN_SEEKER: 0.10,
}

# Variant C (bargain-seeking): region 0 bargain-seeker-heavy, region 1 brand-loyal-heavy
# with non-trivial bargain-seeker.
_VARIANT_C_REGION_0_MIX: dict[str, float] = {
    PRICE_SENSITIVE: 0.20,
    QUALITY_SEEKING: 0.10,
    CONVENIENCE: 0.20,
    BRAND_LOYAL: 0.10,
    PREMIUM_HUNTER: 0.10,
    BARGAIN_SEEKER: 0.30,
}
_VARIANT_C_REGION_1_MIX: dict[str, float] = {
    PRICE_SENSITIVE: 0.20,
    QUALITY_SEEKING: 0.10,
    CONVENIENCE: 0.10,
    BRAND_LOYAL: 0.35,
    PREMIUM_HUNTER: 0.05,
    BARGAIN_SEEKER: 0.20,
}

# Region 0 (large) / region 1 (smaller) pie sizes — the curriculum varies the mix, not the size.
_PHASE_1_4_REGION_0_DEMAND: float = 1000.0
_PHASE_1_4_REGION_1_DEMAND: float = 600.0


def _phase_1_4_demand_config(
    region_0_mix: dict[str, float],
    region_1_mix: dict[str, float],
    *,
    base: DemandConfig | None = None,
) -> DemandConfig:
    """Build a 2-region ``DemandConfig`` with the given per-region 6-segment mixes.

    Other ``DemandConfig`` fields come from ``base`` (defaults to ``CoreConfig.default().demand``); only the per-region mix changes. The fallback scalar ``segment_mix`` mirrors the region-0 mix.
    """
    base_demand = base if base is not None else CoreConfig.default().demand
    return replace(
        base_demand,
        base_regional_demand=_PHASE_1_4_REGION_0_DEMAND,
        segment_mix=dict(region_0_mix),
        regions=(
            RegionConfig(
                base_regional_demand=_PHASE_1_4_REGION_0_DEMAND,
                segment_mix=dict(region_0_mix),
            ),
            RegionConfig(
                base_regional_demand=_PHASE_1_4_REGION_1_DEMAND,
                segment_mix=dict(region_1_mix),
            ),
        ),
    )


def phase_1_4_six_segment_config(
    base_config: CoreConfig | None = None,
    *,
    region_0_mix: dict[str, float] | None = None,
    region_1_mix: dict[str, float] | None = None,
) -> CoreConfig:
    """6-segment world config with (optionally) overridden per-region mixes; variant A by default."""
    base = base_config if base_config is not None else CoreConfig.default()
    r0 = region_0_mix if region_0_mix is not None else _VARIANT_A_REGION_0_MIX
    r1 = region_1_mix if region_1_mix is not None else _VARIANT_A_REGION_1_MIX
    return replace(
        base,
        demand=_phase_1_4_demand_config(r0, r1, base=base.demand),
        segments=phase_1_4_six_segments(),
    )


# --- ScenarioDistribution protocol + UniformScenarioDistribution -------------


@runtime_checkable
class ScenarioDistribution(Protocol):
    """Protocol: a tuple of scenarios + a ``sample(rng)`` draw."""

    scenarios: tuple[CoreConfig, ...]

    def sample(self, rng: np.random.Generator) -> CoreConfig:  # pragma: no cover - Protocol
        ...


@dataclass(frozen=True)
class UniformScenarioDistribution:
    """Sample scenarios uniformly at random; requires at least one scenario."""

    scenarios: tuple[CoreConfig, ...]

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError(
                "UniformScenarioDistribution requires at least one scenario "
                "(empty tuple makes sample() ill-defined)"
            )

    def sample(self, rng: np.random.Generator) -> CoreConfig:
        """Draw ONE ``CoreConfig`` uniformly at random via ``rng``."""
        index = int(rng.integers(0, len(self.scenarios)))
        return self.scenarios[index]


# --- The basic curriculum (3 variants) ---------------------------------------


PHASE_1_4_BASIC_CURRICULUM_NAME: str = "phase_1_4_basic"


def phase_1_4_basic_curriculum(
    base_config: CoreConfig | None = None,
) -> UniformScenarioDistribution:
    """Build the basic curriculum: 3 variants of the 6-segment world (balanced / premium-leaning / bargain-seeking).

    The curriculum samples one uniformly at each reset; a PPO trained across the distribution learns a policy robust over the competitive regimes. ``base_config`` lets the caller layer extra knobs BEFORE the per-region mix is swapped in.
    """
    return UniformScenarioDistribution(
        scenarios=(
            phase_1_4_six_segment_config(
                base_config,
                region_0_mix=_VARIANT_A_REGION_0_MIX,
                region_1_mix=_VARIANT_A_REGION_1_MIX,
            ),
            phase_1_4_six_segment_config(
                base_config,
                region_0_mix=_VARIANT_B_REGION_0_MIX,
                region_1_mix=_VARIANT_B_REGION_1_MIX,
            ),
            phase_1_4_six_segment_config(
                base_config,
                region_0_mix=_VARIANT_C_REGION_0_MIX,
                region_1_mix=_VARIANT_C_REGION_1_MIX,
            ),
        )
    )


_CURRICULUM_REGISTRY: dict[str, Any] = {
    PHASE_1_4_BASIC_CURRICULUM_NAME: phase_1_4_basic_curriculum,
}


def known_curricula() -> tuple[str, ...]:
    """The names of curricula the CLI's ``--curriculum=<name>`` flag accepts.

    A small name → builder registry; default ``None`` ⇒ no curriculum.
    """
    return tuple(sorted(_CURRICULUM_REGISTRY))


def build_curriculum(name: str, base_config: CoreConfig | None = None) -> ScenarioDistribution:
    """Look up a curriculum builder by name and call it.

    An unknown name raises ``ValueError`` listing the known names (boundary validation, never a silent fallback).
    """
    if name not in _CURRICULUM_REGISTRY:
        raise ValueError(f"unknown curriculum {name!r}; known curricula: {list(known_curricula())}")
    builder = _CURRICULUM_REGISTRY[name]
    distribution = builder(base_config)
    if not isinstance(distribution, ScenarioDistribution):
        # Defensive: a registry entry that returns the wrong type would silently
        # break the gate. Surface it at the boundary instead.
        raise TypeError(
            f"curriculum builder for {name!r} returned {type(distribution).__name__}, "
            f"expected a ScenarioDistribution"
        )
    return distribution


# --- CurriculumEnvWrapper ----------------------------------------------------


# pettingzoo's ParallelEnv is the base class SuperSuit's bridge requires via an isinstance
# check (it does not duck-type); inheriting it makes the wrapper a first-class PettingZoo
# Parallel env. The module-top import is the documented, stable pattern.
from pettingzoo import ParallelEnv  # noqa: E402 - kept near the wrapper for locality


class CurriculumEnvWrapper(ParallelEnv):  # type: ignore[misc, unused-ignore]
    """PettingZoo Parallel wrapper that resamples a scenario (rebuilding the env) on every reset."""

    metadata: dict[str, Any] = {"name": "retail_sim_curriculum_parallel_v0", "render_modes": []}

    def __init__(
        self,
        distribution: ScenarioDistribution,
        *,
        n_learning_agents: int = 2,
        npc_archetypes: tuple[str, ...] = (),
        curriculum_seed: int = 0,
        core_seed: int | None = None,
    ) -> None:
        """Seed the curriculum RNG and build the FIRST scenario's parallel env.

        ``curriculum_seed`` keys the scenario-pick RNG; ``core_seed`` seeds the first built env (later episodes use the seed passed to ``reset(seed=...)``).
        """
        # Lazy import: the parallel env pulls pettingzoo at module load.
        from retail_simulator.envs.parallel_env import RetailParallelEnv

        self._distribution = distribution
        self._n_learning_agents = n_learning_agents
        self._npc_archetypes = npc_archetypes
        self._curriculum_rng: np.random.Generator = np.random.default_rng(curriculum_seed)
        self._curriculum_seed = curriculum_seed
        # Build an INITIAL env from the first scenario so the spaces are queryable BEFORE
        # the first reset. The draw uses a SNAPSHOT of the curriculum rng (reproducible
        # without consuming its state; the first reset redraws).
        first_config = self._distribution.sample(np.random.default_rng(curriculum_seed))
        self._env: RetailParallelEnv = RetailParallelEnv(
            first_config,
            n_learning_agents=n_learning_agents,
            npc_archetypes=npc_archetypes,
            seed=core_seed,
        )
        # The PettingZoo agent interface delegates straight to the wrapped env.
        self.possible_agents: list[str] = list(self._env.possible_agents)
        self.agents: list[str] = list(self.possible_agents)
        # render_mode = None on the wrapper so SuperSuit's bridge (which reads
        # unwrapped.render_mode) sees a well-defined attribute.
        self.render_mode: Any = None
        # Track the last sampled config (a DX hook for ``info`` / diagnostics).
        self._last_scenario: CoreConfig = first_config

    @property
    def unwrapped(self) -> "CurriculumEnvWrapper":
        """The "innermost" env — we are the outermost wrapper, so return self.

        The inner RetailParallelEnv is rebuilt every reset(), so returning it would leak a stale reference; the wrapper IS the env the bridge sees.
        """
        return self

    @property
    def curriculum_rng(self) -> np.random.Generator:
        """The harness-only RNG used for scenario picks."""
        return self._curriculum_rng

    @property
    def last_scenario(self) -> CoreConfig:
        """The most recently sampled ``CoreConfig`` (set by the last ``reset()``)."""
        return self._last_scenario

    def observation_space(self, agent: str) -> Any:
        """The homogeneous observation Box (schema v10 — UNCHANGED across scenarios)."""
        return self._env.observation_space(agent)

    def action_space(self, agent: str) -> Any:
        """The homogeneous action Box (schema v10 — UNCHANGED across scenarios)."""
        return self._env.action_space(agent)

    @property
    def observation_spaces(self) -> dict[str, Any]:
        """The per-agent observation Box dict (PettingZoo / SuperSuit attribute surface)."""
        return self._env.observation_spaces

    @property
    def action_spaces(self) -> dict[str, Any]:
        """The per-agent action Box dict (PettingZoo / SuperSuit attribute surface)."""
        return self._env.action_spaces

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, dict[str, Any]]]:
        """Draw the next scenario, rebuild the env, and return its ``(observations, infos)``.

        The draw happens here (NOT inside ``World.step``): a fresh env is constructed from the sampled config (same seat plan); the new env's ``reset(seed)`` is called. ``seed`` seeds the new env's CORE PCG64 only — the curriculum RNG is NOT advanced by it.
        """
        from retail_simulator.envs.parallel_env import RetailParallelEnv

        scenario = self._distribution.sample(self._curriculum_rng)
        self._last_scenario = scenario
        # Close the prior env; rebuild from the new config (parallel envs are cheap to construct).
        self._env.close()
        self._env = RetailParallelEnv(
            scenario,
            n_learning_agents=self._n_learning_agents,
            npc_archetypes=self._npc_archetypes,
            seed=seed,
        )
        self.possible_agents = list(self._env.possible_agents)
        self.agents = list(self.possible_agents)
        return self._env.reset(seed=seed, options=options)

    def step(
        self, actions: dict[str, npt.NDArray[np.float32]]
    ) -> tuple[
        dict[str, npt.NDArray[np.float32]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Forward one tick to the wrapped env."""
        observations, rewards, term, trunc, infos = self._env.step(actions)
        self.agents = list(self._env.agents)
        return observations, rewards, term, trunc, infos

    def close(self) -> None:
        """Close the wrapped env (the curriculum carries no resources of its own)."""
        self._env.close()


@dataclass(frozen=True)
class RecipeIVRegressionResult:
    """The regression sweep verdicts for the SCM and automation legs."""

    scm_degenerate: bool
    scm_reason: str
    automation_degenerate: bool
    automation_reason: str
    best_scm_service: float
    best_automation_policy: str

    @property
    def passed(self) -> bool:
        """True iff both the SCM and the automation verdicts are non-degenerate."""
        return not (self.scm_degenerate or self.automation_degenerate)


def phase_1_4_recipe_iv_regression_sweep(
    six_segment_config: CoreConfig | None = None,
    *,
    grid_points: int = 11,
    ticks_per_point: int = 200,
    seeds: tuple[int, ...] = (0, 1, 2),
    start_present_both: bool = True,
) -> RecipeIVRegressionResult:
    """Run both legs of the regression sweep (SCM service grid + automation timing) on the 6-segment world."""
    base = six_segment_config if six_segment_config is not None else phase_1_4_six_segment_config()

    # SCM leg: layer the SCM scenario knobs on the 6-segment base (base_segments preserves
    # the 6-segment dict).
    scm_config = scm_gate_config(base, base_segments=phase_1_4_six_segments())
    scm_result = run_sweep(
        scm_config,
        grid_points=grid_points,
        ticks_per_point=ticks_per_point,
        seeds=seeds,
        start_present_both=start_present_both,
        run_research=False,
        run_expansion_timing=False,
        run_scm=True,
        run_automation=False,
        run_loyalty=False,
    )
    if not scm_result.service_rows:
        raise RuntimeError(
            "phase_1_4_recipe_iv_regression_sweep: SCM leg returned no rows; the regression sweep needs a multi-region 6-segment scenario for the SCM leg to run (see scm_verdict_for_service / of)"
        )
    scm_degenerate, scm_reason, best_constant_service, _scm_profit = scm_verdict_for_service(
        scm_result.service_rows,
        scm_result.service_variable_mean if scm_result.service_variable_mean is not None else 0.0,
        (
            scm_result.service_variable_std_error
            if scm_result.service_variable_std_error is not None
            else 0.0
        ),
    )

    automation_config = automation_gate_config(base)
    automation_result = run_sweep(
        automation_config,
        grid_points=grid_points,
        ticks_per_point=ticks_per_point,
        seeds=seeds,
        start_present_both=start_present_both,
        run_research=False,
        run_expansion_timing=False,
        run_scm=False,
        run_automation=True,
        run_loyalty=False,
    )
    if not automation_result.automation_rows:
        raise RuntimeError(
            "phase_1_4_recipe_iv_regression_sweep: automation leg returned no rows; the regression sweep needs the automation scenario for the leg to run (see automation_verdict_for_timing / of)"
        )
    (
        automation_degenerate,
        automation_reason,
        best_automation_policy,
        _automation_profit,
    ) = automation_verdict_for_timing(automation_result.automation_rows)

    return RecipeIVRegressionResult(
        scm_degenerate=scm_degenerate,
        scm_reason=scm_reason,
        automation_degenerate=automation_degenerate,
        automation_reason=automation_reason,
        best_scm_service=best_constant_service,
        best_automation_policy=best_automation_policy,
    )


@dataclass(frozen=True)
class GeneralizationReport:
    """Curriculum vs fixed-model per-scenario score comparison."""

    mean_curriculum_score: float
    mean_fixed_score: float
    per_scenario_curriculum_means: tuple[float, ...]
    per_scenario_fixed_means: tuple[float, ...]
    margin: float
    n_scenarios: int
    n_episodes_per_scenario: int


def _evaluate_policy_in_config(
    policy: Any,
    config: CoreConfig,
    *,
    n_episodes: int,
    n_learning_agents: int,
    npc_archetypes: tuple[str, ...],
    seed: int,
) -> float:
    """Mean stationary episode score of a policy in one scenario config."""
    from retail_simulator.harness.parallel_gate import _episode_stationary_score

    # Lazy import: the parallel env pulls pettingzoo (a 1.5+s import).
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    cfg = config.reward
    env = RetailParallelEnv(
        config, n_learning_agents=n_learning_agents, npc_archetypes=npc_archetypes, seed=seed
    )

    def predict(obs: Any) -> Any:
        action, _state = policy.predict(obs, deterministic=True)
        return action

    scores = [
        _episode_stationary_score(env, cfg, seed=seed + episode, predict=predict)
        for episode in range(n_episodes)
    ]
    env.close()
    return float(np.mean(scores)) if scores else 0.0


def evaluate_generalization(
    curriculum_model: Any,
    fixed_model: Any,
    scenarios: tuple[CoreConfig, ...] | list[CoreConfig],
    *,
    episodes_per_scenario: int = 5,
    n_learning_agents: int = 2,
    npc_archetypes: tuple[str, ...] = ("discounter",),
    seed: int = 0,
) -> GeneralizationReport:
    """Compare a curriculum-trained model against a fixed-scenario model across held-out scenarios."""
    scenarios_tuple = tuple(scenarios)
    if not scenarios_tuple:
        raise ValueError(
            "evaluate_generalization requires at least one held-out scenario; "
            "the diagnostic measures per-scenario means"
        )
    if episodes_per_scenario < 1:
        raise ValueError(
            f"episodes_per_scenario must be >= 1, got {episodes_per_scenario}; "
            "at least one episode per scenario is needed for a mean"
        )

    curriculum_means: list[float] = []
    fixed_means: list[float] = []
    for index, scenario in enumerate(scenarios_tuple):
        scenario_seed = seed + index * 10_000
        curriculum_means.append(
            _evaluate_policy_in_config(
                curriculum_model,
                scenario,
                n_episodes=episodes_per_scenario,
                n_learning_agents=n_learning_agents,
                npc_archetypes=npc_archetypes,
                seed=scenario_seed,
            )
        )
        fixed_means.append(
            _evaluate_policy_in_config(
                fixed_model,
                scenario,
                n_episodes=episodes_per_scenario,
                n_learning_agents=n_learning_agents,
                npc_archetypes=npc_archetypes,
                seed=scenario_seed,
            )
        )

    mean_curriculum = float(np.mean(curriculum_means))
    mean_fixed = float(np.mean(fixed_means))
    return GeneralizationReport(
        mean_curriculum_score=mean_curriculum,
        mean_fixed_score=mean_fixed,
        per_scenario_curriculum_means=tuple(curriculum_means),
        per_scenario_fixed_means=tuple(fixed_means),
        margin=mean_curriculum - mean_fixed,
        n_scenarios=len(scenarios_tuple),
        n_episodes_per_scenario=int(episodes_per_scenario),
    )


__all__ = [
    "CurriculumEnvWrapper",
    "GeneralizationReport",
    "PHASE_1_4_BARGAIN_SEEKER_PROFILE",
    "PHASE_1_4_BASIC_CURRICULUM_NAME",
    "PHASE_1_4_PREMIUM_HUNTER_PROFILE",
    "RecipeIVRegressionResult",
    "ScenarioDistribution",
    "UniformScenarioDistribution",
    "build_curriculum",
    "evaluate_generalization",
    "known_curricula",
    "phase_1_4_basic_curriculum",
    "phase_1_4_recipe_iv_regression_sweep",
    "phase_1_4_six_segment_config",
    "phase_1_4_six_segments",
]
