"""Multi-round league training loop.

``run_league_training`` drives the pool round by round: a round-0 random-init
snapshot, then per round train -> checkpoint -> round-end head-to-head fold
into the win-rate matrix. Heavy imports (SB3, SuperSuit, envs) stay lazy.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from retail_simulator.logging_setup import human_steps

from .evaluation import _run_round_end_evaluation
from .training import (
    LeagueEnvWrapper,
    LeagueTrainingResult,
    _load_warm_start_parameters,
    _MatrixAwareDistribution,
    _single_learner_wrapped_factory_guard,
)
from .winrate import WinRateMatrix

if TYPE_CHECKING:  # type-only; never pulled in at runtime import time
    from retail_simulator.core.config import CoreConfig
    from retail_simulator.envs.curriculum import ScenarioDistribution

    from .pool import FrozenPolicyPool, PoolMember
    from .winrate import OpponentDistribution

logger = logging.getLogger(__name__)


def run_league_training(
    initial_pool: FrozenPolicyPool,
    n_rounds: int,
    distribution_factory: Callable[[FrozenPolicyPool], OpponentDistribution],
    train_steps_per_round: int,
    seeds: tuple[int, int, int],
    *,
    scenario_distribution: ScenarioDistribution | None = None,
    config: CoreConfig | None = None,
    n_learning_agents: int = 1,
    pfsp_alpha: float = 2.0,
    round_end_eval_episodes: int = 5,
    distribution: OpponentDistribution | None = None,
    policy_kind: Literal["mlp", "lstm"] = "mlp",
    ent_coef: float = 0.0,
    gae_lambda: float = 0.95,
    home_regions: tuple[int, ...] | None = None,
    mask_expansion: bool = False,
    reward_mode: str | None = None,
    max_episode_steps: int | None = None,
    warm_start: bool = False,
) -> LeagueTrainingResult:
    """Train the pool across n_rounds, checkpointing and folding round-end head-to-heads."""
    if n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")

    # Lazy heavy imports keep module import cheap.
    import supersuit as ss

    from retail_simulator.envs.curriculum import CurriculumEnvWrapper
    from retail_simulator.envs.parallel_env import RetailParallelEnv
    from retail_simulator.envs.parallel_gate import (
        _make_curve_callback,
        _make_shared_ppo,
        competition_config,
    )

    if scenario_distribution is not None and home_regions is not None:
        raise ValueError(
            "run_league_training: scenario_distribution and home_regions cannot "
            "both be given — CurriculumEnvWrapper has no seating seam of its own"
        )
    if scenario_distribution is not None and mask_expansion:
        raise ValueError(
            "run_league_training: scenario_distribution and mask_expansion cannot "
            "both be given — CurriculumEnvWrapper has no masking seam of its own"
        )

    league_seed, curriculum_seed, core_seed = seeds
    run_started_at = time.monotonic()
    logger.info(
        "league training starting: %d round(s), %s steps/round, pool=%s",
        n_rounds,
        human_steps(train_steps_per_round),
        initial_pool.pool_dir,
    )
    # The effective config, computed once: reused by both the single-learner
    # env factory and the provenance hash recorded on the pool, so the two
    # can't silently diverge.
    effective_config = config if config is not None else competition_config()

    master_seq = np.random.SeedSequence(league_seed)
    round_seqs = master_seq.spawn(n_rounds)
    per_round_league_seeds: tuple[int, ...] = tuple(
        int(round_seqs[k].generate_state(1, dtype=np.uint32)[0]) for k in range(n_rounds)
    )

    # Load (or initialize) the persistent win-rate matrix: a fresh pool
    # yields an empty matrix, a resumed run restores the cumulative record
    # from <pool_dir>/win_rate_matrix.json.
    win_rate_matrix_path = initial_pool.pool_dir / "win_rate_matrix.json"
    try:
        win_rate_matrix = WinRateMatrix.load(win_rate_matrix_path)
    except FileNotFoundError:
        win_rate_matrix = WinRateMatrix()

    # Record the seating provenance on the pool once, before round 0,
    # always (even when nothing is to record) - see
    # FrozenPolicyPool.record_arena_seating. The region-table hash alone
    # cannot tell arena configs apart (same regions, different economics
    # knobs), so the distinguishing knobs are recorded alongside: the three
    # the lock fixes (overhead_reference_stores/overhead_exponent/
    # provided_capacity_per_store), plus credit_limit/overdraft_rate (which
    # distinguish "arena" from "arena-balanced" and are never touched by a
    # lock amendment), plus writeoff_frac_max for the same future-proofing
    # reason. They are read straight off the effective config.
    if home_regions is not None:
        from retail_simulator.envs.arena import (
            arena_region_table_hash as _arena_region_table_hash,
        )

        region_table_hash = _arena_region_table_hash(effective_config)
        arena_lock_knobs: dict[str, float] | None = {
            "overhead_reference_stores": effective_config.wage.overhead_reference_stores,
            "overhead_exponent": effective_config.wage.overhead_exponent,
            "provided_capacity_per_store": effective_config.warehouse.provided_capacity_per_store,
            "credit_limit": effective_config.wage.credit_limit,
            "overdraft_rate": effective_config.wage.overdraft_rate,
            "writeoff_frac_max": effective_config.wage.writeoff_frac_max,
        }
    else:
        region_table_hash = None
        arena_lock_knobs = None
    initial_pool.record_arena_seating(
        home_regions=home_regions,
        arena_region_table_hash=region_table_hash,
        arena_lock_knobs=arena_lock_knobs,
    )
    # Record the reward mode unconditionally (even when None) - see
    # FrozenPolicyPool.record_reward_mode for why "notable" is the caller's
    # decision.
    initial_pool.record_reward_mode(reward_mode)

    def _single_learner_factory(n_learn: int, archetypes: tuple[str, ...]) -> Any:
        """Build the single-learner env (curriculum or parallel) for the round's trainee."""
        if scenario_distribution is not None:
            return CurriculumEnvWrapper(
                scenario_distribution,
                n_learning_agents=n_learn,
                npc_archetypes=archetypes,
                curriculum_seed=curriculum_seed,
                core_seed=core_seed,
            )
        return RetailParallelEnv(
            effective_config,
            n_learning_agents=n_learn,
            npc_archetypes=archetypes,
            seed=core_seed,
            home_regions=home_regions,
            # Seat 0 is always the trainee under single_learner mode (never
            # the frozen opponent).
            mask_expansion_seats=(0,) if mask_expansion else (),
        )

    def _build_bridged_vec(wrapper: LeagueEnvWrapper) -> Any:
        """Run the SuperSuit bridge over an existing wrapper (mirrors ``build_parallel_vec_env``).

        Bridges the already-constructed :class:`LeagueEnvWrapper` so its ``league_rng``
        state and the round-specific distribution are preserved. The call shape is
        verbatim from :func:`build_parallel_vec_env`.
        """
        if not hasattr(wrapper, "render_mode") or wrapper.render_mode is None:
            wrapper.render_mode = None
        vec_pz = ss.pettingzoo_env_to_vec_env_v1(wrapper)
        return ss.concat_vec_envs_v1(
            vec_pz, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3"
        )

    learning_curves: list[tuple[tuple[int, float], ...]] = []

    # ---- Round 0: build the random-init snapshot WITHOUT calling .learn
    #
    # The throw-away wrapper exists only to satisfy PPO's vec-env requirement;
    # it is closed right after PPO is constructed. Sampling the first scripted
    # member directly keeps round 0 deterministic and independent of the
    # caller's distribution.
    initial_members = initial_pool.members()
    if not initial_members:
        raise ValueError(
            "initial_pool must contain at least one member to host the round-0 "
            "PPO constructor; use make_seeded_pool(...) to seed the pool"
        )
    first_scripted: PoolMember | None = next(
        (m for m in initial_members if m.kind == "scripted"), None
    )
    if first_scripted is None:
        raise ValueError(
            "initial_pool must contain at least one scripted member to host the "
            "round-0 PPO constructor; the F5-A seeded pool provides this"
        )

    # A throw-away distribution that always returns the first scripted
    # member: the league_rng is never advanced beyond the wrapper __init__
    # snapshot draw, and the wrapper's reset() never fires (no .learn()).
    @dataclass(frozen=True)
    class _FixedMemberDistribution:
        member: PoolMember

        def sample(
            self,
            _rng: np.random.Generator,
            _pool: FrozenPolicyPool,
            *,
            win_rate_matrix: WinRateMatrix | None = None,
            learner_id: str | None = None,
        ) -> PoolMember:
            # Protocol-extension kwargs accepted for compatibility; the
            # fixed-member shim ignores them (round 0 has no win-rate data).
            del win_rate_matrix, learner_id
            return self.member

    round_zero_distribution: OpponentDistribution = _FixedMemberDistribution(first_scripted)
    logger.info("round 0/%d: saving the random-init snapshot (no training)", n_rounds)
    round_zero_wrapper = LeagueEnvWrapper(
        wrapped_env_factory=_single_learner_wrapped_factory_guard,
        pool=initial_pool,
        distribution=round_zero_distribution,
        # Per-round spawn-derived seed (round 0).
        league_seed=per_round_league_seeds[0],
        league_mode="single_learner",
        single_learner_env_factory=_single_learner_factory,
        max_episode_steps=max_episode_steps,
        episode_seed_base=core_seed,
    )
    try:
        vec_throw = _build_bridged_vec(round_zero_wrapper)
        try:
            model_zero = _make_shared_ppo(
                vec_throw,
                seed=core_seed,
                policy_kind=policy_kind,
                ent_coef=ent_coef,
                gae_lambda=gae_lambda,
            )
            round_zero_member = initial_pool.add_checkpoint(
                model_zero,
                round_index=0,
                seed=core_seed,
                policy_kind=policy_kind,
            )
            learning_curves.append(())
            logger.info("round 0/%d: saved %s", n_rounds, round_zero_member.member_id)
        finally:
            vec_throw.close()
    finally:
        round_zero_wrapper.close()

    # ---- Rounds 1..n_rounds-1: train then snapshot ------------------------
    # ``pfsp_alpha`` is informational for callers passing ``distribution``
    # directly (the PFSP instance carries its own alpha).
    del pfsp_alpha
    for round_index in range(1, n_rounds):
        # Distribution selection: an explicit caller distribution is used
        # as-is; otherwise rebuild via the factory at the top of each round so
        # the pool growth from the prior round is visible to this round's
        # opponent draws.
        base_distribution: OpponentDistribution = (
            distribution if distribution is not None else distribution_factory(initial_pool)
        )
        # Thread the win-rate matrix + prior-round learner id into
        # sample(): the wrapper's call site does not pass these kwargs, so
        # wrap with a per-round proxy that forwards them (uniform/latest-N
        # ignore them; PFSP consumes them). The learner_id is the most
        # recently appended checkpoint (round K-1's snapshot) - round K's
        # does not exist yet.
        prior_learner_members = [m for m in initial_pool.members() if m.kind == "checkpoint"]
        prior_learner_id = prior_learner_members[-1].member_id if prior_learner_members else None
        round_distribution: OpponentDistribution = _MatrixAwareDistribution(
            inner=base_distribution,
            win_rate_matrix=win_rate_matrix,
            learner_id=prior_learner_id,
        )
        round_started_at = time.monotonic()
        n_pool_members = len(initial_pool.members())
        logger.info(
            "round %d/%d: training %s steps (pool has %d member(s), league_seed=%d) · %s",
            round_index,
            n_rounds,
            human_steps(train_steps_per_round),
            n_pool_members,
            per_round_league_seeds[round_index],
            "warm start" if (warm_start and round_index >= 2) else "from-scratch init",
        )
        wrapper = LeagueEnvWrapper(
            wrapped_env_factory=_single_learner_wrapped_factory_guard,
            pool=initial_pool,
            distribution=round_distribution,
            # Per-round spawn-derived seed.
            league_seed=per_round_league_seeds[round_index],
            league_mode="single_learner",
            single_learner_env_factory=_single_learner_factory,
            max_episode_steps=max_episode_steps,
            episode_seed_base=core_seed,
        )
        try:
            vec = _build_bridged_vec(wrapper)
            try:
                curve: list[tuple[int, float]] = []
                callback = _make_curve_callback(
                    curve,
                    log_every=max(train_steps_per_round // 20, 1),
                    label=f"round {round_index}/{n_rounds}",
                    logger=logger,
                )
                model = _make_shared_ppo(
                    vec,
                    seed=core_seed,
                    policy_kind=policy_kind,
                    ent_coef=ent_coef,
                    gae_lambda=gae_lambda,
                )
                if warm_start and round_index >= 2:
                    warm_start_source = prior_learner_members[-1]
                    warm_start_checkpoint_path = warm_start_source.checkpoint_path
                    if warm_start_checkpoint_path is None:
                        # Defensive: a checkpoint member is guaranteed a path by
                        # __post_init__, but typing infers Path | None.
                        raise RuntimeError(
                            f"warm start: checkpoint PoolMember "
                            f"{warm_start_source.member_id!r} has no checkpoint_path; "
                            f"the dataclass invariant was bypassed"
                        )
                    _load_warm_start_parameters(
                        model,
                        source_checkpoint=initial_pool.pool_dir / warm_start_checkpoint_path,
                        policy_kind=policy_kind,
                    )
                    logger.info(
                        "warm start: round %d policy initialized from %s",
                        round_index,
                        warm_start_source.member_id,
                    )
                model.learn(
                    total_timesteps=train_steps_per_round,
                    callback=callback,
                    progress_bar=False,
                )
                logger.info(
                    "round %d/%d: training done in %.1fs",
                    round_index,
                    n_rounds,
                    time.monotonic() - round_started_at,
                )
                round_k_member = initial_pool.add_checkpoint(
                    model,
                    round_index=round_index,
                    seed=core_seed,
                    policy_kind=policy_kind,
                )
                learning_curves.append(tuple(curve))
                checkpoint_path = (
                    initial_pool.pool_dir / round_k_member.checkpoint_path
                    if round_k_member.checkpoint_path is not None
                    else None
                )
                logger.info(
                    "round %d/%d: saved %s%s",
                    round_index,
                    n_rounds,
                    round_k_member.member_id,
                    f" -> {checkpoint_path}" if checkpoint_path is not None else "",
                )
            finally:
                vec.close()
        finally:
            wrapper.close()

        # Round-end head-to-head evaluation: runs for all distributions
        # (uniform/latest-N ignore the matrix; PFSP consumes it next round).
        # The matrix is persisted atomically after each fold so a resumed run
        # sees the cumulative record. The per-round seed offset keeps the eval
        # reproducible and distinct from the evaluator's internal offsets.
        n_opponents = max(len(initial_pool.members()) - 1, 0)
        logger.info(
            "round %d/%d: round-end head-to-head vs %d opponent(s), %d episode(s) each",
            round_index,
            n_rounds,
            n_opponents,
            round_end_eval_episodes,
        )
        round_end_results = _run_round_end_evaluation(
            initial_pool,
            round_k_member,
            n_episodes=round_end_eval_episodes,
            config=config,
            seed=core_seed + round_index,
            home_regions=home_regions,
        )
        if round_end_results:
            summary_parts = [
                f"{opp_id} {entry.wins}/{entry.total}"
                for opp_id, entry in sorted(round_end_results.items())
            ]
            logger.info(
                "round %d/%d: eval wins — %s",
                round_index,
                n_rounds,
                "  ".join(summary_parts),
            )
        logger.info(
            "round %d/%d: done in %.1fs",
            round_index,
            n_rounds,
            time.monotonic() - round_started_at,
        )
        for opp_id, entry in round_end_results.items():
            # record(...) adds one outcome per call; fold the per-pair
            # bucketed wins/losses into the cumulative matrix.
            for _ in range(entry.wins):
                win_rate_matrix.record(
                    learner_id=round_k_member.member_id,
                    opponent_id=opp_id,
                    learner_won=True,
                )
            for _ in range(entry.total - entry.wins):
                win_rate_matrix.record(
                    learner_id=round_k_member.member_id,
                    opponent_id=opp_id,
                    learner_won=False,
                )
        win_rate_matrix.save(win_rate_matrix_path)

    checkpoints = tuple(m for m in initial_pool.members() if m.kind == "checkpoint")
    logger.info(
        "league training complete: %d round(s), %d checkpoint(s) in %s (%.1fs total)",
        n_rounds,
        len(checkpoints),
        initial_pool.pool_dir,
        time.monotonic() - run_started_at,
    )
    # n_learning_agents is reserved for forward compatibility (a 2-learner
    # shared-policy variant); the current loop is single-learner only.
    del n_learning_agents
    return LeagueTrainingResult(
        pool_dir=initial_pool.pool_dir,
        rounds_completed=n_rounds,
        checkpoints=checkpoints,
        learning_curves=tuple(learning_curves),
        seeds=seeds,
        train_steps_per_round=train_steps_per_round,
        win_rate_matrix_snapshot=win_rate_matrix,
        max_episode_steps=max_episode_steps,
    )


