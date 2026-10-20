#!/usr/bin/env python
"""Train a PFSP self-play league, tournament the distinct checkpoints, measure non-transitivity.

Three steps: (1) train a league under calibrated_spike_config, snapshotting a PPO
checkpoint per round; (2) tournament the distinct checkpoints (deduped by weight
hash) head-to-head, position-controlled over a fixed shared seed set; (3) measure
non-transitivity (dominance graph, 3-cycles, Condorcet winner, decisive cycles)
and write a summary JSON.

``rl_confirms = True`` means >=1 DECISIVE cycle (every edge's bootstrap CI
excludes 0) AND no Condorcet winner. ``False`` is a legitimate finding — the
game has counters but RL collapsed to a corner — and is reported as-is.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from phase4 import distinct_checkpoints, measure, tournament, train_league
from phase4.tournament import _print_table, _summary_payload
from phase4.training import DEFAULT_ENT_COEF, DEFAULT_GAE_LAMBDA, DEFAULT_PFSP_MIX_UNIFORM

from retail_simulator.core.config import CoreConfig
from retail_simulator.envs.spike import calibrated_spike_config

REPO = Path(__file__).resolve().parent.parent
RUNS_ROOT = REPO / "runs" / "phase4_rl_confirm"
DOCS_DATA = REPO / "docs" / "data"

logger = logging.getLogger("phase4_rl_confirm")

# Defaults (full run). PFSP grows a diverse pool over R rounds of S steps each.
DEFAULT_ROUNDS = 3
DEFAULT_STEPS_PER_ROUND = 300_000
DEFAULT_PFSP_ALPHA = 2.0
# The (league, curriculum, core) seed triple threaded into run_league_training.
DEFAULT_SEEDS: tuple[int, int, int] = (0, 0, 42)
# Tournament: fixed shared seed set 0..K-1 (the spike/ladder fairness contract).
DEFAULT_TOURNEY_SEEDS = 24
DEFAULT_TOURNEY_EPISODES = 1
DEFAULT_BOOTSTRAP_RESAMPLES = 1000

# Smoke budgets — exercise EVERY code path (train -> snapshot -> dedup -> tournament ->
# cycle measurement) fast enough to validate the driver end-to-end in ~1-2 min.
SMOKE_ROUNDS = 2
SMOKE_STEPS_PER_ROUND = 2_000
SMOKE_TOURNEY_SEEDS = 3
SMOKE_BOOTSTRAP_RESAMPLES = 200


def _load_existing_pool_state(
    pool_dir: Path,
) -> tuple[
    tuple[Any, ...],
    tuple[str, ...],
    tuple[int, ...] | None,
    str | None,
    dict[str, float] | None,
    str | None,
]:
    """Reload the trained checkpoints + training provenance from an existing pool dir (--resume).

    Every value is read off the pool's on-disk manifest, NOT from this invocation's CLI flags: under --resume, train_league is skipped, so --seed-archetypes/--env/--reward reflect nothing about what the resumed pool actually contains or was trained under.

    Returns ``(checkpoints, seed_archetypes, home_regions, arena_region_table_hash, arena_lock_knobs, reward_mode)``; ``reward_mode`` is ``None`` for pools that never trained under an explicit "weighted" mode — the caller treats that like the "profit" default.
    """
    from retail_simulator.envs.league import FrozenPolicyPool

    pool = FrozenPolicyPool.load(pool_dir)
    members = pool.members()
    checkpoints = tuple(m for m in members if m.kind == "checkpoint")
    seed_archetypes = tuple(m.archetype_name for m in members if m.kind == "archetype")
    return (
        checkpoints,
        seed_archetypes,
        pool.home_regions,
        pool.arena_region_table_hash,
        pool.arena_lock_knobs,
        pool.reward_mode,
    )


def _resolve_env(
    env: str,
) -> tuple[CoreConfig | None, tuple[int, ...] | None, str | None, dict[str, float] | None]:
    """Resolve ``--env`` into ``(config, home_regions, arena_region_table_hash, arena_lock_knobs)``.

    ``"calibrated"`` (default) -> all None (train_league's own :func:`calibrated_spike_config` default, no explicit seating). ``"arena"`` -> the 25-region lock (:func:`~retail_simulator.envs.arena.arena_lock_config`); ``"arena-balanced"`` -> the same table with a working-capital credit line + smaller warehouse footprint (:func:`~retail_simulator.envs.arena.arena_balanced_config`). Both arena variants seat both seats and record the region-table hash plus the economics knobs the hash alone cannot discriminate — ``credit_limit``/``overdraft_rate`` distinguish the two variants, and they are read off the resolved config so config/homes/hash/knobs are computed once and can never diverge.
    """
    if env in ("arena", "arena-balanced"):
        from retail_simulator.envs.arena import (
            arena_balanced_config,
            arena_home_regions,
            arena_lock_config,
            arena_region_table_hash,
        )

        config = arena_balanced_config() if env == "arena-balanced" else arena_lock_config()
        home_regions = arena_home_regions(2)
        arena_lock_knobs = {
            "overhead_reference_stores": config.wage.overhead_reference_stores,
            "overhead_exponent": config.wage.overhead_exponent,
            "provided_capacity_per_store": config.warehouse.provided_capacity_per_store,
            "credit_limit": config.wage.credit_limit,
            "overdraft_rate": config.wage.overdraft_rate,
            "writeoff_frac_max": config.wage.writeoff_frac_max,
        }
        return config, home_regions, arena_region_table_hash(config), arena_lock_knobs
    return None, None, None, None


def _apply_reward_override(config: CoreConfig | None, reward: str | None) -> CoreConfig | None:
    """Return ``config`` with its reward mode replaced by ``reward`` (unchanged when ``reward is None``); a ``None`` config falls back to :func:`calibrated_spike_config` first."""
    if reward is None:
        return config
    effective = config if config is not None else calibrated_spike_config()
    return replace(effective, reward=replace(effective.reward, mode=reward))


def _effective_reward_mode(config: CoreConfig | None) -> str:
    """The reward mode ``config`` will actually train/tournament under.

    Mirrors train_league's/tournament's own ``calibrated_spike_config`` fallback, so the summary always names the SAME mode those functions resolve to. Always a concrete "profit"/"weighted" (never nulled), including under --resume, because the tournament step DOES run under --resume.
    """
    effective = config if config is not None else calibrated_spike_config()
    return effective.reward.mode


def run(args: argparse.Namespace) -> int:
    rounds = SMOKE_ROUNDS if args.smoke else args.rounds
    steps = SMOKE_STEPS_PER_ROUND if args.smoke else args.train_steps_per_round
    tourney_seeds = SMOKE_TOURNEY_SEEDS if args.smoke else args.tourney_seeds
    resamples = SMOKE_BOOTSTRAP_RESAMPLES if args.smoke else args.bootstrap_resamples
    seeds = args.seeds
    ent_coef = args.ent_coef
    gae_lambda = args.gae_lambda
    mask_expansion = args.mask_expansion
    max_episode_steps = args.league_max_episode_steps
    warm_start = args.warm_start
    run_label = args.label
    resumed = args.resume

    seed_archetypes: tuple[str, ...]
    if args.seed_archetypes:
        if args.env in ("arena", "arena-balanced"):
            from retail_simulator.envs.geo_archetypes import (
                GEO_ARCHETYPE_LABELS,
            )

            seed_archetypes = GEO_ARCHETYPE_LABELS
        else:
            from retail_simulator.envs.league import (
                POOL_SPIKE_ARCHETYPE_LABELS,
            )

            seed_archetypes = POOL_SPIKE_ARCHETYPE_LABELS
    else:
        seed_archetypes = ()

    run_name = "smoke" if args.smoke else args.run_name
    pool_dir = Path(args.pool_dir) if args.pool_dir else RUNS_ROOT / run_name
    pool_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        # --resume never calls train_league, so this invocation's CLI flags reflect nothing
        # about this pool: the resumed pool's own manifest is authoritative for seeding,
        # seating, and reward mode.
        (
            checkpoints,
            seed_archetypes,
            home_regions,
            region_table_hash,
            arena_lock_knobs,
            stored_reward_mode,
        ) = _load_existing_pool_state(pool_dir)
        if home_regions is None:
            env_choice = "calibrated"
        else:
            # "arena" and "arena-balanced" share the exact region table, so disambiguate on
            # `credit_limit` (0 vs ARENA_BALANCED_CREDIT_LIMIT), NOT provided_capacity_per_store:
            # the lock's own knobs may move (a future lock amendment could set its cap to 300,
            # colliding with the balanced variant), but credit_limit/overdraft_rate exist only
            # on the balanced variant.
            from retail_simulator.envs.arena import (
                ARENA_BALANCED_CREDIT_LIMIT,
            )

            is_balanced = (
                arena_lock_knobs is not None
                and arena_lock_knobs.get("credit_limit") == ARENA_BALANCED_CREDIT_LIMIT
            )
            env_choice = "arena-balanced" if is_balanced else "arena"
        # The manifest-wins precedence above is deliberate; log when --env disagrees so it is
        # not silent.
        if args.env != env_choice:
            logger.warning(
                "--resume: --env=%s was given but the resumed pool's manifest "
                "says %s; using %s (the pool's recorded seating is authoritative, "
                "not the --env flag)",
                args.env,
                env_choice,
                env_choice,
            )
        # Only the config half is needed here (homes/hash/knobs came from the manifest);
        # fresh_region_table_hash guards against a moved lock.
        league_config, _, fresh_region_table_hash, _ = _resolve_env(env_choice)
        # --reward under --resume: train_league never runs, so the flag only affects the
        # tournament scoring. When OMITTED, the pool's manifest-recorded mode wins (a
        # weighted-trained pool must not be silently re-scored on profit). An EXPLICIT
        # --reward still wins over the manifest (re-scoring without retraining is a
        # deliberate, documented use), with a warning on mismatch.
        effective_reward = args.reward if args.reward is not None else stored_reward_mode
        if args.reward is not None:
            historical_reward_mode = (
                stored_reward_mode if stored_reward_mode is not None else "profit"
            )
            if args.reward != historical_reward_mode:
                logger.warning(
                    "--resume: --reward=%s was given but the resumed pool's manifest "
                    "records training under reward_mode=%s; the tournament will score "
                    "these checkpoints under %s, which is NOT what they were trained under",
                    args.reward,
                    historical_reward_mode,
                    args.reward,
                )
        league_config = _apply_reward_override(league_config, effective_reward)
        # Guard against a moved lock: the stored hash came from the manifest, but
        # league_config was rebuilt from today's arena config — if the lock moved since
        # training, the tournament would roll out on the new world while the summary
        # reports the old hash.
        if region_table_hash is not None and region_table_hash != fresh_region_table_hash:
            logger.error(
                "--resume: pool was trained under a different arena region table; refusing"
            )
            return 1
        if not checkpoints:
            logger.error(
                "--resume: no trained checkpoints found in %s; run without --resume", pool_dir
            )
            return 1
        logger.info("--resume: reusing %d trained checkpoint(s) in %s", len(checkpoints), pool_dir)
    else:
        env_choice = args.env
        league_config, home_regions, region_table_hash, arena_lock_knobs = _resolve_env(env_choice)
        # Apply --reward AFTER _resolve_env, so a resolver that already set mode="weighted"
        # is never silently overridden back to "profit".
        league_config = _apply_reward_override(league_config, args.reward)
        training_result = train_league(
            pool_dir=pool_dir,
            n_rounds=rounds,
            train_steps_per_round=steps,
            pfsp_alpha=args.pfsp_alpha,
            seeds=seeds,
            ent_coef=ent_coef,
            gae_lambda=gae_lambda,
            seed_archetypes=seed_archetypes,
            config=league_config,
            home_regions=home_regions,
            mask_expansion=mask_expansion,
            max_episode_steps=max_episode_steps,
            pfsp_mix_uniform=args.pfsp_mix_uniform,
            warm_start=warm_start,
        )
        checkpoints = training_result.checkpoints

    # The reward mode league_config ends up carrying (post-override, both branches) — always
    # concrete, never null.
    reward_mode = _effective_reward_mode(league_config)

    label_to_zip = distinct_checkpoints(pool_dir, checkpoints)
    if len(label_to_zip) < 3:
        logger.warning(
            "only %d distinct trained checkpoint(s) — a tournament needs >=3 for a 3-cycle; "
            "the dominance graph is reported but cannot exhibit non-transitivity. "
            "(Expected with --smoke's tiny round count; increase --rounds for a real run.)",
            len(label_to_zip),
        )

    # The tournament rolls out on the SAME (config, home_regions) the checkpoints were
    # trained under — but always UNMASKED, even for masked-trained checkpoints (see
    # MASK_EXPANSION_SCOPE_NOTE).
    pairwise = tournament(
        label_to_zip,
        n_seeds=tourney_seeds,
        n_episodes=args.tourney_episodes,
        bootstrap_resamples=resamples,
        config=league_config,
        home_regions=home_regions,
    )
    result = measure(pairwise, tuple(sorted(label_to_zip)))
    _print_table(result)

    # Under --resume the recipe knobs (rounds, steps, seeds, ...) describe nothing about how
    # this pool was trained — the manifest carries no recipe — so null them out; `resumed`
    # records why.
    payload = _summary_payload(
        result,
        rounds=None if resumed else rounds,
        train_steps_per_round=None if resumed else steps,
        pfsp_alpha=None if resumed else args.pfsp_alpha,
        pfsp_mix_uniform=None if resumed else args.pfsp_mix_uniform,
        seeds=None if resumed else seeds,
        ent_coef=None if resumed else ent_coef,
        gae_lambda=None if resumed else gae_lambda,
        mask_expansion=None if resumed else mask_expansion,
        max_episode_steps=None if resumed else max_episode_steps,
        warm_start=None if resumed else warm_start,
        reward_mode=reward_mode,
        seed_archetypes=seed_archetypes,
        env=env_choice,
        home_regions=home_regions,
        arena_region_table_hash=region_table_hash,
        arena_lock_knobs=arena_lock_knobs,
        label=run_label,
        tourney_seeds=tourney_seeds,
        tourney_episodes=args.tourney_episodes,
        smoke=args.smoke,
        resumed=resumed,
    )
    if args.summary:
        summary_path = Path(args.summary)
    else:
        today = _dt.date.today().isoformat()
        smoke_suffix = "_smoke" if args.smoke else ""
        label_suffix = f"_{run_label}" if run_label else ""
        summary_path = DOCS_DATA / f"phase4_rl_confirm_{today}{smoke_suffix}{label_suffix}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"summary written to {summary_path}")
    return 0


def _parse_seeds(value: str) -> tuple[int, int, int]:
    """Parse ``--seeds league,curriculum,core`` into the 3-tuple ``run_league_training`` wants.

    Boundary validation (argparse ``type=``): exactly 3 comma-separated ints, else a clean argparse usage error.
    """
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--seeds must be 3 comma-separated ints 'league,curriculum,core'; got {value!r}"
        )
    try:
        parsed_seeds = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--seeds must be 3 comma-separated ints 'league,curriculum,core'; got {value!r}"
        ) from exc
    league_seed, curriculum_seed, core_seed = parsed_seeds
    return (league_seed, curriculum_seed, core_seed)


def _parse_label(value: str) -> str:
    """Validate ``--label`` as a safe filename component (``[A-Za-z0-9_-]+``)."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise argparse.ArgumentTypeError(
            f"--label must match [A-Za-z0-9_-]+ (a safe filename component); got {value!r}"
        )
    return value


def _parse_gae_lambda(value: str) -> float:
    """Validate ``--gae-lambda`` as a float in ``(0, 1]`` (argparse ``type=``).

    0 or below, or above 1, degrades PPO's advantage estimator silently (SB3 does not clamp it); rejecting it at the CLI boundary turns a typo into a clean usage error.
    """
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--gae-lambda must be a float in (0, 1]; got {value!r}"
        ) from exc
    if not (0.0 < parsed <= 1.0):
        raise argparse.ArgumentTypeError(f"--gae-lambda must be a float in (0, 1]; got {value!r}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny train+tournament to validate the driver end-to-end",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip training; tournament the existing pool's checkpoints",
    )
    parser.add_argument(
        "--run-name", default="run", help="sub-dir under runs/phase4_rl_confirm/ (full runs)"
    )
    parser.add_argument(
        "--pool-dir",
        default=None,
        help="override the pool dir (default: runs/phase4_rl_confirm/<run-name>)",
    )
    parser.add_argument(
        "--rounds", type=int, default=DEFAULT_ROUNDS, help="league rounds (full run)"
    )
    parser.add_argument(
        "--train-steps-per-round",
        type=int,
        default=DEFAULT_STEPS_PER_ROUND,
        help="PPO steps per round (full run)",
    )
    parser.add_argument(
        "--pfsp-alpha", type=float, default=DEFAULT_PFSP_ALPHA, help="PFSP weighting exponent"
    )
    parser.add_argument(
        "--pfsp-mix-uniform",
        type=float,
        default=DEFAULT_PFSP_MIX_UNIFORM,
        help=(
            f"""decision 10 (taken 2026-09-19): uniform-mixing floor passed to PFSPOpponentDistribution(mix_uniform=...) -- guarantees every eligible pool member keeps at least mix_uniform/n draw probability each round, preventing the single-opponent collapse measured (default:{DEFAULT_PFSP_MIX_UNIFORM}, no floor; must be in [0.0, 1.0])"""
        ),
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=DEFAULT_SEEDS,
        help=(
            "'league,curriculum,core' seed triple threaded into run_league_training "
            f"(comma-separated ints; default: {','.join(str(s) for s in DEFAULT_SEEDS)})"
        ),
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=DEFAULT_ENT_COEF,
        help=(
            "PPO/RecurrentPPO entropy-bonus coefficient, passed straight through to "
            f"_make_shared_ppo (SB3 default: {DEFAULT_ENT_COEF})"
        ),
    )
    parser.add_argument(
        "--gae-lambda",
        type=_parse_gae_lambda,
        default=DEFAULT_GAE_LAMBDA,
        help=(
            "PPO/RecurrentPPO GAE lambda, passed straight through to "
            "_make_shared_ppo; raising it lengthens the advantage estimator's "
            "horizon 1/(1-gamma*lambda) so a learner can see slower-payoff "
            "levers like expansion (the pre-registered arena recipe uses 0.98) "
            f"(SB3 default: {DEFAULT_GAE_LAMBDA}; must be in (0, 1])"
        ),
    )
    parser.add_argument(
        "--mask-expansion",
        action="store_true",
        help=(
            "S8 (/ row S8): mask the learner's illegal expansion logits (already-open/unaffordable regions, and slots beyond n_regions) to -inf before its argmax decode, so it can only ever make a legal choice (hold is always legal but is not itself a region); the frozen opponent is never masked. Applies to the learner's TRAINING rollouts only -- the post-training tournament below and every evaluation/scoring surface (the ladder, league snapshot/round-end evaluations, the arena) always roll out UNMASKED; see the summary JSON's training.mask_expansion_scope field. Recorded in training.mask_expansion. Default (off) is byte-identical to every prior run"
        ),
    )
    parser.add_argument(
        "--league-max-episode-steps",
        type=int,
        default=None,
        help=(
            "decision 8 (taken 2026-09-18): the league training-episode horizon, passed straight through to run_league_training(max_episode_steps=...) -> every LeagueEnvWrapper it builds. Default (unset -> None) never truncates -- one opponent per round, byte-identical to every prior run. A positive int truncates each round's game at that many ticks so the SuperSuit bridge auto-resets mid-round and draws a NEW opponent per EPISODE instead (the arena recipe uses 1000, the arena match horizon). Recorded in training.max_episode_steps; null under --resume (train_league skipped, mirrors --mask-expansion/--ent-coef/--gae-lambda)"
        ),
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help=(
            "owner decision 12a (taken 2026-09-19): round k>=2 continues from round "
            "(k-1)'s just-trained checkpoint's policy PARAMETERS instead of an "
            "independent from-scratch init, passed straight through to "
            "run_league_training(warm_start=...) -- see that function's own "
            "docstring for the mechanism (only the policy weights transfer; the "
            "optimizer stays fresh). Round 1 is unaffected either way -- it always "
            "starts from round 0's random init. Default (off) is byte-identical to "
            "every prior run. Recorded in training.warm_start; null under --resume "
            "(train_league skipped, mirrors --mask-expansion/--ent-coef/--gae-lambda)"
        ),
    )
    parser.add_argument(
        "--seed-archetypes",
        action="store_true",
        help=(
            "seed the pool with a fixed-opponent archetype roster alongside the default scripted seed, following --env (H4, addendum; RLB-8b): under --env calibrated (default), the three calibrated Phase 4 spike postures (lean, aggressor, patient); under --env arena, the four closed-loop geo postures (sprawler, fortress, cherry-picker, fast-follower) instead — the spike postures' fixed expansion_choice can only ever open region 1 there. Default behavior (off) is byte-identical to every prior run"
        ),
    )
    parser.add_argument(
        "--env",
        choices=["calibrated", "arena", "arena-balanced"],
        default="calibrated",
        help=(
            "which world train_league trains on and the tournament rates on "
            "(RLB-8a; credit-line-and-rebalance): 'calibrated' (default) is "
            "calibrated_spike_config, byte-identical to every prior run; 'arena' is "
            "the 25-region candidate-lock arena (harness.arena.arena_lock_config); "
            "'arena-balanced' is the same 25-region arena with a working-capital "
            "credit line + a smaller warehouse footprint (harness.arena."
            "arena_balanced_config) — both arena envs seat both seats via "
            "arena_home_regions(2). Ignored under --resume (the resumed pool's own "
            "recorded seating is authoritative, like --seed-archetypes already is)."
        ),
    )
    parser.add_argument(
        "--reward",
        choices=["profit", "weighted"],
        default=None,
        help=(
            "override the resolved --env's reward mode (core/reward.py; H5) for BOTH training and the tournament: 'profit' is raw profit, 'weighted' is the DESIGN-locked per-component z-normalized blend (RewardConfig's own weights). Default (unset) leaves the resolved env's own reward mode untouched -- calibrated/arena/arena-balanced all default to 'profit', byte-identical to every prior run, and a caller that already resolved a 'weighted' config (arena_train_player.py's monkey-patched env resolver) is never silently reset to 'profit' by this flag's own default. Applied AFTER --env resolves, so it always wins when given explicitly. Recorded in training.reward_mode (never null, even under --resume, since it also governs the tournament step)."
        ),
    )
    parser.add_argument(
        "--label",
        type=_parse_label,
        default=None,
        help=(
            "run label recorded in the summary JSON and appended to the default "
            "summary filename (phase4_rl_confirm_<date>_<label>.json); must match "
            "[A-Za-z0-9_-]+"
        ),
    )
    parser.add_argument(
        "--tourney-seeds",
        type=int,
        default=DEFAULT_TOURNEY_SEEDS,
        help="shared seed-set size K for the tournament",
    )
    parser.add_argument(
        "--tourney-episodes",
        type=int,
        default=DEFAULT_TOURNEY_EPISODES,
        help="episodes per head-to-head seed",
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=DEFAULT_BOOTSTRAP_RESAMPLES,
        help="bootstrap CI resamples",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="summary JSON path (default:)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
