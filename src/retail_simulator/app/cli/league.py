"""CLI commands: league train, evaluate, generalization-diagnostic."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal, cast

import typer

# Module-load import: Typer resolves Option defaults at function-definition time.
from retail_simulator.envs.league import DEFAULT_LEAGUE_NAME

from . import _echo, _log, _status, league_app


@league_app.command("train")
def league_train(
    pool_dir: Path = typer.Option(
        ...,
        "--pool-dir",
        help="On-disk pool directory (created if missing). The seeded archetypes + each "
        "round's SB3 checkpoint zip are written here; `league evaluate` reads it back.",
    ),
    rounds: int = typer.Option(
        3,
        "--rounds",
        help="Number of league rounds (>=1). Round-0 is the random-init snapshot (no "
        "`.learn` call — the AlphaStar canonical seed); rounds 1..N each train a fresh "
        "shared PPO against an opponent sampled from the GROWN pool.",
    ),
    train_steps_per_round: int = typer.Option(
        500_000,
        "--train-steps-per-round",
        help="PPO `learn` budget per round-1..N (round-0 does not call `.learn`).",
    ),
    league_seed: int = typer.Option(
        0,
        "--league-seed",
        help="Seeds the league's opponent-picker Generator (kept SEPARATE from the curriculum + core RNGs — three-RNG isolation contract).",
    ),
    curriculum_seed: int = typer.Option(
        0,
        "--curriculum-seed",
        help="Seeds the curriculum wrapper's harness Generator under F3-A composition.",
    ),
    core_seed: int = typer.Option(
        42,
        "--core-seed",
        help="Seeds the core PCG64 + the SB3 PPO constructor (training reproducibility).",
    ),
    curriculum: str = typer.Option(
        "",
        "--curriculum",
        help="Phase 1.4 curriculum NAME (e.g. 'phase_1_4_basic') to compose under the "
        "league (F3-A stacked). Empty (default) = no curriculum.",
    ),
    league: str = typer.Option(
        DEFAULT_LEAGUE_NAME,
        "--league",
        help="Named league (pool seed). Defaults to the F5-A seeded basic league.",
    ),
    distribution: str = typer.Option(
        "uniform",
        "--distribution",
        help=(
            "Opponent distribution: 'uniform' (default; backward compat with Slice B) "
            "or 'pfsp' (Slice C; win-rate-weighted via PFSPOpponentDistribution)."
        ),
    ),
    pfsp_alpha: float = typer.Option(
        2.0,
        "--pfsp-alpha",
        help=(
            "PFSP hardness exponent for (1-x)^alpha weighting; ignored under "
            "--distribution uniform."
        ),
    ),
    round_end_eval_episodes: int = typer.Option(
        5,
        "--round-end-eval-episodes",
        help=(
            "Episodes per opponent in the round-end head-to-head eval (populates "
            "the win-rate matrix)."
        ),
    ),
    reset_pool: bool = typer.Option(
        False,
        "--reset-pool",
        help=(
            "Before running, clear the pool dir of any prior win_rate_matrix.json "
            "+ round_*.zip artifacts (operator UX for fresh PFSP runs)."
        ),
    ),
    policy_kind: str = typer.Option(
        "mlp",
        "--policy-kind",
        help="Phase 1.6 (F-LOCUS=A): policy class for every per-round _make_shared_ppo call — 'mlp' (default; vanilla SB3 PPO, preserves every Slice C byte-identity) or 'lstm' (RecurrentPPO with MlpLstmPolicy; requires the [recurrent] extra — sb3-contrib). The lstm path is opt-in; the default 'mlp' threads through to envs.league.run_league_training unchanged.",
    ),
) -> None:
    """Train rounds of PPO against a growing opponent pool; write the pool + a JSON result."""
    from retail_simulator.envs.curriculum import (
        ScenarioDistribution,
        build_curriculum,
        known_curricula,
    )
    from retail_simulator.envs.league import (
        FrozenPolicyPool,
        OpponentDistribution,
        PFSPOpponentDistribution,
        UniformOpponentDistribution,
        build_league,
        known_leagues,
        run_league_training,
    )

    # Validate operator inputs up front (clean [FAIL] + exit 1, no traceback)
    # before any heavy work.
    if distribution not in ("uniform", "pfsp"):
        _echo(
            _status(
                "[FAIL]",
                f"unknown distribution {distribution!r}; expected 'uniform' or 'pfsp'",
                "red",
            )
        )
        raise typer.Exit(code=1)
    if policy_kind not in ("mlp", "lstm"):
        _echo(
            _status(
                "[FAIL]",
                f"unknown policy-kind {policy_kind!r}; expected 'mlp' or 'lstm'",
                "red",
            )
        )
        raise typer.Exit(code=1)
    if pfsp_alpha < 0:
        _echo(
            _status(
                "[FAIL]",
                f"--pfsp-alpha must be >= 0, got {pfsp_alpha}",
                "red",
            )
        )
        raise typer.Exit(code=1)
    if round_end_eval_episodes < 1:
        _echo(
            _status(
                "[FAIL]",
                (f"--round-end-eval-episodes must be >= 1, got {round_end_eval_episodes}"),
                "red",
            )
        )
        raise typer.Exit(code=1)

    # Reset-pool: clear prior win-rate matrix + round_*.zip artifacts (the manifest
    # is left alone; build_league rebuilds the pool from scratch).
    if reset_pool:
        pool_dir.mkdir(parents=True, exist_ok=True)
        wr_path = pool_dir / "win_rate_matrix.json"
        if wr_path.exists():
            wr_path.unlink()
        for zip_path in pool_dir.glob("round_*.zip"):
            zip_path.unlink()
        _echo(f"--reset-pool: cleared win_rate_matrix.json + round_*.zip from {pool_dir}")

    # Resolve the named league; an unknown name is a clean [FAIL] + exit 1.
    try:
        pool = build_league(league, pool_dir)
    except ValueError as exc:
        _echo(_status("[FAIL]", str(exc), "red"))
        _echo(f"Known leagues: {list(known_leagues())}")
        raise typer.Exit(code=1) from exc

    # Resolve the optional curriculum (training-time scenario distribution).
    selected_curriculum: ScenarioDistribution | None
    if curriculum:
        try:
            selected_curriculum = build_curriculum(curriculum)
        except ValueError as exc:
            _echo(_status("[FAIL]", str(exc), "red"))
            _echo(f"Known curricula: {list(known_curricula())}")
            raise typer.Exit(code=1) from exc
    else:
        selected_curriculum = None

    _log.info(
        "running league training: league=%s, rounds=%d, train_steps_per_round=%d, "
        "seeds=(league=%d, curriculum=%d, core=%d), curriculum=%s, "
        "distribution=%s, pfsp_alpha=%s, round_end_eval_episodes=%d, pool_dir=%s",
        league,
        rounds,
        train_steps_per_round,
        league_seed,
        curriculum_seed,
        core_seed,
        curriculum or "none",
        distribution,
        pfsp_alpha,
        round_end_eval_episodes,
        pool_dir,
    )

    # Resolve the opponent distribution once; the factory closure returns the same
    # instance each round (cross-round growth is handled inside the harness).
    dist_instance: OpponentDistribution
    if distribution == "uniform":
        dist_instance = UniformOpponentDistribution()
    elif distribution == "pfsp":
        dist_instance = PFSPOpponentDistribution(alpha=pfsp_alpha)
    else:
        # Defensive: unreachable unless a new distribution value is added.
        _echo(_status("[FAIL]", f"unknown distribution {distribution!r}", "red"))
        raise typer.Exit(code=1)

    def _distribution_factory(_pool: FrozenPolicyPool) -> OpponentDistribution:
        return dist_instance

    # Pre-run checkpoint count (kind=="checkpoint" only) so the final echo can
    # report pool growth.
    n_existing_checkpoints = sum(1 for m in pool.members() if m.kind == "checkpoint")

    policy_kind_literal = cast(Literal["mlp", "lstm"], policy_kind)
    result = run_league_training(
        initial_pool=pool,
        n_rounds=rounds,
        distribution_factory=_distribution_factory,
        train_steps_per_round=train_steps_per_round,
        seeds=(league_seed, curriculum_seed, core_seed),
        scenario_distribution=selected_curriculum,
        distribution=dist_instance,
        pfsp_alpha=pfsp_alpha,
        round_end_eval_episodes=round_end_eval_episodes,
        policy_kind=policy_kind_literal,
    )

    # Final pool size, echoed before the JSON so the payload stays clean.
    _echo(
        f"final pool size: {len(result.checkpoints)} checkpoints "
        f"(was {n_existing_checkpoints} before this run)"
    )

    # The snapshot has tuple-keyed dicts (not JSON-native); strip it from the
    # payload and print its entry count + on-disk path.
    printable_result = replace(result, win_rate_matrix_snapshot=None)
    snapshot = result.win_rate_matrix_snapshot
    if snapshot is not None:
        _echo(
            f"win_rate_matrix_snapshot: {len(snapshot)} entries at "
            f"{result.pool_dir / 'win_rate_matrix.json'}"
        )

    # default=str handles non-JSON-native types (Paths, PoolMember).
    _echo(json.dumps(asdict(printable_result), indent=2, default=str))


@league_app.command("evaluate")
def league_evaluate(
    pool_dir: Path = typer.Option(
        ...,
        "--pool-dir",
        help="On-disk pool directory previously written by `league train`.",
    ),
    n_episodes: int = typer.Option(
        20,
        "--n-episodes",
        help="Position-controlled head-to-head episode budget for the round-N vs round-1 comparison (-A).",
    ),
    seed: int = typer.Option(
        0,
        "--seed",
        help="Base seed for the head-to-head episodes (per-episode seed = base + index).",
    ),
) -> None:
    """Evaluate a trained pool's round-1 vs final-round improvement; print a JSON report."""
    from retail_simulator.envs.arena import arena_lock_config
    from retail_simulator.envs.league import (
        FrozenPolicyPool,
        LeagueTrainingResult,
        evaluate_pool_snapshot_improvement,
    )

    # Missing pool is operator error — surface cleanly and exit 1.
    try:
        pool = FrozenPolicyPool.load(pool_dir)
    except FileNotFoundError as exc:
        _echo(_status("[FAIL]", str(exc), "red"))
        raise typer.Exit(code=1) from exc

    # A home_regions-bearing pool was trained under the arena — evaluate it on
    # arena_lock_config(), not the default world (wrong geometry).
    home_regions = pool.home_regions
    eval_config = arena_lock_config() if home_regions is not None else None

    # Only checkpoints are consumed; the other fields are zero-valued placeholders,
    # so evaluate needs no re-training.
    checkpoints = tuple(
        sorted(
            (m for m in pool.members() if m.kind == "checkpoint"),
            key=lambda m: m.round_index,
        )
    )
    training_result = LeagueTrainingResult(
        pool_dir=pool.pool_dir,
        rounds_completed=len(checkpoints),
        checkpoints=checkpoints,
        learning_curves=(),
        seeds=(0, 0, 0),
        train_steps_per_round=0,
    )

    _log.info(
        "evaluating pool snapshot improvement: pool_dir=%s, checkpoints=%d, n_episodes=%d",
        pool_dir,
        len(checkpoints),
        n_episodes,
    )
    try:
        report = evaluate_pool_snapshot_improvement(
            pool,
            training_result,
            n_eval_episodes=n_episodes,
            seed=seed,
            config=eval_config,
            home_regions=home_regions,
        )
    except ValueError as exc:
        _echo(_status("[FAIL]", str(exc), "red"))
        raise typer.Exit(code=1) from exc

    _echo(json.dumps(asdict(report), indent=2, default=str))


@league_app.command("generalization-diagnostic")
def league_generalization(
    curriculum: str = typer.Option(
        "phase_1_4_basic",
        "--curriculum",
        help="Phase 1.4 curriculum NAME the curriculum-trained PPO is trained against.",
    ),
    train_steps: int = typer.Option(
        2_000_000,
        "--train-steps",
        help="PPO `learn` budget for BOTH the curriculum-PPO and the fixed-PPO (same "
        "hyperparameters, same budget — the symmetric comparison).",
    ),
    n_eval_episodes: int = typer.Option(
        5,
        "--n-eval-episodes",
        help="Eval episodes per held-out scenario; -B mini-eval budget.",
    ),
    curriculum_seed: int = typer.Option(
        0,
        "--curriculum-seed",
        help="Seeds the curriculum wrapper's harness Generator (SEPARATE from core).",
    ),
    core_seed: int = typer.Option(
        42,
        "--core-seed",
        help="Seeds the core PCG64 + the SB3 PPO constructors (training reproducibility).",
    ),
) -> None:
    """Compare a curriculum-trained PPO against a fixed-policy baseline on held-out scenarios."""
    from retail_simulator.envs.curriculum import (
        build_curriculum,
        known_curricula,
    )
    from retail_simulator.envs.league import (
        default_holdout_scenarios,
        run_real_ppo_generalization_diagnostic,
    )

    try:
        distribution = build_curriculum(curriculum)
    except ValueError as exc:
        _echo(_status("[FAIL]", str(exc), "red"))
        _echo(f"Known curricula: {list(known_curricula())}")
        raise typer.Exit(code=1) from exc

    _log.info(
        "running real-PPO generalization diagnostic: curriculum=%s, train_steps=%d, "
        "n_eval_episodes=%d, seeds=(curriculum=%d, core=%d)",
        curriculum,
        train_steps,
        n_eval_episodes,
        curriculum_seed,
        core_seed,
    )
    report = run_real_ppo_generalization_diagnostic(
        curriculum=distribution,
        holdout_scenarios=default_holdout_scenarios(),
        train_steps=train_steps,
        seeds=(curriculum_seed, core_seed),
        n_eval_episodes_per_scenario=n_eval_episodes,
    )

    _echo(json.dumps(asdict(report), indent=2, default=str))


