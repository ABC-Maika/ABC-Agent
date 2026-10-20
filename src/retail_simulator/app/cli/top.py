"""CLI commands: sweep, learnability-check, validate, calibrate, train, play-human, serve-web."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import typer

from retail_simulator import __version__

from . import _echo, _log, _new_run_dir, _state, _status, app


@app.command()
def sweep(
    grid_points: int = typer.Option(
        11,
        "--grid-points",
        help="Grid points per lever (2-D price x marketing => grid_points^2 cells, "
        "plus a grid_points-cell assortment axis at the joint optimum).",
    ),
    ticks: int = typer.Option(1000, "--ticks", help="Ticks per cell per seed (incl. burn-in)."),
    seeds: int = typer.Option(3, "--seeds", help="Number of seeds to average over."),
) -> None:
    """Run the analytical pricing sweep; write the report + CSVs under ``runs/``."""
    from retail_simulator.envs.sweep import (
        assortment_axis_csv_rows,
        automation_axis_csv_rows,
        expansion_timing_csv_rows,
        format_sweep_table,
        loyalty_axis_csv_rows,
        promotion_axis_csv_rows,
        research_axis_csv_rows,
        run_sweep,
        scm_axis_csv_rows,
        sweep_csv_rows,
    )

    seed_tuple = tuple(range(_state["seed"], _state["seed"] + seeds))
    _log.info(
        "running sweep: 2-D %dx%d price x marketing + %d-cell assortment axis "
        "+ %d-cell promotion axis (steady state) present-in-both-regions "
        "+ expansion-timing leg x %d ticks x %d seeds",
        grid_points,
        grid_points,
        grid_points,
        grid_points,
        ticks,
        seeds,
    )
    result = run_sweep(
        grid_points=grid_points,
        ticks_per_point=ticks,
        seeds=seed_tuple,
        start_present_both=True,
    )

    table = format_sweep_table(result)
    _echo(table)

    run_dir = _new_run_dir("sweep")
    (run_dir / "sweep_report.txt").write_text(table + "\n", encoding="utf-8")
    _write_csv(run_dir / "sweep.csv", sweep_csv_rows(result))
    _write_csv(run_dir / "sweep_assortment.csv", assortment_axis_csv_rows(result))
    _write_csv(run_dir / "sweep_promotion.csv", promotion_axis_csv_rows(result))
    # Expansion-timing leg: one row per full-trajectory policy; empty when the leg
    # was not run — _write_csv handles [].
    _write_csv(run_dir / "sweep_expansion_timing.csv", expansion_timing_csv_rows(result))
    # Research leg: one row per research spend (vs the Balanced NPC); empty when
    # not run — _write_csv handles [].
    _write_csv(run_dir / "sweep_research.csv", research_axis_csv_rows(result))
    # SCM leg: one row per service_level (vs the Discounter in the SCM scenario);
    # empty when not run / single-region — _write_csv handles [].
    _write_csv(run_dir / "sweep_scm.csv", scm_axis_csv_rows(result))
    # Automation leg: one row per constant tier (vs the Discounter in the automation
    # scenario, with the steady-state seated automation_tier); empty when not run —
    # _write_csv handles [].
    _write_csv(run_dir / "sweep_automation.csv", automation_axis_csv_rows(result))
    # Loyalty leg: one row per constant loyalty_spend (vs the Discounter in the
    # loyalty scenario); empty when not run — _write_csv handles [].
    _write_csv(run_dir / "sweep_loyalty.csv", loyalty_axis_csv_rows(result))
    _log.info("sweep artifacts written to %s", run_dir)

    if result.passed:
        _echo(
            _status(
                "PASS",
                "no dominant/degenerate strategy (every lever incl. expansion-timing "
                "+ research value-of-information + SCM + Phase-1.2 automation pays)",
                "green",
            )
        )
    else:
        _echo(_status("FAIL", "degenerate/dominant strategy detected", "red"))
        raise typer.Exit(code=1)


@app.command("learnability-check")
def learnability_check(
    quick: bool = typer.Option(False, "--quick", help="~200k-step smoke (CI); else full ~2M."),
    total_steps: int = typer.Option(
        0, "--total-steps", help="Override the training budget (0 = use --quick/full default)."
    ),
    eval_episodes: int = typer.Option(20, "--eval-episodes", help="Eval episodes per policy."),
    n_envs: int = typer.Option(
        1, "--n-envs", help="Parallel training envs (SubprocVecEnv) — multiplies throughput."
    ),
) -> None:
    """Run the single-agent learnability gate; print the margin + PASS/FAIL."""
    from retail_simulator.envs.learnability import (
        FULL_TOTAL_STEPS,
        QUICK_TOTAL_STEPS,
        run_gate,
    )

    if total_steps > 0:
        steps = total_steps
    elif quick:
        steps = QUICK_TOTAL_STEPS
    else:
        steps = FULL_TOTAL_STEPS

    run_dir = _new_run_dir("learnability")
    _log.info(
        "running learnability gate: %d steps, %d eval episodes, %d env(s)",
        steps,
        eval_episodes,
        n_envs,
    )
    report = run_gate(
        total_steps=steps,
        eval_episodes=eval_episodes,
        seed=_state["seed"],
        n_envs=n_envs,
        report_path=run_dir / "gate_report.json",
        progress=_state["verbose"],
    )

    _echo(_format_gate_summary(report))
    _log.info("gate artifacts written to %s", run_dir)

    if report.passed:
        _echo(_status("PASS", f"PPO beats random by {report.margin:.2f} (>= 2 sigma)", "green"))
    else:
        _echo(
            _status(
                "FAIL",
                f"margin {report.margin:.2f} vs threshold {report.threshold:.2f} "
                f"(beats_random={report.beats_random}, trends_up={report.curve_trends_up})",
                "red",
            )
        )
        raise typer.Exit(code=1)


@app.command()
def validate(
    scenario: Path = typer.Argument(..., help="Path to the scenario YAML to validate."),
) -> None:
    """Load + validate a scenario YAML; print its operator-facing summary."""
    from retail_simulator.app import ScenarioConfig
    from retail_simulator.app.models import ScenarioValidationError, load_scenario

    try:
        core_config = load_scenario(scenario)
    except ScenarioValidationError as exc:
        # Aggregated per-field error (formatted by the loader): to stderr + a FAIL line.
        typer.echo(str(exc), err=True)
        _echo(_status("FAIL", f"scenario {scenario} is invalid", "red"))
        raise typer.Exit(code=1) from exc
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        _echo(_status("FAIL", f"scenario file not found: {scenario}", "red"))
        raise typer.Exit(code=1) from exc

    # CoreConfig is the load-bearing artifact; the surface is re-parsed only to
    # report the operator's fields.
    surface = ScenarioConfig.from_yaml(scenario)
    n_regions = len(core_config.demand.regions) if core_config.demand.regions else 1
    _echo(_status("INFO", f"scenario {surface.name!r}", "blue"))
    _echo(_format_scenario_summary(surface, n_regions))
    _echo(_status("PASS", f"scenario {scenario} is valid", "green"))


def _format_scenario_summary(surface: Any, n_regions: int) -> str:
    """Render the validated scenario's operator-facing fields as a small ASCII table."""
    archetypes = ", ".join(surface.npc_archetypes) if surface.npc_archetypes else "(none)"
    rows = [
        ("name", str(surface.name)),
        ("seed", str(surface.seed)),
        ("n_regions", str(n_regions)),
        ("n_learning_agents", str(surface.n_learning_agents)),
        ("npc_archetypes", archetypes),
        ("reward mode", str(surface.reward_mode)),
    ]
    lines = [f"{'field':>18} {'value':>20}", "-" * 39]
    lines += [f"{label:>18} {value:>20}" for label, value in rows]
    return "\n".join(lines)


@app.command()
def calibrate(
    seeds: int = typer.Option(5, "--seeds", help="Seeds averaged per archetype (taming noise)."),
    ticks: int = typer.Option(104, "--ticks", help="Scoring horizon per rollout (~2 years)."),
) -> None:
    """Measure each NPC archetype's beat-random margin over several seeds."""
    from retail_simulator.envs.calibrate import calibrate_all

    seed_tuple = tuple(range(_state["seed"], _state["seed"] + seeds))
    _log.info("calibrating NPC archetypes: %d seeds x %d ticks on the fast seam", seeds, ticks)
    results = calibrate_all(seeds=seed_tuple, horizon=ticks)

    _echo(_format_calibration_table(results))
    out_of_band = [name for name, cal in results.items() if not cal.in_band]
    if out_of_band:
        _echo(
            _status(
                "WARN",
                f"archetype(s) {out_of_band} are out of the 20-65% competence band "
                "(tune their lever constants; this is a report, not a gate failure)",
                "yellow",
            )
        )
    else:
        _echo(_status("INFO", "all archetypes within the 20-65% competence band", "blue"))


def _format_calibration_table(results: dict[str, Any]) -> str:
    """Render the per-archetype beat-random margins as an ASCII table."""
    header = (
        f"{'archetype':>12} {'metric':>14} {'archetype':>12} "
        f"{'random':>12} {'margin':>10} {'in band':>9}"
    )
    lines = [header, "-" * len(header)]
    for name, cal in results.items():
        margin_str = "inf" if cal.margin == float("inf") else f"{cal.margin:.3f}"
        lines.append(
            f"{name:>12} {cal.primary_metric:>14} {cal.archetype_score:>12.4f} "
            f"{cal.random_score:>12.4f} {margin_str:>10} {cal.in_band!s:>9}"
        )
    return "\n".join(lines)


@app.command()
def train(
    total_steps: int = typer.Option(100_000, "--total-steps", help="PPO training timesteps."),
    max_episode_steps: int = typer.Option(
        104, "--max-episode-steps", help="TimeLimit horizon (scoring window)."
    ),
    n_envs: int = typer.Option(
        1, "--n-envs", help="Parallel training envs (SubprocVecEnv) — multiplies throughput."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the plan and exit without training."
    ),
) -> None:
    """Train an SB3 PPO policy, then record a deterministic rollout's metrics."""
    import gymnasium as gym
    from stable_baselines3 import PPO

    import retail_simulator  # noqa: F401  (registers RetailSim-v0)

    if dry_run:
        _echo(
            f"[dry-run] would train PPO for {total_steps} steps "
            f"(max_episode_steps={max_episode_steps}, seed={_state['seed']}, n_envs={n_envs})"
        )
        return

    run_dir = _new_run_dir("train")
    _write_config_receipt(run_dir, total_steps, max_episode_steps)

    # Single env for the post-training metrics rollout (one deterministic trajectory).
    env = gym.make("RetailSim-v0", max_episode_steps=max_episode_steps)
    if n_envs > 1:
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv

        train_env: Any = make_vec_env(
            "RetailSim-v0",
            n_envs=n_envs,
            seed=_state["seed"],
            env_kwargs={"max_episode_steps": max_episode_steps},
            vec_env_cls=SubprocVecEnv,
            vec_env_kwargs={"start_method": "fork"},  # robust regardless of launcher
        )
    else:
        train_env = env
    _log.info("training PPO for %d steps (%d env(s)) -> %s", total_steps, n_envs, run_dir)
    model = PPO("MlpPolicy", train_env, seed=_state["seed"], verbose=1 if _state["verbose"] else 0)
    model.learn(total_timesteps=total_steps, progress_bar=False)
    if train_env is not env:
        train_env.close()
    model.save(str(run_dir / "policy.zip"))

    summary = _rollout_and_record_metrics(model, env, run_dir, max_episode_steps)
    env.close()

    (run_dir / "run_summary.txt").write_text(summary, encoding="utf-8")
    _echo(_status("[OK]", f"trained policy + metrics written to {run_dir}", "green"))
    _echo(str(run_dir))


def _rollout_and_record_metrics(
    model: Any,
    env: Any,
    run_dir: Path,
    max_episode_steps: int,
) -> str:
    """Roll out the trained policy once; write metrics.jsonl + metrics.csv, return a summary."""
    import time

    fieldnames = [
        "tick",
        "episode",
        "episode_step",
        "reward",
        "revenue",
        "profit",
        "market_share",
        "loyalty",
        "cash",
        "stores",
        "price_index",
        "assortment",
        "promotion",
        "stockpile",
        "expansion_action",
        "research_action",
        "research_fidelity",
        # SCM action this tick + per-region stockout rate + seated per-region
        # service_score (pure reads, no RNG).
        "service_level_action",
        "stockout_rate_region_0",
        "stockout_rate_region_1",
        "service_score_region_0",
        "service_score_region_1",
        "automation_tier_action",
        "automation_tier",
        "last_automation_tier_action",
        "automation_capex",
        # Decoded loyalty-program intensity this tick (0.0 at the default).
        "loyalty_spend_action",
        "npc_market_share",
        "seed",
        "algo",
        "wall_time",
    ]
    rows: list[dict[str, Any]] = []
    obs, info = env.reset(seed=_state["seed"])
    done = False
    total_reward = 0.0
    start = time.time()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        rows.append(
            {
                "tick": int(info.get("tick", 0)),
                "episode": 0,
                "episode_step": int(info.get("episode_step", 0)),
                "reward": float(reward),
                "revenue": float(info.get("revenue", 0.0)),
                "profit": float(info.get("profit", 0.0)),
                "market_share": float(info.get("market_share", 0.0)),
                "loyalty": float(info.get("loyalty", 0.0)),
                "cash": float(info.get("cash", 0.0)),
                # Multi-region open-store count (tracks expansion).
                "stores": int(info.get("stores", 0)),
                "price_index": float(np.asarray(action).reshape(-1)[0]),
                # Decoded assortment breadth this tick (0.0 at reset).
                "assortment": float(info.get("assortment", 0.0)),
                # Decoded promotion intensity this tick (0.0 at reset) + seated
                # regional stockpile debt.
                "promotion": float(info.get("promotion", 0.0)),
                "stockpile": float(info.get("stockpile", 0.0)),
                # Decoded expansion choice this tick (0 = no-op, r = open region r;
                # 0.0 at reset).
                "expansion_action": float(info.get("expansion_action", 0.0)),
                # Decoded research spend this tick + seated perception fidelity
                # (0.0 at reset).
                "research_action": float(info.get("research_action", 0.0)),
                "research_fidelity": float(info.get("research_fidelity", 0.0)),
                # SCM action this tick + per-region stockout rate + seated
                # service_score (pure reads, no RNG; 1.0/0.0 sentinels at reset).
                "service_level_action": float(info.get("service_level_action", 1.0)),
                "stockout_rate_region_0": float(info.get("stockout_rate_region_0", 0.0)),
                "stockout_rate_region_1": float(info.get("stockout_rate_region_1", 0.0)),
                "service_score_region_0": float(info.get("service_score_region_0", 1.0)),
                "service_score_region_1": float(info.get("service_score_region_1", 1.0)),
                "automation_tier_action": float(info.get("automation_tier_action", 0.0)),
                "automation_tier": int(info.get("automation_tier", 0)),
                "last_automation_tier_action": int(info.get("last_automation_tier_action", 0)),
                "automation_capex": float(info.get("automation_capex", 0.0)),
                "loyalty_spend_action": float(info.get("loyalty_spend_action", 0.0)),
                "npc_market_share": float(1.0 - float(info.get("market_share", 0.0))),
                "seed": int(_state["seed"]),
                "algo": "PPO",
                "wall_time": round(time.time() - start, 6),
            }
        )
        done = bool(terminated) or bool(truncated)

    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    _write_csv(run_dir / "metrics.csv", rows, fieldnames=fieldnames)

    return (
        f"algo: PPO\nseed: {_state['seed']}\nmax_episode_steps: {max_episode_steps}\n"
        f"eval_episode_steps: {len(rows)}\neval_total_reward: {total_reward:.4f}\n"
        f"final_cash: {rows[-1]['cash'] if rows else 0.0:.2f}\n"
    )


def _write_config_receipt(run_dir: Path, total_steps: int, max_episode_steps: int) -> None:
    """Write ``config.json`` — the reproducibility receipt (seed + versions)."""
    import gymnasium
    import numpy
    import stable_baselines3

    from retail_simulator.core.config import CoreConfig

    receipt = {
        "seed": _state["seed"],
        "total_steps": total_steps,
        "max_episode_steps": max_episode_steps,
        "algo": "PPO",
        "policy": "MlpPolicy",
        "env_id": "RetailSim-v0",
        "core_config": asdict(CoreConfig.default()),
        "versions": {
            "retail_simulator": __version__,
            "numpy": numpy.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
    }
    (run_dir / "config.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    """Write a list of flat dicts as CSV (pandas-readable). Empty => header-less."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = fieldnames if fieldnames is not None else list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _format_gate_summary(report: Any) -> str:
    """Render the gate report as an ASCII metric table."""
    lines = [
        f"{'metric':>22} {'value':>14}",
        "-" * 38,
        f"{'PPO mean reward':>22} {report.ppo_mean_reward:>14.2f}",
        f"{'random mean reward':>22} {report.random_mean_reward:>14.2f}",
        f"{'random std reward':>22} {report.random_std_reward:>14.2f}",
        f"{'margin':>22} {report.margin:>14.2f}",
        f"{'threshold (2 sigma)':>22} {report.threshold:>14.2f}",
        f"{'beats random':>22} {report.beats_random!s:>14}",
        f"{'curve trends up':>22} {report.curve_trends_up!s:>14}",
        f"{'total steps':>22} {report.total_steps:>14}",
    ]
    uptake = getattr(report, "expansion_uptake", None)
    if uptake is not None:
        median = uptake.median_open_tick
        median_str = f"{median:.1f}" if median is not None else "never"
        lines.append(f"{'expansion uptake':>22} {uptake.fraction_opening:>14.3f}")
        lines.append(f"{'median open tick':>22} {median_str:>14}")
        if uptake.fraction_opening == 0.0:
            # Never opening region 1 is a calibration smell, not an auto-fail.
            lines.append(
                "note: PPO never opened region 1 (uptake 0.0) — expansion is "
                "learnable-to-ignore; a calibration smell to review (not a gate failure)"
            )
    return "\n".join(lines)


@app.command("play-human")
def play_human(
    seed: int = typer.Option(0, "--seed", help="Seed: picks the hidden opponent + the world."),
    ticks: int = typer.Option(50, "--ticks", help="Game length in ticks (default 50)."),
    debrief: Path = typer.Option(
        Path("hotseat_debrief.jsonl"),
        "--debrief",
        help="Local .jsonl file the game record is appended to.",
    ),
) -> None:
    """Play a hot-seat game in the terminal and append a debrief record."""
    # Lazy import: keeps the env/pettingzoo stack out of CLI module import.
    from retail_simulator.envs.hotseat import (
        append_debrief,
        play_hotseat_game,
        print_rps_primer,
    )
    from retail_simulator.envs.spike import calibrated_spike_config

    if ticks < 1:
        _echo(_status("FAIL", f"--ticks must be >= 1, got {ticks}", "red"))
        raise typer.Exit(code=2)

    def _input(prompt: str) -> str:
        return input(prompt)

    def _output(message: str) -> None:
        typer.echo(message)

    print_rps_primer(_output)
    result = play_hotseat_game(
        config=calibrated_spike_config(),
        seed=seed,
        n_ticks=ticks,
        input_fn=_input,
        output_fn=_output,
    )
    append_debrief(result, debrief)
    _echo(_status("PASS", f"game recorded to {debrief}", "green"))


@app.command("serve-web")
def serve_web(
    host: str = typer.Option("127.0.0.1", "--host", help="HTTP bind host (default 127.0.0.1)."),
    port: int = typer.Option(8000, "--port", help="HTTP bind port (default 8000)."),
    dist_dir: Path = typer.Option(
        None,
        "--dist-dir",
        help=(
            "Built SPA dir to serve same-origin (default: frontend/dist at the repo root). "
            "Build it first with `npm --prefix frontend run build`. If absent, the API still "
            "serves and `/` shows a 'not built' note."
        ),
    ),
) -> None:
    """Serve the built SPA + the FastAPI JSON API as one same-origin app via uvicorn.

    Build the SPA first (``npm --prefix frontend run build``). Single-process only:
    sessions are in-memory, so the server runs with ``workers=1``.
    """
    # Lazy import: uvicorn is a heavy extra, paid for only when serving.
    import uvicorn

    # The dist dir reaches the app factory via env var (uvicorn owns the app lifecycle).
    if dist_dir is not None:
        os.environ["RETAIL_SIM_WEB_DIST"] = str(dist_dir)

    _log.info("serving web platform on http://%s:%d (single-process, workers=1)", host, port)
    # workers=1 is load-bearing: the in-memory SessionManager is per-process.
    uvicorn.run("retail_simulator.web.app:app", host=host, port=port, workers=1)


_PER_AGENT_METRIC_FIELDS: tuple[str, ...] = (
    "reward",
    "profit",
    "market_share",
    "loyalty",
    "cash",
    "stores",
    "expansion_action",
    # Research: decoded spend + seated perception fidelity.
    "research_action",
    "research_fidelity",
    # SCM: decoded service, per-region stockout rate, seated service_score.
    "service_level_action",
    "stockout_rate_region_0",
    "stockout_rate_region_1",
    "service_score_region_0",
    "service_score_region_1",
    "automation_tier_action",
    "automation_tier",
    "last_automation_tier_action",
    "automation_capex",
    "loyalty_spend_action",
)


def _scenario_seat_plan(scenario: Path | None) -> tuple[int, str]:
    """Resolve ``(n_learning_agents, reward_mode)`` from a scenario YAML (or defaults).

    Only those two fields are threaded to the gate; the scenario's full per-region
    economics are not forwarded. Absent a scenario: the competition defaults (2
    learning agents, weighted reward).
    """
    if scenario is None:
        from retail_simulator.envs.parallel_gate import DEFAULT_N_LEARNING_AGENTS

        return DEFAULT_N_LEARNING_AGENTS, "weighted"

    from retail_simulator.app import ScenarioConfig
    from retail_simulator.app.models import load_scenario

    load_scenario(scenario)  # validate at the boundary (raises ScenarioValidationError)
    surface = ScenarioConfig.from_yaml(scenario)
    return surface.n_learning_agents, surface.reward_mode


