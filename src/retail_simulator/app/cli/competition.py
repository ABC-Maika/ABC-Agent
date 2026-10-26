"""CLI commands: competition run, replay, register, leaderboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import typer

from . import _echo, _log, _new_run_dir, _state, _status, competition_app
from .top import _format_gate_summary, _scenario_seat_plan


@competition_app.command("run")
def competition_run(
    scenario: Path = typer.Option(
        None, "--scenario", help="Scenario YAML (threads n_learning_agents + reward_mode)."
    ),
    total_steps: int = typer.Option(
        0, "--total-steps", help="Override the shared-policy training budget (0 = default)."
    ),
    n_envs: int = typer.Option(
        1, "--n-envs", help="Parallel vec env copies (num_vec_envs) — multiplies throughput."
    ),
    eval_episodes: int = typer.Option(20, "--eval-episodes", help="Eval episodes per policy."),
    quick: bool = typer.Option(False, "--quick", help="~200k-step smoke (CI); else full ~2M."),
    regional_breakdown: bool = typer.Option(
        False,
        "--regional-breakdown",
        help="Add a per-region split to the per-agent metrics.jsonl (opt-in; off by default).",
    ),
    research_gate: bool = typer.Option(
        False,
        "--research-gate",
        help="Value-of-information gate: PPO WITH research vs the SAME policy with research FORCED OFF, in a VARYING-opponent substrate.",
    ),
    research_opponent: str = typer.Option(
        "wandering",
        "--research-opponent",
        help="The varying opponent the research gate runs against (NEVER the constant discounter): default 'wandering' (exogenous, non-inferable); 'balanced' is the partly-inferable reported-secondary.",
    ),
    scm_gate: bool = typer.Option(
        False,
        "--scm-gate",
        help="Value-of-SCM gate: PPO WITH SCM vs the SAME policy with SCM FORCED-TO-DEFAULT (service_level=1.0) in the SCM scenario substrate (scm_gate_config — cogs_premium>0 AND lost_sales_share_penalty>0 AND CONVENIENCE-beta_service>0) vs the default Discounter.",
    ),
    automation_gate: bool = typer.Option(
        False,
        "--automation-gate",
        help="Value-of-automation gate: PPO WITH automation vs the SAME policy with the 3-wide automation block FORCED-TO-TIER-0 (the no-upgrade baseline + the byte-identity anchor) in the automation scenario substrate (automation_gate_config — capex_per_tier>0 AND savings_per_tier>0) vs the default Discounter.",
    ),
    loyalty_gate: bool = typer.Option(
        False,
        "--loyalty-gate",
        help="Value-of-active-loyalty-program gate: PPO WITH loyalty vs the SAME policy with loyalty_spend FORCED-TO-0.0 (the no-program baseline + the byte-identity anchor) in the loyalty scenario substrate (loyalty_gate_config — cost_per_unit_spend>0 AND BRAND_LOYAL beta_program>0) vs the default Discounter.",
    ),
    curriculum: str = typer.Option(
        "",
        "--curriculum",
        help="Training-time scenario curriculum name (e.g. 'phase_1_4_basic' — the 6-segment 3-variant curriculum). Training samples a fresh CoreConfig from the named distribution at each episode reset (a SEPARATE rng — NEVER the core PCG64); the eval / gate scenario is unchanged. Empty (the default) = no curriculum.",
    ),
    policy_kind: str = typer.Option(
        "mlp",
        "--policy-kind",
        help="Policy class — 'mlp' (default; vanilla SB3 PPO, preserves the default byte-identity) or 'lstm' (RecurrentPPO with MlpLstmPolicy; requires the [recurrent] extra — sb3-contrib). The lstm path is opt-in; the default 'mlp' threads through to retail_simulator.envs.parallel_gate._make_shared_ppo unchanged.",
    ),
) -> None:
    """Run the multi-agent competition gate; write the report + per-agent metrics."""
    from retail_simulator.envs.curriculum import (
        ScenarioDistribution,
        build_curriculum,
        known_curricula,
    )
    from retail_simulator.envs.learnability import FULL_TOTAL_STEPS, QUICK_TOTAL_STEPS
    from retail_simulator.envs.parallel_gate import diversity_report, run_parallel_gate

    n_learning_agents, reward_mode = _scenario_seat_plan(scenario)
    if policy_kind not in ("mlp", "lstm"):
        _echo(
            _status(
                "[FAIL]",
                f"unknown policy-kind {policy_kind!r}; expected 'mlp' or 'lstm'",
                "red",
            )
        )
        raise typer.Exit(code=1)
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

    if total_steps > 0:
        steps = total_steps
    elif quick:
        steps = QUICK_TOTAL_STEPS
    else:
        steps = FULL_TOTAL_STEPS

    run_dir = _new_run_dir("competition")
    if research_gate:
        gate_label = "research-on-vs-off"
    elif scm_gate:
        gate_label = "scm-on-vs-off"
    elif automation_gate:
        gate_label = "automation-on-vs-off"
    elif loyalty_gate:
        gate_label = "loyalty-on-vs-off"
    else:
        gate_label = "self-play vs random"
    curriculum_suffix = f", curriculum={curriculum}" if curriculum else ""
    _log.info(
        "running multi-agent gate (%s): %d steps, %d eval episodes, %d learning agent(s), "
        "%d vec env(s), reward=%s%s%s",
        gate_label,
        steps,
        eval_episodes,
        n_learning_agents,
        n_envs,
        reward_mode,
        f", opponent={research_opponent}" if research_gate else "",
        curriculum_suffix,
    )
    policy_kind_literal = cast(Literal["mlp", "lstm"], policy_kind)
    report = run_parallel_gate(
        total_steps=steps,
        eval_episodes=eval_episodes,
        n_learning_agents=n_learning_agents,
        seed=_state["seed"],
        num_vec_envs=n_envs,
        reward_mode=reward_mode,
        research_gate=research_gate,
        research_opponent=research_opponent,
        scm_gate=scm_gate,
        automation_gate=automation_gate,
        loyalty_gate=loyalty_gate,
        curriculum=selected_curriculum,
        policy_kind=policy_kind_literal,
    )

    # Write the gate report (GateReport.to_dict is JSON-serializable).
    (run_dir / "gate_report.json").write_text(
        json.dumps(report.to_dict(), indent=2), encoding="utf-8"
    )

    # Per-agent metrics.jsonl: a random-field rollout (the gate returns the report
    # only; learned scores live in gate_report.json).
    _write_competition_metrics(
        run_dir,
        n_learning_agents=n_learning_agents,
        reward_mode=reward_mode,
        regional_breakdown=regional_breakdown,
    )

    _echo(_format_gate_summary(report))
    roster = {"trained_selfplay": report.ppo_mean_reward, "random": report.random_mean_reward}
    _echo(_format_diversity_report(diversity_report(roster)))
    _log.info("competition artifacts written to %s", run_dir)

    if report.passed:
        if research_gate:
            winner, beaten = "research-on", "research-off"
        elif scm_gate:
            winner, beaten = "scm-on", "scm-off-at-default"
        elif automation_gate:
            winner, beaten = "automation-on", "automation-off-at-tier-0"
        elif loyalty_gate:
            winner, beaten = "loyalty-on", "loyalty-off-at-zero"
        else:
            winner, beaten = "self-play", "random"
        _echo(
            _status(
                "PASS",
                f"{winner} beats {beaten} by {report.margin:.2f} (>= 2 sigma)",
                "green",
            )
        )
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


@competition_app.command("replay")
def competition_replay(
    report: Path = typer.Argument(
        ..., help="A gate_report.json (or a competition run dir containing one)."
    ),
) -> None:
    """Re-print a saved competition gate report (a file or a run dir) — exit 0/1 on PASS/FAIL."""
    report_path = report / "gate_report.json" if report.is_dir() else report
    if not report_path.exists():
        typer.echo(f"gate report not found: {report_path}", err=True)
        _echo(_status("FAIL", f"no gate report at {report_path}", "red"))
        raise typer.Exit(code=1)

    data = json.loads(report_path.read_text(encoding="utf-8"))
    _echo(_format_saved_gate_report(data))
    if bool(data.get("passed")):
        _echo(_status("PASS", f"saved report at {report_path} PASSED", "green"))
    else:
        _echo(_status("FAIL", f"saved report at {report_path} FAILED", "red"))
        raise typer.Exit(code=1)


@competition_app.command("register")
def competition_register(
    policy: Path = typer.Argument(..., help="A trained policy.zip to register as an entrant."),
) -> None:
    """Print that registration (persistent roster) is not available yet."""
    _echo(
        _status(
            "INFO",
            f"register is a Phase-2 feature (persistent competition roster); {policy} "
            "was NOT registered. 0.5 has no persistence — use `competition run` + "
            "`competition replay` (gate_report.json) instead.",
            "blue",
        )
    )


@competition_app.command("leaderboard")
def competition_leaderboard() -> None:
    """Print that the leaderboard is not available yet."""
    _echo(
        _status(
            "INFO",
            "leaderboard is a Phase-2 feature (it ranks a persistent entrant roster, "
            "which 0.5 does not build). Use `competition replay <gate_report.json>` for a "
            "run's standing, or `retail-sim calibrate` for archetype competence.",
            "blue",
        )
    )


def _write_competition_metrics(
    run_dir: Path,
    *,
    n_learning_agents: int,
    reward_mode: str,
    regional_breakdown: bool,
) -> None:
    """Roll out a random policy through the parallel env; write per-agent metrics.jsonl."""
    from retail_simulator.core.config import CoreConfig, RewardConfig
    from retail_simulator.envs import RetailParallelEnv

    config = CoreConfig(reward=RewardConfig(mode=reward_mode))
    env = RetailParallelEnv(config, n_learning_agents=n_learning_agents, seed=_state["seed"])
    action_space = env.action_space(env.possible_agents[0])
    action_space.seed(_state["seed"])

    _observations, _infos = env.reset(seed=_state["seed"])
    horizon = 104  # the ~2-year scoring window (matches the gate's eval horizon)
    rows: list[dict[str, Any]] = []
    for _ in range(horizon):
        actions = {aid: action_space.sample() for aid in env.agents}
        _observations, rewards, _term, _trunc, infos = env.step(actions)
        for aid in env.possible_agents:
            info = infos[aid]
            row: dict[str, Any] = {
                "agent": aid,
                "tick": int(info.get("tick", 0)),
                "episode_step": int(info.get("episode_step", 0)),
                "reward": float(rewards[aid]),
                "profit": float(info.get("profit", 0.0)),
                "market_share": float(info.get("market_share", 0.0)),
                "loyalty": float(info.get("loyalty", 0.0)),
                "cash": float(info.get("cash", 0.0)),
                "stores": int(info.get("stores", 0)),
                "expansion_action": float(info.get("expansion_action", 0.0)),
                # Research: decoded spend + seated fidelity.
                "research_action": float(info.get("research_action", 0.0)),
                "research_fidelity": float(info.get("research_fidelity", 0.0)),
                # SCM: decoded service, per-region stockout rate, seated
                # service_score (pure reads, no RNG).
                "service_level_action": float(info.get("service_level_action", 1.0)),
                "stockout_rate_region_0": float(info.get("stockout_rate_region_0", 0.0)),
                "stockout_rate_region_1": float(info.get("stockout_rate_region_1", 0.0)),
                "service_score_region_0": float(info.get("service_score_region_0", 1.0)),
                "service_score_region_1": float(info.get("service_score_region_1", 1.0)),
                # Automation: decoded target tier + seated tier (authoritative for
                # next-tick demand) + last target + tick capex (0 on a no-op).
                "automation_tier_action": float(info.get("automation_tier_action", 0.0)),
                "automation_tier": int(info.get("automation_tier", 0)),
                "last_automation_tier_action": int(info.get("last_automation_tier_action", 0)),
                "automation_capex": float(info.get("automation_capex", 0.0)),
                # Decoded loyalty-program intensity this tick (0.0 at the default).
                "loyalty_spend_action": float(info.get("loyalty_spend_action", 0.0)),
            }
            if regional_breakdown:
                row["regional_breakdown"] = _regional_breakdown_for_seat(env, aid)
            rows.append(row)
    env.close()

    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _regional_breakdown_for_seat(env: Any, agent_id: str) -> list[int]:
    """Per-region store counts for the agent's seat."""
    seat = env._seat_of[agent_id]
    retailer = env._state.retailers[seat]
    return [int(n) for n in retailer.stores_per_region]


def _format_diversity_report(report: Any) -> str:
    """Render the soft diversity report as an ASCII table."""
    ratio = "inf" if report.max_min_ratio == float("inf") else f"{report.max_min_ratio:.2f}"
    lines = [
        f"{'diversity (soft)':>22} {'value':>14}",
        "-" * 38,
        f"{'best score':>22} {report.best_score:>14.2f}",
        f"{'min score':>22} {report.min_score:>14.2f}",
        f"{'fraction within 80%':>22} {report.fraction_within:>14.3f}",
        f"{'count within 80%':>22} {report.n_within:>14}",
        f"{'max/min ratio':>22} {ratio:>14}",
        f"{'meets soft goal':>22} {report.meets_soft_goal!s:>14}",
        "note: diversity is a SOFT goal — REPORTED, never part of any gate's PASS/FAIL",
    ]
    return "\n".join(lines)


def _format_saved_gate_report(data: dict[str, Any]) -> str:
    """Render a saved ``gate_report.json`` dict as the gate summary table (replay)."""
    lines = [
        f"{'metric':>22} {'value':>14}",
        "-" * 38,
        f"{'trained mean':>22} {float(data.get('ppo_mean_reward', 0.0)):>14.2f}",
        f"{'random mean':>22} {float(data.get('random_mean_reward', 0.0)):>14.2f}",
        f"{'random std':>22} {float(data.get('random_std_reward', 0.0)):>14.2f}",
        f"{'margin':>22} {float(data.get('margin', 0.0)):>14.2f}",
        f"{'threshold (2 sigma)':>22} {float(data.get('threshold', 0.0)):>14.2f}",
        f"{'beats random':>22} {data.get('beats_random')!s:>14}",
        f"{'curve trends up':>22} {data.get('curve_trends_up')!s:>14}",
        f"{'total steps':>22} {int(data.get('total_steps', 0)):>14}",
        f"{'passed':>22} {data.get('passed')!s:>14}",
    ]
    return "\n".join(lines)


