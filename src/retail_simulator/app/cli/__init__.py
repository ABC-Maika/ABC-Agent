"""``retail-sim`` — the Typer CLI for the retail-simulator.

* ``check-env``          — run ``gymnasium.utils.env_checker.check_env`` -> [OK]/[FAIL]
* ``train``              — SB3 PPO; write ``runs/<run-id>/{config.json, metrics.jsonl,
                           metrics.csv, policy.zip, run_summary.txt}``
* ``sweep``              — analytical pricing sweep; write a report under ``runs/``
* ``learnability-check`` — run the single-agent gate; print the margin + PASS/FAIL
* ``validate``           — load + validate a scenario YAML (aggregated errors)
* ``calibrate``          — measure each NPC archetype's beat-random margin
* ``competition``        — multi-agent competition sub-app (run|replay|register|leaderboard)
* ``version``            — print the package version

Sub-apps: ``league`` (train|evaluate|generalization-diagnostic|ladder),
``live`` (serve|run-series|replay|play), plus ``play-human`` and ``serve-web``.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import typer

from retail_simulator import __version__

# Global state populated by the root callback and read by subcommands.
_state: dict[str, Any] = {"seed": 42, "quiet": False, "verbose": False, "run_dir": Path("runs")}

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="retail-simulator CLI — check, train, sweep, validate, calibrate, and gate the env.",
)

competition_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Multi-agent competition — run|replay the self-play gate; register|leaderboard.",
)
app.add_typer(competition_app, name="competition")

league_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Phase 1.5 self-play tooling — league train|evaluate|generalization-diagnostic.",
)
app.add_typer(league_app, name="league")

live_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Phase 2.0 demo server — live multiplayer over WebSocket.",
)
app.add_typer(live_app, name="live")

_log = logging.getLogger("retail_simulator")


def _use_color() -> bool:
    """Whether status prefixes may use ANSI colors (TTY, no --quiet, no NO_COLOR)."""
    if _state["quiet"]:
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


# ANSI codes used only for the status prefixes; gated behind _use_color().
_COLORS = {
    "green": "\033[32m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "reset": "\033[0m",
}


def _status(prefix: str, message: str, color: str) -> str:
    """Format a ``[PREFIX] message`` status line, colored only when allowed."""
    if _use_color():
        return f"{_COLORS[color]}{prefix}{_COLORS['reset']} {message}"
    return f"{prefix} {message}"


def _echo(message: str) -> None:
    """Write a result line to stdout (suppressed entirely under --quiet)."""
    if not _state["quiet"]:
        typer.echo(message)


def _configure_logging() -> None:
    """Attach a stderr handler at the resolved level.

    Progress goes to stderr (stdout is reserved for results). ``--quiet`` => errors
    only; ``--verbose`` => DEBUG; default => INFO.
    """
    if _state["quiet"]:
        level = logging.ERROR
    elif _state["verbose"]:
        level = logging.DEBUG
    else:
        level = logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    _log.handlers.clear()
    _log.addHandler(handler)
    _log.setLevel(level)
    _log.propagate = False


def _new_run_dir(kind: str) -> Path:
    """Create and return a fresh ``<run-dir>/<kind>-<timestamp>`` directory."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(_state["run_dir"]) / f"{kind}-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@app.callback()
def main(
    seed: int = typer.Option(42, "--seed", help="Global RNG seed (default 42)."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Errors only; no color/results."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="DEBUG-level logging to stderr."),
    run_dir: Path = typer.Option(
        Path("runs"), "--run-dir", help="Root directory for run artifacts (default ./runs)."
    ),
) -> None:
    """Resolve global options and configure logging before any subcommand runs."""
    _state["seed"] = seed
    _state["quiet"] = quiet
    _state["verbose"] = verbose
    _state["run_dir"] = run_dir
    _configure_logging()


@app.command()
def version() -> None:
    """Print the package version."""
    _echo(__version__)


@app.command("check-env")
def check_env_command() -> None:
    """Run the Gymnasium env checker on ``RetailEnv`` -> [OK]/[FAIL], exit 0/1."""
    import warnings

    from gymnasium.utils.env_checker import check_env

    from retail_simulator.envs.gym_env import RetailEnv

    try:
        with warnings.catch_warnings():
            # The Box-action-space symmetry warning is expected; check_env must still pass.
            warnings.simplefilter("ignore")
            check_env(RetailEnv())
    except Exception as exc:
        _echo(_status("[FAIL]", f"env_checker reported: {exc}", "red"))
        _echo("This is a bug in RetailEnv, not your agent.")
        raise typer.Exit(code=1) from exc
    _echo(_status("[OK]", "RetailEnv passes gymnasium.utils.env_checker.check_env", "green"))


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via a sibling ``.tmp`` + ``os.replace``.

    The temp file is in the same directory, so the rename is same-filesystem
    (atomic on POSIX) and a crash never leaves a partial file.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# Register the command modules (side effect: their @*_app.command defs attach).
from . import competition, ladder_cli, league, live, top  # noqa: F401
