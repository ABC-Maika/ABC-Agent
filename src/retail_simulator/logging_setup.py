"""Shared console-logging setup for every entry point.

Every long-running process in this application (training scripts, the
``retail-sim`` CLI, the live server, the tournament, the ladder, the arena
competition) configures itself through :func:`configure_logging`, so the
operator always sees timestamped, aligned progress on stderr:

* stdout is reserved for results (tables, JSON, the run directory path);
  progress goes to stderr, so it streams live and survives pipes that
  capture only stdout.
* one line per event, fixed-width columns (``[HH:MM:SS] LEVEL  source  msg``),
  colors on a TTY, no colors when piped or ``NO_COLOR`` is set.
* the level can be overridden per invocation with the ``LOG_LEVEL``
  environment variable (``DEBUG``/``INFO`` (default)/``WARNING``/``ERROR``);
  ``retail-sim --quiet/--verbose`` map onto it as well.

The handler attached here is tagged so repeated calls replace it instead of
duplicating it (scripts wrap drivers; tests call entry points repeatedly).
"""

from __future__ import annotations

import logging
import os
import sys

__all__ = ["configure_logging", "human_steps", "resolve_level"]

_DATE_FORMAT = "%H:%M:%S"

# Tag that marks handlers installed by configure_logging so a second call
# replaces the first instead of stacking duplicate handlers.
_HANDLER_TAG = "_retail_sim_console_handler"

# Chatty third-party loggers demoted to WARNING so their transport noise
# (connection churn, deprecations) does not drown run progress.
_QUIET_LOGGERS: tuple[str, ...] = ("asyncio", "websockets", "anyio", "urllib3")

_RESET = "\033[0m"
_LEVEL_COLORS: dict[str, str] = {
    "DEBUG": "\033[38;5;245m",  # dim gray
    "INFO": "\033[32m",  # green
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[1;31m",  # bold red
}

# Short display names: full module paths (retail_simulator.envs.league.runner)
# would eat the line. Longest-prefix match wins.
_SHORT_NAMES: dict[str, str] = {
    "retail_simulator.envs.league": "league",
    "retail_simulator.envs.ladder": "ladder",
    "retail_simulator.envs.parallel_gate": "gate",
    "retail_simulator.envs.parallel_env": "env",
    "retail_simulator.envs.arena_match": "arena",
    "retail_simulator.envs.arena": "arena",
    "retail_simulator.envs.spike": "spike",
    "retail_simulator.envs.curriculum": "curriculum",
    "retail_simulator.envs.learnability": "gate",
    "retail_simulator.envs": "envs",
    "retail_simulator.live": "live",
    "retail_simulator.app": "cli",
    "retail_simulator.core": "core",
    "retail_simulator": "retail-sim",
    "phase4_rl_confirm": "phase4",
    "phase4": "phase4",
    "arena_competition": "arena",
    "tournament": "arena",
}


def _short_name(logger_name: str) -> str:
    """Collapse a logger name to a short source tag (see ``_SHORT_NAMES``)."""
    best_prefix = ""
    for prefix in _SHORT_NAMES:
        if (logger_name == prefix or logger_name.startswith(prefix + ".")) and len(
            prefix
        ) > len(best_prefix):
            best_prefix = prefix
    if best_prefix:
        return _SHORT_NAMES[best_prefix]
    if logger_name.startswith("retail_simulator."):
        return logger_name[len("retail_simulator.") :][:12]
    return (logger_name or "retail-sim")[:12]


class _ConsoleFormatter(logging.Formatter):
    """Fixed-width, TTY-colored formatter: ``[12:04:31] INFO    league  msg``."""

    def __init__(self, color: bool) -> None:
        super().__init__(datefmt=_DATE_FORMAT)
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        if self._color and level in _LEVEL_COLORS:
            level_field = f"{_LEVEL_COLORS[level]}{level:<7}{_RESET}"
        else:
            level_field = f"{level:<7}"
        source = f"{_short_name(record.name):<11}"
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)
        return f"[{self.formatTime(record, _DATE_FORMAT)}] {level_field} {source} {message}"


def resolve_level(level: int | str | None = None) -> int:
    """Resolve the effective logging level.

    Explicit ints pass through; explicit names are case-insensitive;
    ``None`` falls back to the ``LOG_LEVEL`` environment variable, then
    to ``INFO``.
    """
    if level is None:
        raw = os.environ.get("LOG_LEVEL", "").strip()
        if not raw:
            return logging.INFO
        level = raw
    if isinstance(level, int):
        return level
    try:
        return int(logging.getLevelName(level.strip().upper()))
    except ValueError as exc:
        raise ValueError(
            f"unknown log level {level!r}; expected DEBUG, INFO, WARNING, or ERROR"
        ) from exc


def _use_color() -> bool:
    """ANSI colors only on a TTY without NO_COLOR set (stdout OR stderr TTY)."""
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stderr.isatty() or sys.stdout.isatty()


def human_steps(value: float) -> str:
    """Compact step counts for progress lines: ``42`` -> ``42``, ``300000`` -> ``300.0k``."""
    n = int(value)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def configure_logging(level: int | str | None = None) -> int:
    """Attach an aligned, timestamped stderr console handler to the root logger.

    Idempotent: any handler previously installed by this function is
    replaced. Returns the resolved level. Safe to call from entry points
    at process start — it does not touch handler state owned by
    libraries (e.g. SB3's run directory logging).
    """
    resolved = resolve_level(level)
    root = logging.getLogger()
    root.setLevel(resolved)
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            root.removeHandler(handler)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(_ConsoleFormatter(color=_use_color()))
    setattr(stream_handler, _HANDLER_TAG, True)
    root.addHandler(stream_handler)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    return resolved
