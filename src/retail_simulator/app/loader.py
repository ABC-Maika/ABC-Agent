"""``load_scenario(path)`` — the operator entry point the CLI consumes.

Parses a scenario YAML, validates it at the boundary (collecting ALL errors), and returns the framework-free ``CoreConfig`` with the seat plan on ``CoreConfig.seats``. Invalid input raises :class:`ScenarioValidationError` with a clean aggregated message, never a bare pydantic traceback.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from retail_simulator.core.config import CoreConfig
from retail_simulator.app.models import ScenarioConfig


class ScenarioValidationError(ValueError):
    """A scenario YAML failed boundary validation; carries the aggregated operator-facing message."""


def _format_validation_error(path: str | Path, exc: ValidationError) -> str:
    """Render a pydantic ``ValidationError`` as one aggregated message, one line per problem."""
    lines = [f"invalid scenario config at {path}:"]
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def load_scenario(path: str | Path) -> CoreConfig:
    """Load a scenario YAML and return its validated ``CoreConfig``.

    Missing file → ``FileNotFoundError``; bad config → :class:`ScenarioValidationError`.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"scenario file not found: {path}")
    try:
        scenario = ScenarioConfig.from_yaml(path)
    except ValidationError as exc:
        raise ScenarioValidationError(_format_validation_error(path, exc)) from exc
    return scenario.to_core_config()


__all__ = ["ScenarioValidationError", "load_scenario"]
