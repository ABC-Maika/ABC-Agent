"""Live server configuration, per-connection state, and wire-shaping helpers.

Holds the public dataclasses (:class:`ServeConfig`, :class:`ServeResult`),
the server-internal per-connection state (:class:`_ClientState`), and the
pure helpers that shape the seam's outputs into the wire's JSON-safe form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ServeConfig:
    """Configuration for one live game; built by the CLI and handed to :func:`serve_game`.

    All fields are required so the CLI surfaces every operator choice. ``token=None``
    is ``--no-auth``; ``tick_interval_seconds=0.0`` is CI fast mode;
    ``npc_archetypes=()`` is the all-remote mode.
    """

    n_agents: int
    n_ticks: int
    tick_interval_seconds: float
    host: str
    port: int
    registration_timeout_seconds: float
    token: str | None
    npc_archetypes: tuple[str, ...]
    scenario_name: str
    runs_dir: Path
    seed: int


@dataclass(frozen=True)
class ServeResult:
    """Outcome of one :func:`serve_game` call.

    ``exit_status`` mirrors a Unix exit code: ``0`` clean game, ``1`` registration
    timeout, ``2`` any other unrecoverable error.
    """

    game_id: str
    run_dir: Path
    rounds_completed: int
    exit_status: int
    summary: str
    final_scores: dict[str, float] = field(default_factory=dict)
    winner_seat: int | None = None


@dataclass
class _ClientState:
    """Per-connection mutable state held by the server during one game (keyed by seat_index).

    ``websocket`` is typed ``Any``: the websockets server protocol class is only
    imported under :data:`TYPE_CHECKING` (in :mod:`retail_simulator.live.server.game`).
    """

    seat_index: int
    websocket: Any  # WebSocketServerProtocol; ``Any`` keeps the runtime import lazy.
    client_name: str | None
    agent_name: str
    last_valid_action: npt.NDArray[np.float32] | None = None
    missed_deadlines: int = 0


def _flatten_action_mask(action_mask: Any) -> list[float] | None:
    """Flatten the seam's per-lever ``action_mask`` dict into the wire's flat list.

    Concatenates the per-key arrays in sorted-key order (deterministic layout).
    Returns ``None`` if the seam's mask is ``None``.
    """
    if action_mask is None:
        return None
    if not isinstance(action_mask, dict):
        # Defensive: a future seam revision might emit a flat array. Coerce.
        return [float(x) for x in np.asarray(action_mask).ravel()]
    flat: list[float] = []
    for key in sorted(action_mask.keys()):
        arr = np.asarray(action_mask[key]).ravel()
        flat.extend(float(x) for x in arr)
    return flat


def _coerce_components(info: dict[str, Any]) -> dict[str, float]:
    """Pull the reward ``components`` dict out of the seam's info and coerce to floats."""
    raw = info.get("components")
    if not isinstance(raw, dict):
        return {}
    return {str(k): float(v) for k, v in raw.items()}


def _jsonsafe_info(info: dict[str, Any]) -> dict[str, Any]:
    """Strip the seam's info dict of non-JSON-safe values for the wire.

    Drops ``action_mask`` and ``components`` (they have dedicated wire fields),
    coerces numpy scalars to Python types, and silently drops anything else — the
    wire is for display, not authority.
    """
    safe: dict[str, Any] = {}
    for key, value in info.items():
        if key in ("action_mask", "components"):
            continue
        if isinstance(value, (str, bool, int)) and not isinstance(value, np.bool_):
            safe[key] = value
        elif isinstance(value, (float, np.floating)):
            safe[key] = float(value)
        elif isinstance(value, np.integer):
            safe[key] = int(value)
        elif isinstance(value, np.bool_):
            safe[key] = bool(value)
        elif value is None:
            safe[key] = None
        # Else: silently drop unsupported types — the wire is operator-display; structured data
        # goes through the protocol's dedicated fields.
    return safe
