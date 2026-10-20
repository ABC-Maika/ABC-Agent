"""Seat-indexed joint-action assembly for the live server.

Package-internal (leading underscore); imported directly by ``server.py``
rather than through ``live/__init__.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from retail_simulator.core.encoding import flatten_action_space_layout

# Derived from the current action schema's flat layout, not hardcoded.
# Actions are validated at the boundary so a malformed entry surfaces here,
# not deep inside ``env.step``.
_EXPECTED_ACTION_SHAPE: tuple[int, ...] = (flatten_action_space_layout().total_dim,)
_EXPECTED_ACTION_DTYPE: np.dtype[np.float32] = np.dtype(np.float32)


def _validate_action(seat: int, action: npt.NDArray[np.float32]) -> None:
    """Validate an action's shape + dtype; raise ``ValueError`` with the seat index."""
    if action.shape != _EXPECTED_ACTION_SHAPE:
        raise ValueError(
            f"seat {seat}: action has shape {action.shape}, expected {_EXPECTED_ACTION_SHAPE}"
        )
    if action.dtype != _EXPECTED_ACTION_DTYPE:
        raise ValueError(
            f"seat {seat}: action has dtype {action.dtype}, expected {_EXPECTED_ACTION_DTYPE}"
        )


def permute_actions_by_seat(
    arrived: dict[int, npt.NDArray[np.float32]],
    last_valid: dict[int, npt.NDArray[np.float32]],
    n_seats: int,
    registry_default_action: npt.NDArray[np.float32],
) -> tuple[dict[str, npt.NDArray[np.float32]], list[int]]:
    """Assemble the joint action keyed by agent id for every seat.

    Per-seat fallback: ``arrived`` → ``last_valid`` → a copy of
    ``registry_default_action``. Returns the joint action plus the seats
    that used a fallback.
    """
    if n_seats < 1:
        raise ValueError(f"n_seats must be >= 1, got {n_seats}")

    # Validate every action landing in the joint action against the current
    # schema. The registry default is shape/dtype-correct by construction
    # (built via ``encode_action``), so it is not revalidated.
    for seat, action in arrived.items():
        _validate_action(seat, action)
    for seat, action in last_valid.items():
        _validate_action(seat, action)

    joint_action: dict[str, npt.NDArray[np.float32]] = {}
    missing_seats: list[int] = []

    for seat in range(n_seats):
        agent_id = f"retailer_{seat}"
        if seat in arrived:
            joint_action[agent_id] = arrived[seat]
        elif seat in last_valid:
            joint_action[agent_id] = last_valid[seat]
            missing_seats.append(seat)
        else:
            # Copy the shared registry default so a downstream writer cannot corrupt it.
            joint_action[agent_id] = registry_default_action.copy()
            missing_seats.append(seat)

    return joint_action, missing_seats


__all__ = ["permute_actions_by_seat"]
