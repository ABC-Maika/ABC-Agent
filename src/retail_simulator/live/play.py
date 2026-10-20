"""The reusable play-one-game coroutine + seat-permutation helpers.

Owns the single client play loop: connect a :class:`LiveClient`, submit the
passive default-policy action every observed tick, capture the end-of-game
:class:`FinalScores`, and close. Shared by the ``live play`` CLI command and
the self-play ``serve_fn``. Presentation (printing, exit codes) stays in
the CLI; this module only drives the client and returns the scores.

Also holds the seat permutation applied to the parallel action buffer before
it reaches the world: the parallel layout lists seats 0..n-1 while the
world's seat order is shuffled, so each row must be routed to the matching
region. ``permute_actions_by_seat`` is a pure numpy function with no
server dependencies.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from retail_simulator.core.encoding import flatten_action_space_layout
from retail_simulator.live.client import LiveClient
from retail_simulator.live.protocol import REGISTRY_DEFAULT_ACTION, FinalScores, SeatAssigned


async def play_one_game(
    uri: str,
    agent_name: str,
    token: str | None,
    *,
    policy: str = "default",
    on_seat_assigned: Callable[[SeatAssigned], None] | None = None,
) -> FinalScores | None:
    """Connect one :class:`LiveClient`, play a whole game, return its scores.

    Returns ``None`` if the socket closed without an end-of-game frame (an
    aborted game). The client is always closed in a ``finally``.

    The ``default`` policy is obs-independent: a fresh writable float32 copy of
    :data:`REGISTRY_DEFAULT_ACTION` each tick. Other ``policy`` values raise
    :class:`ValueError` (a learned policy is deferred). The optional
    ``on_seat_assigned`` callback fires once, right after seat assignment.
    """
    if policy != "default":
        raise ValueError(f"unknown policy {policy!r}; only 'default' is supported in this slice")

    client = LiveClient(uri=uri, agent_name=agent_name, token=token)
    try:
        seat = await client.connect()
        if on_seat_assigned is not None:
            on_seat_assigned(seat)
        async for _obs in client.iter_observations():
            await client.submit_action(_default_action())
        return client.final_scores
    finally:
        await client.close()


def _default_action() -> np.ndarray[Any, Any]:
    """A fresh writable float32 copy of the registry-default action (decoupled from the shared read-only singleton)."""
    return np.array(REGISTRY_DEFAULT_ACTION, dtype=np.float32, copy=True)


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
