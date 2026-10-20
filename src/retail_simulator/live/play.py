"""The reusable play-one-game coroutine.

Owns the single client play loop: connect a :class:`LiveClient`, submit the
passive default-policy action every observed tick, capture the end-of-game
:class:`FinalScores`, and close. Shared by the ``live play`` CLI command and
the self-play ``serve_fn``. Presentation (printing, exit codes) stays in
``cli.py``; this coroutine only drives the client and returns the scores.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

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
