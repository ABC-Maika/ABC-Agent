"""Determinism: a single injected PCG64 generator and serializable RNG state.

The sim uses one ``numpy.default_rng(seed)`` (PCG64), threaded through every
stochastic step — same seed + same actions reproduce the trajectory bit-for-bit.

Per-tick draw order (``world.py`` must honor it exactly):
    1. Demand resolution noise.
    2. NPC actions, in agent-index order.
    3. Accounting noise.
    4. Perception noise — one constant-size vectorized draw, last, so economic
       draws #1-#3 consume the same stream at any research level.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def make_rng(seed: int | None) -> np.random.Generator:
    """Construct the canonical PCG64 generator.

    ``seed=None`` is non-reproducible; pass an integer seed for determinism.
    """
    return np.random.default_rng(seed)


def snapshot(rng: np.random.Generator) -> dict[str, Any]:
    """Capture the generator state as plain serializable data (deep copy)."""
    import copy

    return copy.deepcopy(dict(rng.bit_generator.state))


def restore(state: dict[str, Any]) -> np.random.Generator:
    """Rebuild a generator positioned exactly where ``snapshot`` was taken."""
    import copy

    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(state)
    return rng
