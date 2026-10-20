"""Action masks, seat planning, NPC policy construction, and RNG utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import CoreConfig, SeatSpec
from retail_simulator.core.npc import (
    Balanced,
    Discounter,
    NPCPolicy,
    Premium,
    Wandering,
)
from retail_simulator.core.schema import (
    N_AUTOMATION_TIERS,
    N_REGIONS_MAX,
    ActionSchema,
)
from retail_simulator.core.state import WorldState

# Seat 0 is the learning agent; seat 1 is the discounter NPC.
AGENT_INDEX: int = 0


def default_seat_plan() -> tuple[SeatSpec, ...]:
    """Default two-seat plan: learning agent[0] + discounter NPC[1]."""
    return (
        SeatSpec(is_npc=False, name="agent"),
        SeatSpec(is_npc=True, archetype="discounter", name="discounter"),
    )


def _make_npc_policy(archetype: str, schema: ActionSchema) -> NPCPolicy:
    """Build the NPC policy for an archetype name."""
    if archetype == "discounter":
        return Discounter(schema=schema)
    if archetype == "premium":
        return Premium(schema=schema)
    if archetype == "balanced":
        return Balanced(schema=schema)
    if archetype == "wandering":
        return Wandering(schema=schema)
    raise ValueError(f"unknown NPC archetype {archetype!r}")


def _default_seat_name(spec: SeatSpec, seat_index: int) -> str:
    """Display name for a seat without one: ``retailer_{i}`` or ``{archetype}_{i}``."""
    if spec.is_npc:
        return f"{spec.archetype}_{seat_index}"
    return f"retailer_{seat_index}"


STARTING_CASH: float = 15_000.0
INITIAL_STOCKPILE: float = 0.0
INITIAL_SERVICE_SCORE: float = 1.0


def compute_expansion_mask(
    state: WorldState, agent_index: int, config: CoreConfig
) -> npt.NDArray[np.bool_]:
    """Regions where the agent can open a store (affordable, not already present)."""
    me = state.retailers[agent_index]
    capex = config.expansion.expansion_capex
    n_regions = len(state.regions)
    mask = np.zeros(N_REGIONS_MAX, dtype=bool)
    mask[0] = True
    for r in range(1, n_regions):
        already_present = r < len(me.stores_per_region) and me.stores_per_region[r] > 0
        mask[r] = (me.cash >= capex) and (not already_present)
    return mask


def compute_automation_mask(
    state: WorldState, agent_index: int, config: CoreConfig
) -> npt.NDArray[np.bool_]:
    """Automation tiers the agent can afford to step up to."""
    me = state.retailers[agent_index]
    capex_per_tier = config.automation.capex_per_tier
    prev = me.automation_tier
    mask = np.zeros(N_AUTOMATION_TIERS, dtype=bool)
    for t in range(N_AUTOMATION_TIERS):
        monotone = t >= prev
        differential = capex_per_tier[t] - capex_per_tier[prev]
        affordable = me.cash >= differential
        mask[t] = monotone and affordable
    return mask


def _restructuring_survivor_regions(
    stores_per_region: tuple[int, ...],
    last_market_share_per_region: tuple[float, ...],
    n_regions: int,
    keep_stores: int,
) -> set[int]:
    """Present regions to keep: the highest-market-share ones, down to ``keep_stores``."""
    present = [r for r in range(1, n_regions) if stores_per_region[r] > 0]
    ranked = sorted(present, key=lambda r: (-last_market_share_per_region[r], r))
    return set(ranked[:keep_stores])


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
