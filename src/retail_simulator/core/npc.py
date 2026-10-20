"""NPC policies: the ``NPCPolicy`` protocol and the scripted archetypes.

NPCs fill the non-learning seats. Every policy takes the injected RNG so a
stochastic NPC stays replayable; the shipped archetypes (Discounter, Premium,
Balanced, Wandering) are deterministic and never draw. Each ``act`` returns a
flat action vector in the agent's action layout, so a frozen learned policy
drops into the same seat with no adapter.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
import numpy.typing as npt

from retail_simulator.core.encoding import StructuredAction, encode_action
from retail_simulator.core.schema import ActionSchema, default_action_schema
from retail_simulator.core.state import WorldState

# Module defaults; a per-instance value can override.
DISCOUNTER_PRICE_INDEX: float = 0.7
DISCOUNTER_PROMOTION: float = 0.0

PREMIUM_PRICE_INDEX: float = 1.4
PREMIUM_MARKETING: float = 0.7
PREMIUM_ASSORTMENT: float = 0.9
PREMIUM_PROMOTION: float = 0.1

BALANCED_REACT_FACTOR: float = 0.95
BALANCED_MARKETING: float = 0.6
BALANCED_ASSORTMENT: float = 0.7
BALANCED_PROMOTION: float = 0.7
# Fallback price Balanced matches when it is the only present retailer.
BALANCED_BASELINE_PRICE_INDEX: float = 1.0

WANDERING_BASELINE_PRICE_INDEX: float = 1.0
WANDERING_AMPLITUDE: float = 0.3
WANDERING_PERIOD: int = 26
WANDERING_MARKETING: float = 0.4
WANDERING_ASSORTMENT: float = 0.5
WANDERING_PROMOTION: float = 0.3


class NPCPolicy(Protocol):
    """Interface every NPC (and frozen learned policy) seat satisfies.

    ``act`` runs once per tick in seat order and returns a flat action vector.
    Stochastic policies must draw only from the passed ``rng`` so behavior is
    replayable; ``seat_index`` is the policy's own index into ``state.retailers``.
    """

    def act(
        self, state: WorldState, rng: np.random.Generator, seat_index: int
    ) -> npt.NDArray[np.float32]: ...


class Discounter:
    """Fixed aggressive discount, no marketing, narrow range, never expands."""

    def __init__(
        self,
        price_index: float = DISCOUNTER_PRICE_INDEX,
        schema: ActionSchema | None = None,
    ) -> None:
        self._price_index = price_index
        self._schema = schema

    def act(
        self, state: WorldState, rng: np.random.Generator, seat_index: int = 0
    ) -> npt.NDArray[np.float32]:
        """Return the fixed-discount, no-marketing, no-promo, no-expand action.

        Deterministic (no ``rng`` draw); ``seat_index`` is accepted-and-ignored.
        """
        structured = StructuredAction(
            levers={
                "price_index": self._price_index,
                "marketing": 0.0,
                "assortment": 0.0,
                "promotion": DISCOUNTER_PROMOTION,
                "expansion": 0.0,
                "research": 0.0,
                "service_level": 1.0,
                # NPCs never automate: target tier 0 every tick (no capex, no savings).
                "automation": 0.0,
                # NPCs never run a loyalty program.
                "loyalty_spend": 0.0,
                # NPCs never pay premium wages or invest in warehouse capacity.
                "wage_spend": 0.0,
                "warehouse_invest": 0.0,
            }
        )
        return encode_action(structured, self._schema)


class Premium:
    """Fixed premium posture: high price, broad range, high marketing."""

    def __init__(
        self,
        price_index: float = PREMIUM_PRICE_INDEX,
        marketing: float = PREMIUM_MARKETING,
        assortment: float = PREMIUM_ASSORTMENT,
        promotion: float = PREMIUM_PROMOTION,
        schema: ActionSchema | None = None,
    ) -> None:
        self._price_index = price_index
        self._marketing = marketing
        self._assortment = assortment
        self._promotion = promotion
        self._schema = schema

    def act(
        self, state: WorldState, rng: np.random.Generator, seat_index: int = 0
    ) -> npt.NDArray[np.float32]:
        """Return the fixed premium action.

        Deterministic; ``seat_index`` is accepted-and-ignored; expansion is a no-op
        (choice 0) — presence comes from the reset plan.
        """
        structured = StructuredAction(
            levers={
                "price_index": self._price_index,
                "marketing": self._marketing,
                "assortment": self._assortment,
                "promotion": self._promotion,
                "expansion": 0.0,
                "research": 0.0,
                "service_level": 1.0,
                "automation": 0.0,
                # NPCs never run a loyalty program.
                "loyalty_spend": 0.0,
                # NPCs never pay premium wages or invest in warehouse capacity.
                "wage_spend": 0.0,
                "warehouse_invest": 0.0,
            }
        )
        return encode_action(structured, self._schema)


class Balanced:
    """Reactive price-matcher: undercuts the mean price of other present retailers."""

    def __init__(
        self,
        react_factor: float = BALANCED_REACT_FACTOR,
        marketing: float = BALANCED_MARKETING,
        assortment: float = BALANCED_ASSORTMENT,
        promotion: float = BALANCED_PROMOTION,
        min_price_index: float | None = None,
        max_price_index: float | None = None,
        schema: ActionSchema | None = None,
    ) -> None:
        self._react_factor = react_factor
        self._marketing = marketing
        self._assortment = assortment
        self._promotion = promotion
        self._min_price_index = min_price_index
        self._max_price_index = max_price_index
        self._schema = schema

    def _price_bounds(self) -> tuple[float, float]:
        """Resolve (min, max) price bounds: the configured values, else the schema's."""
        if self._min_price_index is not None and self._max_price_index is not None:
            return self._min_price_index, self._max_price_index
        schema = self._schema if self._schema is not None else default_action_schema()
        price_lever = next(lever for lever in schema.levers if lever.name == "price_index")
        assert price_lever.low is not None and price_lever.high is not None
        return price_lever.low, price_lever.high

    def act(
        self, state: WorldState, rng: np.random.Generator, seat_index: int
    ) -> npt.NDArray[np.float32]:
        """Undercut the mean price of the other present retailers, clamped to bounds.

        Falls back to the baseline price when no other retailer is present.
        """
        other_prices = [
            r.price_index
            for idx, r in enumerate(state.retailers)
            if idx != seat_index and r.stores > 0
        ]
        if other_prices:
            observed = sum(other_prices) / len(other_prices)
        else:
            observed = BALANCED_BASELINE_PRICE_INDEX
        min_price, max_price = self._price_bounds()
        price = min(max(observed * self._react_factor, min_price), max_price)
        structured = StructuredAction(
            levers={
                "price_index": price,
                "marketing": self._marketing,
                "assortment": self._assortment,
                "promotion": self._promotion,
                "expansion": 0.0,
                "research": 0.0,
                "service_level": 1.0,
                "automation": 0.0,
                # NPCs never run a loyalty program.
                "loyalty_spend": 0.0,
                # NPCs never pay premium wages or invest in warehouse capacity.
                "wage_spend": 0.0,
                "warehouse_invest": 0.0,
            }
        )
        return encode_action(structured, self._schema)


class Wandering:
    """Price follows a sine wave around a baseline (exogenous, tick-driven)."""

    def __init__(
        self,
        baseline: float = WANDERING_BASELINE_PRICE_INDEX,
        amplitude: float = WANDERING_AMPLITUDE,
        period: int = WANDERING_PERIOD,
        marketing: float = WANDERING_MARKETING,
        assortment: float = WANDERING_ASSORTMENT,
        promotion: float = WANDERING_PROMOTION,
        min_price_index: float | None = None,
        max_price_index: float | None = None,
        schema: ActionSchema | None = None,
    ) -> None:
        if period <= 0:
            raise ValueError(f"Wandering period must be > 0, got {period}")
        self._baseline = baseline
        self._amplitude = amplitude
        self._period = period
        self._marketing = marketing
        self._assortment = assortment
        self._promotion = promotion
        self._min_price_index = min_price_index
        self._max_price_index = max_price_index
        self._schema = schema

    def _price_bounds(self) -> tuple[float, float]:
        """Resolve (min, max) price bounds: the configured values, else the schema's."""
        if self._min_price_index is not None and self._max_price_index is not None:
            return self._min_price_index, self._max_price_index
        schema = self._schema if self._schema is not None else default_action_schema()
        price_lever = next(lever for lever in schema.levers if lever.name == "price_index")
        assert price_lever.low is not None and price_lever.high is not None
        return price_lever.low, price_lever.high

    def act(
        self, state: WorldState, rng: np.random.Generator, seat_index: int = 0
    ) -> npt.NDArray[np.float32]:
        """Price = baseline + amplitude * sin(2π * tick / period), clamped to bounds."""
        del seat_index  # the schedule is seat-independent (exogenous, tick-driven)
        raw = self._baseline + self._amplitude * math.sin(2.0 * math.pi * state.tick / self._period)
        min_price, max_price = self._price_bounds()
        price = min(max(raw, min_price), max_price)
        structured = StructuredAction(
            levers={
                "price_index": price,
                "marketing": self._marketing,
                "assortment": self._assortment,
                "promotion": self._promotion,
                "expansion": 0.0,
                "research": 0.0,
                "service_level": 1.0,
                "automation": 0.0,
                # NPCs never run a loyalty program.
                "loyalty_spend": 0.0,
                # NPCs never pay premium wages or invest in warehouse capacity.
                "wage_spend": 0.0,
                "warehouse_invest": 0.0,
            }
        )
        return encode_action(structured, self._schema)
