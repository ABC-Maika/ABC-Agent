"""More aggressive scripted rivals for our own practice matches (team addition, not organizer code).

The organizer's four archetypes live in ``src/`` and are what the final scoring knows about, so
they stay untouched. This module adds three tougher variants by reusing the organizer's building
blocks (expansion rules and the lever-vector wrapper) with sharper fixed levers:

- ``cherry-picker+``  METRO first, then every other region; cheaper, heavier marketing and loyalty
- ``fast-follower+``  walks into regions rivals already hold; cheap, heavy marketing and promotion
- ``sprawler+``       the lowest-index empty region every tick; cheap, with real marketing behind it

Call :func:`register` once before building an ``ArenaMatch``; it makes the new names valid
``ArenaSeatSpec.archetype`` values for this process only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from retail_simulator.envs import arena_match
from retail_simulator.envs import geo_archetypes as geo

AGGRESSIVE_DESCRIPTIONS = {
    "cherry-picker+": "Takes the big METRO regions first, then every other region. Cheaper than "
    "cherry-picker, with heavy marketing, assortment and loyalty.",
    "fast-follower+": "Walks into regions rivals already hold. Cheap, with heavy marketing and "
    "promotion.",
    "sprawler+": "Opens the lowest-index empty region every round. Cheap, with real marketing "
    "behind it.",
}
AGGRESSIVE_LABELS: tuple[str, ...] = tuple(AGGRESSIVE_DESCRIPTIONS)


def _aggressive_roster(n_regions: int, region_table: Any) -> dict[str, Callable[..., Any]]:
    idx = geo._name_to_index(n_regions)
    metro = geo._metro_indices(region_table)

    def metro_then_everywhere(obs: Any, _tick: int) -> int:
        return geo._cherry_picker_target(obs, idx=idx, metro_indices=metro) or geo._sprawler_target(
            obs, idx=idx, n_regions=n_regions
        )

    def follow_rivals(obs: Any, _tick: int) -> int:
        return geo._fast_follower_target(
            obs, idx=idx, n_regions=n_regions, region_table=region_table
        )

    def sprawl(obs: Any, _tick: int) -> int:
        return geo._sprawler_target(obs, idx=idx, n_regions=n_regions)

    shared = {"n_regions": n_regions, "n_obs": len(idx), "research": 0.0, "automation_tier": 0}
    return {
        "cherry-picker+": geo._make_predict(
            metro_then_everywhere,
            price=1.0,
            marketing=0.9,
            assortment=0.9,
            promotion=0.6,
            service_level=1.0,
            loyalty_spend=0.8,
            **shared,
        ),
        "fast-follower+": geo._make_predict(
            follow_rivals,
            price=0.85,
            marketing=0.9,
            assortment=0.9,
            promotion=0.8,
            service_level=0.7,
            loyalty_spend=0.6,
            **shared,
        ),
        "sprawler+": geo._make_predict(
            sprawl,
            price=0.7,
            marketing=0.8,
            assortment=0.7,
            promotion=0.8,
            service_level=0.5,
            loyalty_spend=0.4,
            **shared,
        ),
    }


def register() -> tuple[str, ...]:
    """Make the aggressive names seatable; returns every valid ``--npc`` name."""
    if AGGRESSIVE_LABELS[0] not in arena_match.GEO_ARCHETYPE_LABELS:
        base = arena_match.geo_archetypes

        def with_aggressive(n_regions: int, region_table: Any) -> dict[str, Callable[..., Any]]:
            return {**base(n_regions, region_table), **_aggressive_roster(n_regions, region_table)}

        arena_match.geo_archetypes = with_aggressive
        arena_match.GEO_ARCHETYPE_LABELS = (*arena_match.GEO_ARCHETYPE_LABELS, *AGGRESSIVE_LABELS)
    return tuple(arena_match.GEO_ARCHETYPE_LABELS)
