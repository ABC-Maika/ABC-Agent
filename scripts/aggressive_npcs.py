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
    "orcho-recipe": "SCRIPTED best-known recipe from our lever tests (not a trained checkpoint): "
    "max price and marketing, wages paid so staff stay happy, stops expanding at 10 stores.",
}
AGGRESSIVE_LABELS: tuple[str, ...] = tuple(AGGRESSIVE_DESCRIPTIONS)


def _recipe_predict(target_fn: Callable[[Any, int], int], n_regions: int, n_obs: int) -> Any:
    """Fixed levers (the wage lever is not exposed by the organizer's wrapper) + a store-capped rule."""
    from retail_simulator.core.encoding import encode_action

    tick = [0]

    def predict(obs: Any, *, episode_start: bool = False, **_kw: Any) -> Any:
        if episode_start:
            tick[0] = 0
        t = tick[0]
        tick[0] += 1
        return encode_action(
            geo._structured_action(
                price=2.0,
                marketing=1.0,
                assortment=1.0,
                promotion=1.0,
                expansion_choice=target_fn(obs, t),
                research=0.0,
                service_level=1.0,
                automation_tier=0,
                loyalty_spend=0.0,
                n_regions=n_regions,
                wage_spend=0.5,
                warehouse_invest=0.0,
            )
        )

    return predict


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

    def capped_sprawl(obs: Any, _tick: int) -> int:
        if float(obs[idx["stores"]]) >= 10:
            return 0
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
        "orcho-recipe": _recipe_predict(capped_sprawl, n_regions, len(idx)),
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
