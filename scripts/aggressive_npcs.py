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

from hunter_bots import load_hunters, persona_predict

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
    "orcho-recipe2": "SCRIPTED v2 recipe: opens the biggest-demand regions first (METRO, AFFLUENT, "
    "LOYAL, FRONTIER), up to 12 stores; staff pay 0.45, raised up to 1.0 when staff get unhappy; "
    "research 0.2.",
    "orcho-recipe3": "SCRIPTED v3 recipe: recipe2, but it opens the METRO regions as a ring starting next "
    "to the ones it already holds (not 'lowest number first'), so copies of it stop piling into the same "
    "regions at the start.",
    "orcho-recipe4": "SCRIPTED v4 recipe: recipe3's levers, but it opens METRO then AFFLUENT regions "
    "HIGHEST number first, dodging the crowd of agents that open lowest number first.",
}
HUNTERS = load_hunters()  # sparring bots searched to beat our agent (scripts/hunter_search.py)
AGGRESSIVE_DESCRIPTIONS.update(
    {
        name: f"SCRIPTED sparring bot searched to beat orcho-a2c: {p['order']} expansion, price "
        f"{p['price']}, marketing {p['marketing']}, promotion {p['promotion']}, up to {p['cap']} stores."
        for name, p in HUNTERS.items()
    }
)
AGGRESSIVE_LABELS: tuple[str, ...] = tuple(AGGRESSIVE_DESCRIPTIONS)


def _recipe_predict(
    target_fn: Callable[[Any, int], int],
    n_regions: int,
    n_obs: int,
    wage_spend: float | Callable[[Any], float] = 0.5,
    research: float = 0.0,
) -> Any:
    """Fixed levers (the wage lever is not exposed by the organizer's wrapper) + a store-capped rule.

    ``wage_spend`` may be a function of the observation (pay more when staff get unhappy).
    """
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
                research=research,
                service_level=1.0,
                automation_tier=0,
                loyalty_spend=0.0,
                n_regions=n_regions,
                wage_spend=wage_spend(obs) if callable(wage_spend) else wage_spend,
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

    def biggest_first(obs: Any, _tick: int) -> int:
        """Up to 12 stores, the highest-demand class first (class cycle from region 1)."""
        if float(obs[idx["stores"]]) >= 12:
            return 0
        empty = list(geo._own_empty_regions(obs, idx=idx, candidates=range(1, n_regions)))
        return min(empty, key=lambda r: ((r - 1) % 4, r)) if empty else 0

    ring = sorted(r for r in range(1, n_regions) if (r - 1) % 4 == 0)  # METRO regions, in order

    def ring_then_affluent(obs: Any, _tick: int) -> int:
        """METRO ring: continue from the end of the chain we hold, so every copy walks to a different
        neighbour; then AFFLUENT by index. Uses only the store counts (no memory of the home)."""
        # No cash check on purpose: the cloned network cannot see cash, and the game already masks
        # purchases we cannot afford (a cash check made identical-looking inputs get opposite labels).
        if float(obs[idx["stores"]]) >= 12:
            return 0
        n = len(ring)
        owned = {i for i, r in enumerate(ring) if float(obs[idx[f"stores_region_{r}"]]) > 0}
        if owned and len(owned) < n:
            tails = [q for q in owned if (q + 1) % n not in owned and (q - 1) % n in owned]
            tails = tails or [q for q in owned if (q + 1) % n not in owned]
            return ring[(tails[0] + 1) % n]
        empty = list(geo._own_empty_regions(obs, idx=idx, candidates=range(1, n_regions)))
        empty = [r for r in empty if (r - 1) % 4 in (0, 1)]  # METRO, AFFLUENT only
        return min(empty, key=lambda r: ((r - 1) % 4, r)) if empty else 0

    def highest_first(obs: Any, _tick: int) -> int:
        """METRO then AFFLUENT, HIGHEST region number first. Most agents (the twin, Krischo's LSTM, ...)
        open lowest-number first, so this dodges their crowd in the opening cash race. Stateless."""
        if float(obs[idx["stores"]]) >= 12:
            return 0
        empty = list(geo._own_empty_regions(obs, idx=idx, candidates=range(1, n_regions)))
        empty = [r for r in empty if (r - 1) % 4 in (0, 1)]  # METRO, AFFLUENT only
        return min(empty, key=lambda r: ((r - 1) % 4, -r)) if empty else 0

    def happiness_pay(obs: Any) -> float:
        """0.45 while staff are happy, up to 1.0 as happiness drops (bigger share = higher wage bar)."""
        return min(1.0, 0.45 + 4.0 * (1.0 - float(obs[idx["employee_happiness"]])))

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
        "orcho-recipe4": _recipe_predict(
            highest_first, n_regions, len(idx), wage_spend=happiness_pay, research=0.2
        ),
        "orcho-recipe3": _recipe_predict(
            ring_then_affluent, n_regions, len(idx), wage_spend=happiness_pay, research=0.2
        ),
        "orcho-recipe2": _recipe_predict(
            biggest_first, n_regions, len(idx), wage_spend=happiness_pay, research=0.2
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
        **{name: persona_predict(p, n_regions) for name, p in HUNTERS.items()},
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
