"""Parametric sparring bots ("hunters") built to beat our own agent (team addition, not organizer code).

A hunter is a plain dict of lever settings plus an expansion ``order``; ``persona_predict`` turns it into a
predict function the arena can seat as an NPC. ``scripts/hunter_search.py`` searches for the dicts that hurt
orcho-a2c the most and stores the best in ``scripts/hunters.json``; ``aggressive_npcs.register`` then seats them
as ``hunter-1``, ``hunter-2``, ... (see ``guides/REWARD_DESIGN_HINTS.md``: sparring partners with distinct styles).

Expansion orders: ``index`` (lowest number first), ``demand`` (biggest class first), ``ring`` (METRO ring from the
stores held, then AFFLUENT), ``steal`` (the empty region where most rivals already sit), ``quiet`` (the one with
fewest rivals), ``affluent`` (AFFLUENT before METRO).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from retail_simulator.envs import geo_archetypes as geo

HUNTERS_FILE = Path(__file__).with_name("hunters.json")
ORDERS = ("index", "demand", "ring", "steal", "quiet", "affluent", "mirror", "blocker", "calm", "calm2", "hyb1", "hyb3")
DEFAULT = {
    "price": 2.0, "marketing": 1.0, "assortment": 1.0, "promotion": 1.0, "service": 1.0, "research": 0.2,
    "wage": 0.45, "smart_pay": 1, "cap": 12, "order": "demand", "early_price": 0.0, "switch_tick": 0,
}
# Hand-made personas from the organizer's reward-design hints, as starting points for the search.
PERSONAS = {
    "expander": {**DEFAULT, "cap": 16, "order": "index", "price": 1.6},
    "defender": {**DEFAULT, "cap": 8, "order": "ring", "marketing": 1.0, "price": 1.9},
    "price-fighter": {**DEFAULT, "price": 1.0, "promotion": 1.0, "order": "demand"},
    "cautious": {**DEFAULT, "cap": 6, "order": "quiet", "marketing": 0.5, "promotion": 0.3},
    "spoiler": {**DEFAULT, "order": "steal", "price": 1.5},
    "mirror": {**DEFAULT, "order": "mirror"},
    "blocker": {**DEFAULT, "order": "blocker", "price": 1.7},
    "starver": {**DEFAULT, "order": "mirror", "early_price": 0.9, "switch_tick": 100},
}


def load_hunters() -> dict[str, dict[str, Any]]:
    """The searched hunters, ``{"hunter-1": params, ...}`` (empty until a search has been run)."""
    if HUNTERS_FILE.exists():
        return json.loads(HUNTERS_FILE.read_text())
    return {}


def _ring_target(obs: Any, idx: dict[str, int], n_regions: int, cap: int) -> int:
    ring = [r for r in range(1, n_regions) if (r - 1) % 4 == 0]
    if float(obs[idx["stores"]]) >= cap:
        return 0
    n = len(ring)
    owned = {i for i, r in enumerate(ring) if float(obs[idx[f"stores_region_{r}"]]) > 0}
    if owned and len(owned) < n:
        tails = [q for q in owned if (q + 1) % n not in owned and (q - 1) % n in owned]
        tails = tails or [q for q in owned if (q + 1) % n not in owned]
        return ring[(tails[0] + 1) % n]
    empty = [r for r in geo._own_empty_regions(obs, idx=idx, candidates=range(1, n_regions)) if (r - 1) % 4 in (0, 1)]
    return min(empty, key=lambda r: ((r - 1) % 4, r)) if empty else 0


def persona_predict(params: dict[str, Any], n_regions: int = 25) -> Callable[..., Any]:
    """Build a predict function (obs -> action) for one hunter."""
    from retail_simulator.core.encoding import encode_action

    p = {**DEFAULT, **params}
    idx = geo._name_to_index(n_regions)

    def target(obs: Any) -> int:
        if p["order"] == "ring":
            return _ring_target(obs, idx, n_regions, int(p["cap"]))
        if float(obs[idx["stores"]]) >= p["cap"]:
            return 0
        empty = list(geo._own_empty_regions(obs, idx=idx, candidates=range(1, n_regions)))
        if not empty:
            return 0
        rivals = lambda r: float(obs[idx[f"others_present_region_{r}"]])  # noqa: E731
        cls = lambda r: (r - 1) % 4  # noqa: E731
        if p["order"] == "index":
            return min(empty)
        if p["order"] == "steal":
            return min(empty, key=lambda r: (-rivals(r), cls(r), r))
        if p["order"] == "quiet":
            return min(empty, key=lambda r: (rivals(r), cls(r), r))
        if p["order"] == "mirror":  # our own v5 opening: METRO then AFFLUENT, highest number first
            pool = [r for r in empty if cls(r) in (0, 1)] or empty
            return min(pool, key=lambda r: (cls(r), -r))
        if p["order"] in ("hyb1", "hyb3"):  # v5 order unless rivals already sit there
            pool = [r for r in empty if cls(r) in (0, 1)] or empty
            v5 = sorted(pool, key=lambda r: (cls(r), -r))
            thr = p.get("thr", 1)  # tolerate fewer than ``thr`` rival stores in a region
            free = [r for r in v5 if rivals(r) < thr] if p["order"] == "hyb3" else ([v5[0]] if rivals(v5[0]) < thr else [])
            if free:
                return free[0]
            return min(pool, key=lambda r: (rivals(r), cls(r), -r))
        if p["order"] in ("calm", "calm2"):  # METRO/AFFLUENT only; fewest rivals present, ties highest number
            pool = [r for r in empty if cls(r) in (0, 1)] or empty
            if p["order"] == "calm":
                return min(pool, key=lambda r: (cls(r), rivals(r), -r))
            return min(pool, key=lambda r: (rivals(r), cls(r), -r))
        if p["order"] == "blocker":  # METRO, lowest number first but only where we are likely to go: highest first, AFFLUENT before METRO
            pool = [r for r in empty if cls(r) in (0, 1)] or empty
            return min(pool, key=lambda r: (-r,))
        if p["order"] == "affluent":
            return min(empty, key=lambda r: ({1: 0, 0: 1, 2: 2, 3: 3}[cls(r)], r))
        return min(empty, key=lambda r: (cls(r), r))  # demand

    tick = [0]

    def predict(obs: Any, *, episode_start: bool = False, **_kw: Any) -> Any:
        if episode_start:
            tick[0] = 0
        t = tick[0]
        tick[0] += 1
        price = p["early_price"] if (p["early_price"] and t < p["switch_tick"]) else p["price"]
        wage = p["wage"]
        if p["smart_pay"]:
            wage = min(1.0, wage + 4.0 * (1.0 - float(obs[idx["employee_happiness"]])))
        return encode_action(
            geo._structured_action(
                price=price, marketing=p["marketing"], assortment=p["assortment"],
                promotion=p["promotion"], expansion_choice=target(obs), research=p["research"],
                service_level=p["service"], automation_tier=0, loyalty_spend=0.0,
                n_regions=n_regions, wage_spend=wage, warehouse_invest=0.0,
            )
        )

    return predict


# Candidate fixes for our own agent (v5 opening + a different price), seated as NPCs ``contender-<name>``.
CONTENDERS = {
    "v5": {**DEFAULT, "order": "mirror"},
    "p185": {**DEFAULT, "order": "mirror", "price": 1.85},
    "p17": {**DEFAULT, "order": "mirror", "price": 1.7},
    "calm": {**DEFAULT, "order": "calm"},
    "h1t2": {**DEFAULT, "order": "hyb1", "thr": 2},
    "h1t3": {**DEFAULT, "order": "hyb1", "thr": 3},
    "h3t2": {**DEFAULT, "order": "hyb3", "thr": 2},
    "h3t3": {**DEFAULT, "order": "hyb3", "thr": 3},
    "h3t4": {**DEFAULT, "order": "hyb3", "thr": 4},
    "calm2": {**DEFAULT, "order": "calm2"},
    "p15": {**DEFAULT, "order": "mirror", "price": 1.5},
    "p17-early": {**DEFAULT, "order": "mirror", "early_price": 1.5, "switch_tick": 150, "price": 1.9},
}
