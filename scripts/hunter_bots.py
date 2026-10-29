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
ORDERS = ("index", "demand", "ring", "steal", "quiet", "affluent", "mirror", "blocker", "calm", "calm2", "hyb1", "hyb3", "score")
DEFAULT = {
    "price": 2.0, "marketing": 1.0, "assortment": 1.0, "promotion": 1.0, "service": 1.0, "research": 0.2,
    "wage": 0.45, "smart_pay": 1, "cap": 12, "order": "demand", "early_price": 0.0, "switch_tick": 0, "frugal": 0, "brake_cash": 0.0, "brake_profit": 0.0,
    # v6 situational knobs (all off by default, so older hunters/contenders play exactly as before):
    "exp_cash": 0.0,      # from the 4th store on, open only with this much cash (no borrowing for stores)
    "lean": 1.0,          # marketing/assortment multiplier while in debt AND losing money (promotion is free: kept)
    "crowd_price": 0.0,   # price -= crowd_price * (mean rivals per own region - 1), floored at crowd_floor
    "crowd_floor": 1.7,
    "war_cash": 0.0,      # at full size with this much cash: price war at war_price until cash halves
    "war_price": 1.6,
    "crowd_cash": -1e18,  # undercut only while cash >= this (0: stop the price war while we are the one in debt)
    "hold_crowded": 0.0,  # while cash < this, open only regions with fewer than 2 rivals (no borrowing into crowds)
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
    # round 4, aimed at v6: its own plan (collides with it), and a deeper undercut that out-prices its crowd price
    "v6-mirror": {**DEFAULT, "order": "score", "k": 0.2, "wage": 0.40, "crowd_price": 0.1, "service": 0.2},
    "undercutter": {**DEFAULT, "order": "score", "k": 0.2, "wage": 0.40, "crowd_price": 0.25, "crowd_floor": 1.4, "service": 0.2},
    "home-raider": {**DEFAULT, "order": "steal", "price": 1.8, "crowd_price": 0.15, "crowd_floor": 1.5, "service": 0.2},
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
        if p["brake_cash"] and float(obs[idx["cash"]]) < p["brake_cash"] and float(obs[idx["last_profit"]]) < p["brake_profit"]:
            return 0  # stores are not paying for themselves: do not open another one
        if p["exp_cash"] and float(obs[idx["stores"]]) >= 3 and float(obs[idx["cash"]]) < p["exp_cash"]:
            return 0  # the game lends up to 5,000 for a store; borrowing for one mid-game starts the debt spiral
        if p["order"] == "ring":
            return _ring_target(obs, idx, n_regions, int(p["cap"]))
        if float(obs[idx["stores"]]) >= p["cap"]:
            return 0
        empty = list(geo._own_empty_regions(obs, idx=idx, candidates=range(1, n_regions)))
        rivals = lambda r: float(obs[idx[f"others_present_region_{r}"]])  # noqa: E731
        if p["hold_crowded"] and float(obs[idx["cash"]]) < p["hold_crowded"]:
            empty = [r for r in empty if rivals(r) < 2]
        if not empty:
            return 0
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
        if p["order"] == "score":  # demand / (1 + k * rivals): k < 0 walks INTO crowds (aggressive), k > 0 avoids them
            pool = [r for r in empty if cls(r) in (0, 1)] or empty
            k = p.get("k", 0.0)
            return min(pool, key=lambda r: (-({0: 1400.0, 1: 900.0}.get(cls(r), 500.0) / max(0.2, 1.0 + k * rivals(r))), -r))
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
    war = [False]

    def predict(obs: Any, *, episode_start: bool = False, **_kw: Any) -> Any:
        if episode_start:
            tick[0] = 0
            war[0] = False
        t = tick[0]
        tick[0] += 1
        price = p["early_price"] if (p["early_price"] and t < p["switch_tick"]) else p["price"]
        cash = float(obs[idx["cash"]])
        if p["crowd_price"] and cash >= p["crowd_cash"]:
            own = [r for r in range(1, n_regions) if float(obs[idx[f"stores_region_{r}"]]) > 0]
            crowd = sum(float(obs[idx[f"others_present_region_{r}"]]) for r in own) / max(1, len(own))
            price = max(p["crowd_floor"], price - p["crowd_price"] * max(0.0, crowd - 1.0))
        if p["war_cash"]:  # hysteresis: start when rich at full size, stop once the war chest is half spent
            if not war[0] and cash >= p["war_cash"] and float(obs[idx["stores"]]) >= p["cap"]:
                war[0] = True
            elif war[0] and cash < p["war_cash"] / 2:
                war[0] = False
            if war[0]:
                price = min(price, p["war_price"])
        mkt, asrt = p["marketing"], p["assortment"]
        if cash < 0 and float(obs[idx["last_profit"]]) < 0:
            mkt, asrt = mkt * p["lean"], asrt * p["lean"]
        wage = p["wage"]
        frugal = t < p["frugal"]  # opening: no research or wage so cash stays above the store gate on round 3
        if p["smart_pay"]:
            wage = min(1.0, wage + 4.0 * (1.0 - float(obs[idx["employee_happiness"]])))
        return encode_action(
            geo._structured_action(
                price=price, marketing=mkt, assortment=asrt,
                promotion=p["promotion"], expansion_choice=target(obs), research=0.0 if frugal else p["research"],
                service_level=p["service"], automation_tier=0, loyalty_spend=0.0,
                n_regions=n_regions, wage_spend=0.0 if frugal else wage, warehouse_invest=0.0,
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
    "brk": {**DEFAULT, "order": "mirror", "brake_cash": 2500.0, "brake_profit": 60.0},
    "brk-calm2": {**DEFAULT, "order": "calm2", "brake_cash": 2500.0, "brake_profit": 60.0},
    "brk-h3t1": {**DEFAULT, "order": "hyb3", "thr": 1, "brake_cash": 2500.0, "brake_profit": 60.0},
    "h3t1": {**DEFAULT, "order": "hyb3", "thr": 1},
    "w40": {**DEFAULT, "order": "score", "k": 0.0, "wage": 0.40},
    "k02": {**DEFAULT, "order": "score", "k": 0.2},
    "w40k02": {**DEFAULT, "order": "score", "k": 0.2, "wage": 0.40},
    "r0": {**DEFAULT, "order": "mirror", "research": 0.0},
    "f2": {**DEFAULT, "order": "mirror", "frugal": 2},
    "r0f2": {**DEFAULT, "order": "mirror", "research": 0.0, "frugal": 2},
    "p15": {**DEFAULT, "order": "mirror", "price": 1.5},
    "p17-early": {**DEFAULT, "order": "mirror", "early_price": 1.5, "switch_tick": 150, "price": 1.9},
    # v6 (contender_search.py --v6 3, "crowd+svc"): recipe5 + undercut where our regions are crowded + service 0.2
    "v6": {**DEFAULT, "order": "score", "k": 0.2, "wage": 0.40, "crowd_price": 0.1, "crowd_floor": 1.7,
           "service": 0.2, "research": 0.1},
    # v6.1 (contender_search.py --v6 4): k 1.0, so a METRO with 2+ rivals scores below an empty AFFLUENT; v6
    # collapsed from home 9 by opening three rivals' homes with no cash. 529k per game, worst game +262k (v6: -232k)
    "v6.1": {**DEFAULT, "order": "score", "k": 1.0, "wage": 0.40, "crowd_price": 0.1, "crowd_floor": 1.7,
             "service": 0.2, "research": 0.1},
}
