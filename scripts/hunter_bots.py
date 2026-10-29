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
ORDERS = ("index", "demand", "ring", "steal", "quiet", "affluent")
DEFAULT = {
    "price": 2.0, "marketing": 1.0, "assortment": 1.0, "promotion": 1.0, "service": 1.0, "research": 0.2,
    "wage": 0.45, "smart_pay": 1, "cap": 12, "order": "demand",
}
# Hand-made personas from the organizer's reward-design hints, as starting points for the search.
PERSONAS = {
    "expander": {**DEFAULT, "cap": 16, "order": "index", "price": 1.6},
    "defender": {**DEFAULT, "cap": 8, "order": "ring", "marketing": 1.0, "price": 1.9},
    "price-fighter": {**DEFAULT, "price": 1.0, "promotion": 1.0, "order": "demand"},
    "cautious": {**DEFAULT, "cap": 6, "order": "quiet", "marketing": 0.5, "promotion": 0.3},
    "spoiler": {**DEFAULT, "order": "steal", "price": 1.5},
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
        if p["order"] == "affluent":
            return min(empty, key=lambda r: ({1: 0, 0: 1, 2: 2, 3: 3}[cls(r)], r))
        return min(empty, key=lambda r: (cls(r), r))  # demand

    def predict(obs: Any, *, episode_start: bool = False, **_kw: Any) -> Any:
        wage = p["wage"]
        if p["smart_pay"]:
            wage = min(1.0, wage + 4.0 * (1.0 - float(obs[idx["employee_happiness"]])))
        return encode_action(
            geo._structured_action(
                price=p["price"], marketing=p["marketing"], assortment=p["assortment"],
                promotion=p["promotion"], expansion_choice=target(obs), research=p["research"],
                service_level=p["service"], automation_tier=0, loyalty_spend=0.0,
                n_regions=n_regions, wage_spend=wage, warehouse_invest=0.0,
            )
        )

    return predict
