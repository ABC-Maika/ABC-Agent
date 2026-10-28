"""Which lever matters? Play scripted copies of a recipe, changing one thing at a time.

The base recipe copies what the strongest rival did (max price, steady marketing, max assortment,
promotion, service and loyalty, no research, no wage spend, automation tier 2, stop expanding at
15 stores). Each variant changes ONE setting. Every variant is seated against the five serious
rivals; the table shows its score, staff happiness and stores, averaged over two seatings.

    python scripts/arena_ablate.py
"""

from __future__ import annotations

from collections.abc import Callable
import sys
from typing import Any

import numpy as np

from retail_simulator.core.encoding import encode_action
from retail_simulator.envs import arena_match
from retail_simulator.envs import geo_archetypes as geo
from retail_simulator.envs.arena import arena_home_regions
from retail_simulator.envs.arena_match import ArenaMatch, ArenaSeatSpec, resolve_arena_world

S = "runs/serious"
RIVALS = {
    "rival-a2c-s1": f"{S}/rival-a2c-s1.zip",
    "rival-a2c-s2": f"{S}/rival-a2c-s2.zip",
    "rival-a2c-s3": f"{S}/rival-a2c-s3.zip",
    "rival-ppo-profit": f"{S}/rival-ppo-profit.zip",
    "rival-ppo-weighted": f"{S}/rival-ppo-weighted.zip",
}
BASE = {
    "price": 2.0, "marketing": 0.81, "assortment": 1.0, "promotion": 1.0, "service_level": 1.0,
    "loyalty_spend": 1.0, "research": 0.0, "wage_spend": 0.0, "warehouse_invest": 0.0,
    "automation_tier": 2, "cap": 15,
}
VARIANTS = {
    "base (copy of winner)": {},
    "automation tier 0": {"automation_tier": 0},
    "automation tier 1": {"automation_tier": 1},
    "loyalty 0.2": {"loyalty_spend": 0.2},
    "marketing 0": {"marketing": 0.0},
    "marketing 1.0": {"marketing": 1.0},
    "research 0.7": {"research": 0.7},
    "wage 0.3": {"wage_spend": 0.3},
    "wage 0.6": {"wage_spend": 0.6},
    "stores cap 8": {"cap": 8},
    "stores cap 23": {"cap": 23},
    "price 1.5": {"price": 1.5},
}


BASE3 = {**BASE, "marketing": 1.0, "wage_spend": 0.5, "cap": 10, "loyalty_spend": 0.0}
VARIANTS3 = {
    "recipe (mkt1 wage.5 cap10)": {},
    "wage 0.40": {"wage_spend": 0.40},
    "wage 0.45": {"wage_spend": 0.45},
    "research 0.3": {"research": 0.3},
    "research 0.6": {"research": 0.6},
    "cap 12": {"cap": 12},
    "cap 14": {"cap": 14},
    "wage.45 research.3 cap12": {"wage_spend": 0.45, "research": 0.3, "cap": 12},
}
VARIANTS4 = {
    "cap 6": {"cap": 6},
    "cap 7": {"cap": 7},
    "cap 8": {"cap": 8},
    "cap 6 price 1.88": {"cap": 6, "price": 1.88},
    "cap 6 price 1.88 res.05": {"cap": 6, "price": 1.88, "research": 0.05},
}
VARIANTS5 = {
    "demand order cap 10": {"order": "demand"},
    "demand order cap 8": {"order": "demand", "cap": 8},
    "aff-first cap 10": {"order": "aff"},
    "aff-first cap 8": {"order": "aff", "cap": 8},
    "quiet-first cap 10": {"order": "quiet"},
    "quiet-first cap 8": {"order": "quiet", "cap": 8},
}
D = {"order": "demand"}
VARIANTS6A = {  # aggression on top of the demand order
    "demand cap 10": {**D},
    "demand cap 12": {**D, "cap": 12},
    "demand cap 14": {**D, "cap": 14},
    "demand cap 18": {**D, "cap": 18},
    "demand cap 24": {**D, "cap": 24},
}
VARIANTS6B = {
    "demand wage .45": {**D, "wage_spend": 0.45},
    "demand research .3": {**D, "research": 0.3},
    "demand price 1.88": {**D, "price": 1.88},
    "demand price 1.7": {**D, "price": 1.7},
    "demand cap 12 wage .45 res .2": {**D, "cap": 12, "wage_spend": 0.45, "research": 0.2},
}
TEAM = {
    "krischo": "agents/Krischo/rppo-lstm/checkpoint.zip",
    "yavkata": "agents/Yavkata/recurrent-ppo/checkpoint.zip",
    "misho-jack": "agents/misho/jack/checkpoint.zip",
    "rival-a2c-s2": f"{S}/rival-a2c-s2.zip",
    "rival-a2c-s1": f"{S}/rival-a2c-s1.zip",
}
COMBOS = {
    "mkt1 wage.6 cap8": {"marketing": 1.0, "wage_spend": 0.6, "cap": 8},
    "mkt1 wage.45 cap8": {"marketing": 1.0, "wage_spend": 0.45, "cap": 8},
    "mkt1 wage.8 cap8": {"marketing": 1.0, "wage_spend": 0.8, "cap": 8},
    "mkt1 wage.6 cap5": {"marketing": 1.0, "wage_spend": 0.6, "cap": 5},
    "mkt1 wage.6 cap6": {"marketing": 1.0, "wage_spend": 0.6, "cap": 6},
    "mkt1 wage.6 cap10": {"marketing": 1.0, "wage_spend": 0.6, "cap": 10},
    "mkt1 wage.6 cap12": {"marketing": 1.0, "wage_spend": 0.6, "cap": 12},
}


CLASS_RANK = {
    "demand": {0: 0, 1: 1, 2: 2, 3: 3},  # METRO, AFFLUENT, LOYAL, FRONTIER
    "aff": {1: 0, 2: 1, 0: 2, 3: 3},  # AFFLUENT, LOYAL, METRO, FRONTIER
}


def _pick(order: str, obs: Any, idx: dict[str, int], n_regions: int) -> int:
    """Which empty region to open next: by index, by class value, or the quietest first."""
    empty = list(geo._own_empty_regions(obs, idx=idx, candidates=range(1, n_regions)))
    if not empty:
        return 0
    if order == "index":
        return empty[0]
    rank = CLASS_RANK["demand" if order in ("demand", "quiet") else order]
    def key(r: int) -> tuple[int, int, int]:
        rival = float(obs[idx[f"others_present_region_{r}"]]) > 0 if order == "quiet" else False
        return (int(rival), rank[(r - 1) % 4], r)
    return min(empty, key=key)


def _make(params: dict[str, Any], n_regions: int, idx: dict[str, int]) -> Callable[..., Any]:
    tick = [0]

    def predict(obs: Any, *, episode_start: bool = False, **_kw: Any) -> Any:
        if episode_start:
            tick[0] = 0
        tick[0] += 1
        target = 0
        if float(obs[idx["stores"]]) < params["cap"]:
            target = _pick(params.get("order", "index"), obs, idx, n_regions)
        return encode_action(
            geo._structured_action(
                price=params["price"], marketing=params["marketing"],
                assortment=params["assortment"], promotion=params["promotion"],
                expansion_choice=target, research=params["research"],
                service_level=params["service_level"], automation_tier=params["automation_tier"],
                loyalty_spend=params["loyalty_spend"], n_regions=n_regions,
                wage_spend=params["wage_spend"], warehouse_invest=params["warehouse_invest"],
            )
        )

    return predict


def main() -> int:
    base = arena_match.geo_archetypes
    variants = COMBOS if "combos" in sys.argv[1:] else VARIANTS
    rivals = RIVALS
    if "team" in sys.argv[1:] or "team4" in sys.argv[1:] or "team5" in sys.argv[1:]:
        variants, rivals = (VARIANTS5 if "team5" in sys.argv[1:] else VARIANTS4 if "team4" in sys.argv[1:] else VARIANTS3), TEAM
        BASE.update(BASE3)
    if "team6a" in sys.argv[1:] or "team6b" in sys.argv[1:]:
        variants, rivals = (VARIANTS6A if "team6a" in sys.argv[1:] else VARIANTS6B), TEAM
        BASE.update(BASE3)
    labels = tuple(variants)

    def with_variants(n_regions: int, table: Any) -> dict[str, Callable[..., Any]]:
        idx = geo._name_to_index(n_regions)
        extra = {n: _make({**BASE, **o}, n_regions, idx) for n, o in variants.items()}
        return {**base(n_regions, table), **extra}

    arena_match.geo_archetypes = with_variants
    arena_match.GEO_ARCHETYPE_LABELS = (*arena_match.GEO_ARCHETYPE_LABELS, *labels)
    cfg, _, _ = resolve_arena_world(balanced=True)
    homes_base = arena_home_regions(6)
    print(f"{'variant':26}{'score':>11}{'happiness':>10}{'stores':>7}{'rank':>6}   (mean of 2 seatings)")
    for name in labels:
        seats = [ArenaSeatSpec(name=name, kind="npc", archetype=name)]
        seats += [ArenaSeatSpec(name=n, kind="rl", checkpoint=p) for n, p in rivals.items()]
        sc, hp, stz, rk = [], [], [], []
        for shift in ((0,) if any(a in sys.argv[1:] for a in ('team', 'team4', 'team5')) else (0, 3)):
            homes = tuple(homes_base[(i + shift) % 6] for i in range(6))
            m = ArenaMatch(cfg, seats, seed=0, homes=homes)
            tot = np.zeros(6)
            last = None
            happy = []
            for _ in range(1000):
                last = m.step()
                tot += [s.reward for s in last.seats]
                happy.append(last.seats[0].employee_happiness)
            sc.append(tot[0])
            hp.append(np.mean(happy[100:]))
            stz.append(last.seats[0].stores)
            rk.append(int(np.argsort(-tot).tolist().index(0)) + 1)
        print(f"{name:26}{np.mean(sc):11,.0f}{np.mean(hp):10.2f}{np.mean(stz):7.0f}{np.mean(rk):6.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
