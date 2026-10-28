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


COMBOS = {
    "mkt1 wage.6 cap8": {"marketing": 1.0, "wage_spend": 0.6, "cap": 8},
    "mkt1 wage.45 cap8": {"marketing": 1.0, "wage_spend": 0.45, "cap": 8},
    "mkt1 wage.8 cap8": {"marketing": 1.0, "wage_spend": 0.8, "cap": 8},
    "mkt1 wage.6 cap5": {"marketing": 1.0, "wage_spend": 0.6, "cap": 5},
    "mkt1 wage.6 cap6": {"marketing": 1.0, "wage_spend": 0.6, "cap": 6},
    "mkt1 wage.6 cap10": {"marketing": 1.0, "wage_spend": 0.6, "cap": 10},
    "mkt1 wage.6 cap12": {"marketing": 1.0, "wage_spend": 0.6, "cap": 12},
}


def _make(params: dict[str, Any], n_regions: int, idx: dict[str, int]) -> Callable[..., Any]:
    tick = [0]

    def predict(obs: Any, *, episode_start: bool = False, **_kw: Any) -> Any:
        if episode_start:
            tick[0] = 0
        tick[0] += 1
        target = 0
        if float(obs[idx["stores"]]) < params["cap"]:
            target = geo._sprawler_target(obs, idx=idx, n_regions=n_regions)
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
        seats += [ArenaSeatSpec(name=n, kind="rl", checkpoint=p) for n, p in RIVALS.items()]
        sc, hp, stz, rk = [], [], [], []
        for shift in (0, 3):
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
