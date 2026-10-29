"""Scripted probe seat vs Orcho + Krischo (3 seats, no NPC pressure) to measure which posture beats them.

Run from repo root: .venv/Scripts/python.exe agents/Yavkata/sac/cs_probe.py
Uses ArenaMatch unchanged and replaces one seat's policy with a scripted closure (src/ untouched).
"""
import itertools, sys
import numpy as np
from retail_simulator.core.encoding import encode_action
from retail_simulator.core.schema import observation_schema
from retail_simulator.envs.arena import arena_balanced_config
from retail_simulator.envs.arena_match import ArenaMatch, ArenaSeatSpec
from retail_simulator.envs.geo_archetypes import _structured_action

O = "agents/orcho/market-share/checkpoint.zip"
K = "agents/Krischo/rppo-lstm/checkpoint.zip"
cfg = arena_balanced_config()
N = len(cfg.demand.regions)
idx = {f.name: i for i, f in enumerate(observation_schema(N))}
CLS = ["METRO", "AFFL", "LOYAL", "FRONT"]
by_demand = sorted(range(1, N), key=lambda r: (-cfg.demand.regions[r].base_regional_demand, r))

def probe(price, mkt, max_stores, order, wage=0.5, start_tick=0, service=1.0, assort=1.0, promo=1.0, loyalty=0.0):
    tick = [0]
    def predict(obs, *, episode_start=False, **_):
        if episode_start: tick[0] = 0
        t = tick[0]; tick[0] += 1
        target = 0
        if t >= start_tick and obs[idx["stores"]] < max_stores:
            for r in order:
                if obs[idx[f"stores_region_{r}"]] == 0:
                    target = r; break
        return encode_action(_structured_action(price=price, marketing=mkt, assortment=assort, promotion=promo,
            expansion_choice=target, research=0.0, service_level=service, automation_tier=0,
            loyalty_spend=loyalty, n_regions=N, wage_spend=wage))
    return predict

def run(pol, seed=1, homes=(1, 5, 9), rounds=1000):
    seats = [ArenaSeatSpec("Orcho", "rl", checkpoint=O), ArenaSeatSpec("Krischo", "rl", checkpoint=K),
             ArenaSeatSpec("probe", "rl", checkpoint=O)]
    m = ArenaMatch(cfg, seats, seed=seed, homes=homes)
    m._policies[2] = pol
    tot = [0.0] * 3
    for _ in range(rounds):
        r = m.step()
        for i, s in enumerate(r.seats): tot[i] += s.profit
    st = r.region_stores
    cls = {n: [0] * 4 for n in range(3)}
    for reg in range(1, N):
        for s in range(3):
            if st[reg][s]: cls[s][(reg - 1) % 4] += 1
    return tot, [sum(st[reg][s] for reg in range(N)) for s in range(3)], cls

if __name__ == "__main__":
    grid = [
        ("base-O-like p2.0 m1.0 12st", dict(price=2.0, mkt=1.0, max_stores=12, order=by_demand)),
        ("p1.5 m1.0 12st", dict(price=1.5, mkt=1.0, max_stores=12, order=by_demand)),
        ("p1.0 m1.0 12st", dict(price=1.0, mkt=1.0, max_stores=12, order=by_demand)),
        ("p1.5 m0.3 12st", dict(price=1.5, mkt=0.3, max_stores=12, order=by_demand)),
        ("p2.0 m1.0 6st", dict(price=2.0, mkt=1.0, max_stores=6, order=by_demand)),
        ("p2.0 m1.0 18st", dict(price=2.0, mkt=1.0, max_stores=18, order=by_demand)),
        ("p2.0 m1.0 24st", dict(price=2.0, mkt=1.0, max_stores=24, order=by_demand)),
        ("p2.0 m0.5 12st", dict(price=2.0, mkt=0.5, max_stores=12, order=by_demand)),
        ("p2.0 m1.0 12st reversed(small first)", dict(price=2.0, mkt=1.0, max_stores=12, order=by_demand[::-1])),
    ]
    only = sys.argv[1:] 
    for name, kw in grid:
        tot, stores, cls = run(probe(**kw))
        print(f"{name:40s} Orcho {tot[0]:>10,.0f} Krischo {tot[1]:>10,.0f} probe {tot[2]:>10,.0f} stores {stores} probe cls {cls[2]} O cls {cls[0]} K cls {cls[1]}", flush=True)
