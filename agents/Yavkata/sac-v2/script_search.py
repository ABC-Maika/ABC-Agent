"""Scripted-demonstrator search in the full6 field (Orcho, Krischo, Jack, JackyT, YavRPPO): expansion order/cap/levers.
    python agents/Yavkata/sac-v2/script_search.py
The winner is the demonstrator that train_v2.py clones into SAC."""
from __future__ import annotations
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_field import FIELDS, OPP, HOMES6

CAPEX = 5000.0
DEF = dict(order="demand", cap=12, price=2.0, marketing=1.0, assortment=1.0, promotion=1.0, service=1.0,
           research=0.2, warehouse=0.0, wage_base=0.45, wage_gain=4.0, loyalty=0.0, reserve=0.0, reserve_from=8)


def make_order(cfg, kind):
    regs = cfg.demand.regions; n = len(regs)
    by = sorted(range(1, n), key=lambda r: (-regs[r].base_regional_demand, r))
    metro = [r for r in by if (r - 1) % 4 == 0]; aff = [r for r in by if (r - 1) % 4 == 1]
    loyal = [r for r in by if (r - 1) % 4 == 2]; fr = [r for r in by if (r - 1) % 4 == 3]
    if kind == "demand": return by
    if kind == "m_a_high": return sorted(metro, reverse=True) + sorted(aff, reverse=True)       # Orcho v5 style
    if kind == "m_a_low": return sorted(metro) + sorted(aff)
    if kind == "m_a_low_l": return sorted(metro) + sorted(aff) + sorted(loyal)
    if kind == "m_a_high_l": return sorted(metro, reverse=True) + sorted(aff, reverse=True) + sorted(loyal, reverse=True)
    if kind == "aff_first": return sorted(aff) + sorted(metro)
    if kind == "all_high": return sorted(range(1, n), reverse=True)
    raise ValueError(kind)


def scripted(cfg, p):
    from retail_simulator.core.encoding import encode_action
    from retail_simulator.core.schema import observation_schema
    from retail_simulator.envs.geo_archetypes import _structured_action
    p = {**DEF, **p}; n = len(cfg.demand.regions)
    idx = {f.name: i for i, f in enumerate(observation_schema(n))}
    order = make_order(cfg, p["order"])
    def predict(obs, *, episode_start=False, **_):
        t = 0; st = obs[idx["stores"]]
        res = p["reserve"] if st >= p["reserve_from"] else 0.0
        if st < p["cap"] and obs[idx["cash"]] >= CAPEX + res:
            t = next((r for r in order if obs[idx[f"stores_region_{r}"]] == 0), 0)
        w = min(1.0, max(0.0, p["wage_base"] + p["wage_gain"] * (1.0 - float(obs[idx["employee_happiness"]]))))
        return encode_action(_structured_action(price=p["price"], marketing=p["marketing"], assortment=p["assortment"],
            promotion=p["promotion"], expansion_choice=t, research=p["research"], service_level=p["service"],
            automation_tier=0, loyalty_spend=p["loyalty"], n_regions=n, wage_spend=w, warehouse_invest=p["warehouse"]))
    return predict


def run(job):
    params, field, seed, rot = job
    from retail_simulator.envs.arena import arena_balanced_config
    from retail_simulator.envs.arena_match import ArenaMatch, ArenaSeatSpec
    cfg = arena_balanced_config(); opps = FIELDS[field]; n = 1 + len(opps)
    homes = tuple(HOMES6[(rot + k) % 6] for k in range(n))
    seats = [ArenaSeatSpec("cand", "rl", checkpoint=str(REPO / OPP["Orcho"]))] + [ArenaSeatSpec(o, "rl", checkpoint=str(REPO / OPP[o])) for o in opps]
    m = ArenaMatch(cfg, seats, seed=seed, homes=homes); m._policies[0] = scripted(cfg, params)
    tot = np.zeros(n)
    for _ in range(1000): tot += [s.profit for s in m.step().seats]
    return tot.tolist()


def evaluate(pool, cands, field="full6", seeds=(1,), rots=range(6)):
    jobs = [(p, field, s, r) for p in cands for s in seeds for r in rots]
    res = list(pool.map(run, jobs)); k = len(seeds) * len(rots); out = []
    for i, p in enumerate(cands):
        a = np.array(res[i*k:(i+1)*k]); rk = (-a).argsort(1).argsort(1)[:, 0] + 1
        out.append((float(a[:, 0].mean()), float(a[:, 0].min()), float(rk.mean()), float(a[:, 1].mean())))
        print(json.dumps(p), "mean %.0f min %.0f rank %.2f orcho %.0f" % out[-1], flush=True)
    return out


if __name__ == "__main__":
    cands = [{"order": o} for o in ["demand", "m_a_high", "m_a_low", "m_a_low_l", "m_a_high_l", "aff_first"]]
    with ProcessPoolExecutor(12) as pool:
        evaluate(pool, cands)
