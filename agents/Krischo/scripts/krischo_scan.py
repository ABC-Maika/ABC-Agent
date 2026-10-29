"""Metro / Affluent scan: per-region profile, company-wide lever sensitivity, revenue wage gate, LOYAL marginal cost.

New script only (reuses krischo_stages.StagedExpert, krischo_order opponents). Outputs in runs/scan/.
Run from the repo root (<= 4 workers while other jobs run):
  .venv\\Scripts\\python.exe agents\\Krischo\\scripts\\krischo_scan.py --set sweep --n 6 --workers 4
Sets: profile, sweep, wage, loyal, confirm (see SETS).  Seeds default 8000..
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "scripts"))
import krischo_league as kl  # noqa: E402
import krischo_optimize as ko  # noqa: E402
import krischo_order as kord  # noqa: E402
import krischo_stages as ks  # noqa: E402

BASE_FILE = "agents/Krischo/rppo-lstm/expert_params_v6.json"
M, A_, L = ks.M, ks.A_, ks.L
S = ks.S
COGS = 0.44
CLASS = {0: "METRO", 1: "AFFLUENT", 2: "LOYAL", 3: "FRONTIER"}


class ScanExpert(ks.StagedExpert):
    """StagedExpert + revenue-based wage gate (wage_rev_gate: threshold; wage_end: last N ticks wage 0) + per-tick region record."""

    def reset(self):
        super().reset()
        self.rev_on = False
        self.rec = {k: [] for k in ("share", "pie", "others", "stores", "price", "rev", "cash")}

    def __call__(self, obs):
        idx, p = self.idx, self.p
        t = self.tick
        R = range(1, ko.N_REGIONS)
        self.rec["share"].append([float(obs[idx[f"market_share_region_{r}"]]) for r in R])
        self.rec["pie"].append([float(obs[idx[f"total_regional_demand_region_{r}"]]) for r in R])
        self.rec["others"].append([float(obs[idx[f"others_present_region_{r}"]]) for r in R])
        self.rec["stores"].append([float(obs[idx[f"stores_region_{r}"]]) for r in R])
        self.rec["price"].append(float(obs[idx["price_level"]]))
        self.rec["rev"].append(float(obs[idx["last_revenue"]]))
        self.rec["cash"].append(float(obs[idx["cash"]]))
        thr, wend = p.get("wage_rev_gate", 0), p.get("wage_end", 0)
        if thr or wend:
            if obs[idx["last_revenue"]] > thr:
                self.rev_on = True
            off = (thr and not self.rev_on) or (wend and t >= 1000 - wend)
            if off:
                saved = self.p
                self.p = {**saved, "wage_spend": 0.0, "k_wage_h": 0.0, "k_wage": 0.0, "wage_stores": 0}
                try:
                    return super().__call__(obs)
                finally:
                    self.p = saved
        return super().__call__(obs)


def scan_match(job):
    name, params, seed = job
    import krischo_zoo as kz
    from retail_simulator.envs.arena import arena_balanced_config
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    pool = [{**ko.BASELINE, **q} for q in json.loads(Path(kord.POOL).read_text())]
    opp = [("zip", z) for z in kord.OPPS] + [("expert", e) for e in kl.draw(pool, seed, 2)]
    specs = [("scan", params), *opp]
    rng = np.random.default_rng(seed)
    homes = tuple(int(h) for h in rng.permutation(kl.HOMES))
    world_seed = int(rng.integers(0, 2**31 - 1))
    env = RetailParallelEnv(arena_balanced_config(), n_learning_agents=6, home_regions=homes, seed=world_seed, mask_expansion_seats=tuple(range(6)))
    agents = env.possible_agents
    pols = [ScanExpert(p) if k == "scan" else ko.Expert(p) if k == "expert" else kz.ZipAny(p) for k, p in specs]
    for p in pols:
        p.reset()
    obs, _ = env.reset(seed=world_seed)
    tot = [0.0] * 6
    for _ in range(1000):
        obs, rw, *_ = env.step({a: np.asarray(pols[i](obs[a]), dtype=np.float32).ravel() for i, a in enumerate(agents)})
        for i, a in enumerate(agents):
            tot[i] += float(rw[a])
    me = pols[0]
    sh, pie, oth, st = (np.array(me.rec[k]) for k in ("share", "pie", "others", "stores"))
    price = np.array(me.rec["price"])
    regs = []
    for j in range(sh.shape[1]):
        r = j + 1
        held = st[:, j] > 0
        if not held.any():
            continue
        t0 = int(np.argmax(held))
        ages = np.arange(1000) - t0
        ss = held & (ages >= 100) & (np.arange(1000) < 985)
        if ss.sum() < 20:
            continue
        units = sh[ss, j] * pie[ss, j]
        regs.append({"region": r, "cls": me.class_rank[r], "open": t0, "share": float(sh[ss, j].mean()), "pie": float(pie[ss, j].mean()), "others": float(oth[ss, j].mean()), "capped": float((sh[ss, j] >= 0.58).mean()), "units": float(units.mean()), "price": float(price[ss].mean()), "contrib": float((units * (price[ss] - COGS)).mean()), "early_share": float(sh[(ages >= 10) & (ages < 40) & held, j].mean()) if ((ages >= 10) & (ages < 40) & held).any() else None})
    stores_t = np.array(me.trace["stores"])
    return {"name": name, "seed": seed, "total": tot, "regs": regs, "stores": [float(stores_t[min(t, 999)]) for t in (100, 300, 600, 999)], "min_cash": float(min(me.rec["cash"])), "steady_rev": float(np.mean(me.rec["rev"][300:985]))}


def base_params():
    return {**ko.BASELINE, **json.loads(Path(BASE_FILE).read_text())}


MET6 = {"max_stores": 6, "max_class": 1}
AFF = {"max_stores": 9, "plan": [S(3, M), S(9, A_)]}
V6 = {}


def sweep_set():
    out = {}
    for cn, ch in (("MET6", MET6), ("AFF3M6A", AFF)):
        out[f"{cn} base"] = ch
        for v in (1.4, 1.7):
            out[f"{cn} price{v}"] = {**ch, "price": v}
        out[f"{cn} mkt0.5"] = {**ch, "marketing": 0.5}
        out[f"{cn} mkt0.0"] = {**ch, "marketing": 0.0}
        out[f"{cn} asr0.5"] = {**ch, "assortment": 0.5}
        out[f"{cn} asr0.0"] = {**ch, "assortment": 0.0}
        out[f"{cn} promo0"] = {**ch, "promotion": 0.0}
        out[f"{cn} wh0.1"] = {**ch, "warehouse": 0.1}
        out[f"{cn} wh0.3"] = {**ch, "warehouse": 0.3}
        out[f"{cn} wage0"] = {**ch, "wage_spend": 0.0, "k_wage_h": 0.0, "wage_stores": 0}
        out[f"{cn} wage_ws0"] = {**ch, "wage_stores": 0}
        out[f"{cn} wage_ws4"] = {**ch, "wage_stores": 4}
        out[f"{cn} service0.5"] = {**ch, "service": 0.5}
    return out


def wage_set():
    ch = {}
    ch["v6 (wage_stores 8)"] = V6
    ch["wage always"] = {"wage_stores": 0}
    ch["wage none"] = {"wage_stores": 0, "wage_spend": 0.0, "k_wage_h": 0.0}
    ch["revgate1800"] = {"wage_stores": 0, "wage_rev_gate": 1800}
    ch["revgate1800 end15"] = {"wage_stores": 0, "wage_rev_gate": 1800, "wage_end": 15}
    ch["revgate1200 end15"] = {"wage_stores": 0, "wage_rev_gate": 1200, "wage_end": 15}
    ch["revgate2500 end15"] = {"wage_stores": 0, "wage_rev_gate": 2500, "wage_end": 15}
    ch["v6 + end15"] = {"wage_end": 15}
    return ch


def loyal_set():
    o = {}
    for n in (8, 9, 10, 11, 12):
        o[f"flatM+A {n}"] = {"max_stores": n, "plan": [S(n, M), S(n, A_)]}
    o["12 + 1 LOYAL"] = {"max_stores": 13, "plan": [S(12, M + A_), S(13, L, 6000, True)]}
    o["12 + 2 LOYAL"] = {"max_stores": 14, "plan": [S(12, M + A_), S(14, L, 6000, True)]}
    o["12 + 4 LOYAL"] = {"max_stores": 16, "plan": [S(12, M + A_), S(16, L, 6000, True)]}
    o["3M+2L early then M+A 12"] = {"max_stores": 12, "plan": [S(3, M), S(5, L), S(12, M + A_)]}
    o["3M+1L early then M+A 12"] = {"max_stores": 12, "plan": [S(3, M), S(4, L), S(12, M + A_)]}
    o["3M+1L early then 13 total"] = {"max_stores": 13, "plan": [S(3, M), S(4, L), S(13, M + A_)]}
    o["10 + 2 LOYAL"] = {"max_stores": 12, "plan": [S(10, M + A_), S(12, L, 6000, True)]}
    return o


SETS = {"profile": lambda: {"v6": V6, "MET6": MET6, "AFF3M6A": AFF, "A+L14": {"max_stores": 14, "plan": [S(12, M + A_), S(14, L, 6000, True)]}}, "sweep": sweep_set, "wage": wage_set, "loyal": loyal_set}


def summarise(rows, names):
    print(f"{'variant':<28}{'mean':>10}{'min':>10}{'rank':>6}{'beatOr':>8}  stores@100/300/600/999  steadyRev")
    out = {}
    for n in names:
        r = [x for x in rows if x["name"] == n]
        t = [x["total"][0] for x in r]
        rk = np.mean([1 + sum(o > x["total"][0] for o in x["total"][1:]) for x in r])
        st = np.mean([x["stores"] for x in r], axis=0)
        print(f"{n:<28}{np.mean(t):>10,.0f}{min(t):>10,.0f}{rk:>6.2f}{sum(x['total'][0] > x['total'][1] for x in r):>5}/{len(r)}  {'/'.join(f'{s:.1f}' for s in st)}  {np.mean([x['steady_rev'] for x in r]):,.0f}", flush=True)
        out[n] = float(np.mean(t))
    return out


def classtable(rows, names):
    for n in names:
        regs = [g for x in rows if x["name"] == n for g in x["regs"]]
        print(f"-- per-class steady state (age>=100, t<985) for {n}")
        for c in sorted({g["cls"] for g in regs}):
            g = [q for q in regs if q["cls"] == c]
            f = lambda k: np.mean([q[k] for q in g])
            es = [q["early_share"] for q in g if q["early_share"] is not None]
            print(f"  {CLASS[c]:<9} stores={len(g):>3} share={f('share'):.3f} early(10-40)={np.mean(es):.3f} capped={f('capped'):.2f} others={f('others'):.2f} pie={f('pie'):7.0f} units={f('units'):6.0f} rev={f('units')*f('price'):6.0f} contrib/tick={f('contrib'):6.0f} price={f('price'):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", default="profile")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base", default=BASE_FILE)
    ap.add_argument("--only", default="", help="comma list of variant names to keep (confirm)")
    ap.add_argument("--extra", default="", help="JSON file name->params overrides replacing the set")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    base = {**ko.BASELINE, **json.loads(Path(a.base).read_text())}
    sets = json.loads(Path(a.extra).read_text(encoding='utf-8-sig')) if a.extra else SETS[a.set]()
    if a.only:
        sets = {k: v for k, v in sets.items() if k in [x.strip() for x in a.only.split(",")]}
    jobs = [(n, {**base, **ov}, a.seed0 + i) for n, ov in sets.items() for i in range(a.n)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(scan_match, jobs, chunksize=1))
    out = Path(a.out or f"runs/scan/{a.set}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows))
    summarise(rows, list(sets))
    classtable(rows, list(sets) if a.set == "profile" else [])


if __name__ == "__main__":
    main()



