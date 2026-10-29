"""Six-seat (or smaller) no-NPC evaluation over rotated homes and seeds, with per-match traces.

    python agents/Yavkata/sac-v2/eval_field.py --field full6 --cands a=path.zip b=path.zip --seeds 1,2 --out runs/v2/eval.json
Fields: full6 = cand + Orcho + Krischo + Jack + JackyT + YavkataRPPO ; top2 = cand + Orcho + Krischo
        orchoX2 = cand + Orcho + Krischo + 3 more Orcho-like? (not used)
Candidate sits in seat 0; the six METRO homes rotate so the candidate gets each home once per seed.
Output: per-match total profit of every seat, final stores, cash trough, profit at round 250/500/750.
"""
from __future__ import annotations

import argparse, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OPP = {
    "Orcho": "agents/orcho/market-share/checkpoint.zip",
    "Krischo": "agents/Krischo/rppo-lstm/checkpoint.zip",
    "Jack": "agents/misho/jack/checkpoint.zip",
    "JackyT": "agents/misho/jacky-t/checkpoint.zip",
    "YavRPPO": "agents/Yavkata/recurrent-ppo/checkpoint.zip",
    "KMlp8": "agents/Krischo/mlp-v8/checkpoint.zip",
    "KMlp9": "agents/Krischo/mlp-v9/checkpoint.zip",
    "OrchoPPO": "agents/orcho/ppo-v5/checkpoint.zip",
}
FIELDS = {
    "full6": ["Orcho", "Krischo", "Jack", "JackyT", "YavRPPO"],
    "top2": ["Orcho", "Krischo"],
    "top2b": ["Orcho", "Krischo", "KMlp9", "OrchoPPO", "Jack"],
}
HOMES6 = [1, 5, 9, 13, 17, 21]


def run_match(job):
    name, cand, field, seed, rot, rounds = job
    from retail_simulator.envs.arena import arena_balanced_config
    from retail_simulator.envs.arena_match import ArenaMatch, ArenaSeatSpec
    opps = FIELDS[field]
    n = 1 + len(opps)
    homes = tuple(HOMES6[(rot + k) % 6] for k in range(n))
    seats = [ArenaSeatSpec(name, "rl", checkpoint=str(Path(cand)))]
    seats += [ArenaSeatSpec(o, "rl", checkpoint=str(REPO / OPP[o])) for o in opps]
    m = ArenaMatch(arena_balanced_config(), seats, seed=seed, homes=homes)
    tot = np.zeros(n); cum = []; stores = []; cash0 = []
    for r in range(rounds):
        res = m.step()
        tot += [s.profit for s in res.seats]
        if (r + 1) % 50 == 0:
            cum.append(tot.round().tolist()); stores.append([s.stores for s in res.seats])
        cash0.append(res.seats[0].cash)
    return dict(cand=name, field=field, seed=seed, rot=rot, homes=homes, names=[name] + opps,
                total=tot.tolist(), cum=cum, stores=stores, min_cash=float(min(cash0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="full6")
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--rots", default="0,1,2,3,4,5")
    ap.add_argument("--rounds", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    jobs = []
    for c in a.cands:
        nm, p = c.split("=")
        for s in map(int, a.seeds.split(",")):
            for r in map(int, a.rots.split(",")):
                jobs.append((nm, str(Path(p).resolve()), a.field, s, r, a.rounds))
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_match, jobs))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res))
    summarize(res)


def summarize(res):
    for cand in dict.fromkeys(r["cand"] for r in res):
        rs = [r for r in res if r["cand"] == cand]
        names = rs[0]["names"]
        arr = np.array([r["total"] for r in rs])
        ranks = (-arr).argsort(1).argsort(1)[:, 0] + 1
        mc = np.mean([r["min_cash"] for r in rs])
        print(f"{cand:14s} n={len(rs)} mean {arr[:,0].mean():>9,.0f} min {arr[:,0].min():>9,.0f} rank {ranks.mean():.2f} "
              f"wins {(ranks==1).sum()} | " + " ".join(f"{n}:{arr[:,i].mean():,.0f}" for i, n in enumerate(names) if i), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summ":
        summarize(json.load(open(sys.argv[2])))
    else:
        main()
