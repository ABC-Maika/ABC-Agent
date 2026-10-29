"""Staged-expansion experts (interleave METRO / AFFLUENT / LOYAL) measured in six-seat matches.

A plan is an ordered list of stages ``(target_total_stores, class_ranks, cash_buffer, need_profit)``.
Stages are tried in order: the first stage whose target is not yet reached decides the next store.
Inside a stage the region is the unheld one of the allowed classes with the highest base demand, then lowest id.
If the stage has no free region left it is skipped; if cash < capex + buffer (or last_profit <= 0 when need_profit) we wait.
Class ranks: 0 METRO, 1 AFFLUENT, 2 LOYAL, 3 FRONTIER. The match harness is ``krischo_order.match`` (reused, not edited).
Run from the repo root:  .venv\\Scripts\\python.exe agents\\Krischo\\scripts\\krischo_stages.py --plans A,B --n 12
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import krischo_league as kl  # noqa: E402
import krischo_optimize as ko  # noqa: E402
import krischo_order as kord  # noqa: E402

M, A_, L = [0], [1], [2]
OL = kord.ORCHO_LIKE  # posture baseline (Orcho-like recipe)


def S(target, ranks, buf=0.0, prof=False):
    return [target, ranks, buf, prof]


PLANS: dict[str, dict] = {
    "flat12": {"max_stores": 12, "plan": [S(12, M), S(12, A_)]},  # = Orcho-like order, sanity row
    "A": {"max_stores": 12, "plan": [S(3, M), S(6, A_), S(9, M), S(12, A_)]},
    "A+L": {"max_stores": 14, "plan": [S(3, M), S(6, A_), S(9, M), S(12, A_), S(14, L, 6000, True)]},
    "B": {"max_stores": 12, "plan": [S(2, M), S(4, A_), S(6, M), S(8, A_), S(10, M), S(12, A_)]},
    "C": {"max_stores": 14, "plan": [S(4, M), S(10, A_), S(12, M), S(14, L, 6000, True)]},
    "D 4M-4A-rest": {"max_stores": 12, "plan": [S(4, M), S(8, A_), S(10, M), S(12, A_)]},
    "E A+buf3000": {"max_stores": 12, "plan": [S(3, M), S(6, A_, 3000), S(9, M, 3000), S(12, A_, 3000)]},
    "F A+profit": {"max_stores": 12, "plan": [S(3, M), S(6, A_, 0, True), S(9, M, 0, True), S(12, A_, 0, True)]},
    "G flat+L": {"max_stores": 14, "plan": [S(12, M + A_), S(14, L, 6000, True)]},
    "H A harvest900": {"max_stores": 12, "harvest_tick": 900, "harvest_cut": 1.0, "plan": [S(3, M), S(6, A_), S(9, M), S(12, A_)]},
    "I 3M-3A-3M-3A-6": {"max_stores": 10, "plan": [S(3, M), S(6, A_), S(9, M), S(10, A_)]},
    "flat12 lean": {"max_stores": 12, "lean": 1, "plan": [S(12, M), S(12, A_)]},
    "flat12 gate2": {"max_stores": 12, "gate": 2, "plan": [S(12, M), S(12, A_)]},
    "flat12 lean+gate2": {"max_stores": 12, "lean": 1, "gate": 2, "plan": [S(12, M), S(12, A_)]},
    "A lean+gate2": {"max_stores": 12, "lean": 1, "gate": 2, "plan": [S(3, M), S(6, A_), S(9, M), S(12, A_)]},
    "B lean+gate2": {"max_stores": 12, "lean": 1, "gate": 2, "plan": [S(2, M), S(4, A_), S(6, M), S(8, A_), S(10, M), S(12, A_)]},
    "flat10": {"max_stores": 10, "plan": [S(10, M), S(10, A_)]},
    "flat9": {"max_stores": 9, "plan": [S(9, M), S(9, A_)]},
    "flat8": {"max_stores": 8, "plan": [S(8, M), S(8, A_)]},
    "flat12 harvest850": {"max_stores": 12, "harvest_tick": 850, "harvest_cut": 1.0, "plan": [S(12, M), S(12, A_)]},
    "J A 16": {"max_stores": 16, "plan": [S(3, M), S(6, A_), S(9, M), S(12, A_), S(16, L, 6000, True)]},
}
REG: list = []


class StagedExpert(ko.Expert):
    """Expert whose expansion follows ``params['plan']``; without a plan it is the plain Expert. Records a light trace."""

    def __init__(self, params):
        super().__init__(params)
        self.plan = params.get("plan")
        self.trace = {"stores": [], "cash": [], "profit": []}
        self.final = {}
        REG.append(self)

    def reset(self):
        super().reset()
        self.trace = {"stores": [], "cash": [], "profit": []}

    def _expansion(self, obs, tick):
        if self.plan is None:
            return super()._expansion(obs, tick)
        p, idx = self.p, self.idx
        stores = obs[idx["stores"]]
        if tick >= p["harvest_tick"] or stores >= p["max_stores"]:
            return 0
        cash = obs[idx["cash"]]
        gate = p.get("gate", 0)
        if gate and stores >= gate and obs[idx["last_profit"]] <= 0:
            return 0  # survival gate: do not add stores while the current ones lose money
        for target, ranks, buf, prof in self.plan:
            if stores >= target:
                continue
            cand = [(-self.base_demand[r], r) for r in range(1, ko.N_REGIONS) if self.class_rank[r] in ranks and obs[idx[f"stores_region_{r}"]] == 0.0]
            if not cand:
                continue
            if cash < ko.CAPEX + max(buf, p["cash_buffer"]) or (prof and obs[idx["last_profit"]] <= 0):
                return 0
            return min(cand)[1]
        return 0

    def __call__(self, obs):
        idx = self.idx
        self.trace["stores"].append(float(obs[idx["stores"]]))
        self.trace["cash"].append(float(obs[idx["cash"]]))
        self.trace["profit"].append(float(obs[idx["last_profit"]]))
        wc = self.p.get("wage_cash", 0)
        ws = self.p.get("wage_stores", 0)
        if (wc and obs[idx["cash"]] < wc) or (ws and obs[idx["stores"]] < ws):
            # wage gate: no staff pay until cash reaches wage_cash (early wage spend kills thin 3-store starts)
            saved = self.p
            self.p = {**saved, "wage_spend": 0.0, "k_wage_h": 0.0, "k_wage": 0.0}
            try:
                return super().__call__(obs)
            finally:
                self.p = saved
        if self.p.get("lean") and self.tick >= 5 and obs[idx["last_profit"]] < 0:
            # survival mode: cut every discretionary spend lever while losing money
            saved = self.p
            self.p = {**saved, "marketing": 0.2, "assortment": 0.2, "promotion": 0.0, "research": 0.0, "wage_spend": 0.2, "k_wage_h": 0.0, "k_mkt": 0.0}
            try:
                return super().__call__(obs)
            finally:
                self.p = saved
        if self.tick == 999:
            held = [r for r in range(1, ko.N_REGIONS) if obs[idx[f"stores_region_{r}"]] > 0]
            by = {}
            for r in held:
                by.setdefault(self.class_rank[r], []).append(float(obs[idx[f"market_share_region_{r}"]]))
            self.final = {"classes": {k: [len(v), float(np.mean(v))] for k, v in by.items()}}
        return super().__call__(obs)


def smatch(job):
    """kord.match with StagedExpert in every expert seat; returns (totals, summary of seat 0)."""
    ko.Expert = StagedExpert
    REG.clear()
    tot = kord.match(job)
    me = REG[0]
    st, ca, pr = np.array(me.trace["stores"]), np.array(me.trace["cash"]), np.array(me.trace["profit"])
    buy = [int(i) for i in np.where(np.diff(st) > 0)[0] + 1]
    drops = [int(i) for i in np.where(np.diff(st) < 0)[0] + 1]
    debt = ca < 0
    # longest run of ticks with cash below -5000 (credit-limit region of the restructuring trap)
    run = best = 0
    for d in ca < -5000:
        run = run + 1 if d else 0
        best = max(best, run)
    summ = {
        "stores": [float(st[min(t, len(st) - 1)]) for t in (100, 300, 600, 999)],
        "buy_ticks": buy,
        "drop_ticks": drops,
        "min_cash": float(ca.min()),
        "min_cash_tick": int(ca.argmin()),
        "debt_ticks": int(debt.sum()),
        "deep_debt_run": int(best),
        "profit_first_pos": int(np.argmax(pr > 0)) if (pr > 0).any() else -1,
        "final": me.final,
    }
    return tot, summ


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plans", default="all", help="comma list of plan names (see PLANS) or 'all'")
    ap.add_argument("--base", default="agents/Krischo/rppo-lstm/expert_params_v5.json")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--posture", default="{}", help="JSON overrides applied on top of the Orcho-like posture (all plans)")
    ap.add_argument("--out", default="runs/strategy/stages.json")
    a = ap.parse_args()
    base = {**ko.BASELINE, **json.loads(Path(a.base).read_text())}
    pool = [{**ko.BASELINE, **p} for p in json.loads(Path(kord.POOL).read_text())]
    names = list(PLANS) if a.plans == "all" else [x.strip() for x in a.plans.split(",")]
    names = [n for n in names if n != "v5"] + (["v5"] if "v5" in names else [])
    post = {**OL, **json.loads(a.posture)}
    jobs, tags = [], []
    for name in names:
        over = {"max_class": 2} if name == "v5" else {**post, **PLANS[name]}
        for i in range(a.n):
            s = a.seed0 + i
            opp = [("zip", z) for z in kord.OPPS] + [("expert", e) for e in kl.draw(pool, s, 2)]
            jobs.append(([("expert", {**base, **over}), *opp], s))
            tags.append(name)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(smatch, jobs, chunksize=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps([{"plan": t, "job_seed": j[1], "totals": r[0], "summ": r[1]} for t, j, r in zip(tags, jobs, res)]))
    print(f"{'plan':<18}{'ours':>10}{'Orcho':>10}{'min':>10}{'rank':>6}{'beat':>7}  stores@100/300/600  buyticks(mean)  class share")
    for name in names:
        r = [x for x, t in zip(res, tags) if t == name]
        tot = [x[0] for x in r]
        rank = [1 + sum(o > x[0] for o in x[1:]) for x in tot]
        st = np.mean([x[1]["stores"][:3] for x in r], axis=0)
        cs = {}
        for x in r:
            for k, (n, sh) in x[1]["final"].get("classes", {}).items():
                cs.setdefault(k, []).append((n, sh))
        cls = " ".join(f"c{k}:{np.mean([v[0] for v in vs]):.1f}@{np.mean([v[1] for v in vs]):.2f}" for k, vs in sorted(cs.items()))
        print(f"{name:<18}{np.mean([x[0] for x in tot]):>10,.0f}{np.mean([x[1] for x in tot]):>10,.0f}{min(x[0] for x in tot):>10,.0f}{np.mean(rank):>6.2f}{sum(x[0] > x[1] for x in tot):>5}/{len(tot)}  {st[0]:.1f}/{st[1]:.1f}/{st[2]:.1f}  {cls}")


if __name__ == "__main__":
    main()





