"""Krischo expert v8: the v6 rule expert with Orcho-style ring expansion and wages paid from round 1.

Why: the v6 expert (a) opens regions lowest-index first, so copies of it crowd regions 1, 5 and 9, and (b) pays no wage
until 8 stores, which on some seeds spirals into 0 happiness and debt with 3 stores (seed 3: -213k). v8 changes only those.
Variants (all memoryless functions of the observation, so an MLP can clone them):
  kv-A  v6 + wage from round 1 (lowest-index order kept)
  kv-B  ring order (METRO ring continued from the end of the held arc, then AFFLUENT) + wage from round 1
  kv-C  as B, but it extends the arc on the side with fewer rivals present, and picks the least contested AFFLUENT region

    python agents/Krischo/scripts/krischo_v8.py eval --variant kv-B --seed 5
Own script; nothing in ``src/`` is edited. Run from the repo root.
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))

from krischo_optimize import CAPEX, N_REGIONS, Expert  # noqa: E402

V6 = HERE.parent / "rppo-lstm" / "expert_params_v6.json"
RING = [r for r in range(1, N_REGIONS) if (r - 1) % 4 == 0]  # the six METRO regions, in order
AFFLUENT = [r for r in range(1, N_REGIONS) if (r - 1) % 4 == 1]


class ExpertV8(Expert):
    def __init__(self, params: dict[str, float], mode: str, frugal_ticks: int = 0, frugal_mkt: float | None = None, guard: dict[str, float] | None = None) -> None:
        super().__init__(params)
        self.mode = mode  # "lowest" | "ring" | "ring_smart"
        self.frugal_ticks = frugal_ticks  # opening rounds with no research/wage (and optionally less marketing) to clear the 5,000 cash gate
        self.frugal_mkt = frugal_mkt
        # cash guard: when cash is low AND last tick lost money, cut discretionary spend (a memoryless rule: it reads last_profit)
        self.guard = guard or {}

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        over: dict[str, float] = {}
        if self.tick < self.frugal_ticks:
            over.update(research=0.0, wage_spend=0.0, k_wage_h=0.0, k_wage=0.0)
            if self.frugal_mkt is not None:
                over.update(marketing=self.frugal_mkt)
        g = self.guard
        if g and self.tick >= 10 and obs[self.idx["cash"]] < g["cash"] and obs[self.idx["last_profit"]] < g.get("profit", 0.0):
            over.update(marketing=g["mkt"], assortment=g["assort"], research=0.0)
            if "service" in g:
                over.update(service=g["service"])
            if g.get("stop_exp"):
                over.update(cash_buffer=1e9)  # retreat: no new stores while cash is low and the last tick lost money
        if not over:
            return super().__call__(obs)
        saved = dict(self.p)
        self.p.update(over)
        try:
            return super().__call__(obs)
        finally:
            self.p = saved

    def _expansion(self, obs: np.ndarray, tick: int) -> int:
        if self.mode == "lowest":
            return super()._expansion(obs, tick)
        if self.mode in ("softhigh", "owner"):  # soft penalty like S1, but ties go to the HIGHEST region number (Orcho v5 trick); "owner" = Misho's rule first
            p, idx = self.p, self.idx
            if tick >= p["harvest_tick"] or obs[idx["stores"]] >= p["max_stores"] or obs[idx["cash"]] < CAPEX + p["cash_buffer"]:
                return 0
            pen = p["contest_pen"]
            free, scored = [], []
            for r in range(1, N_REGIONS):
                if self.class_rank[r] < p["max_class"] and obs[idx[f"stores_region_{r}"]] == 0.0:
                    rv = float(obs[idx[f"others_present_region_{r}"]])
                    if rv == 0.0:
                        free.append((-self.base_demand[r], -r))
                    scored.append((-self.base_demand[r] / (1.0 + pen * rv), -r))
            if self.mode == "owner" and free:  # the biggest-demand region nobody else holds
                return -min(free)[1]
            return -min(scored)[1] if scored else 0
        if self.mode == "combo":  # soft penalty (demand / (1 + pen * rivals)) among regions with <= thr rivals; if every region is crowded, the least crowded
            p, idx = self.p, self.idx
            if tick >= p["harvest_tick"] or obs[idx["stores"]] >= p["max_stores"] or obs[idx["cash"]] < CAPEX + p["cash_buffer"]:
                return 0
            thr, pen = p.get("thr", 1), p["contest_pen"]
            cand = []
            for r in range(1, N_REGIONS):
                if self.class_rank[r] < p["max_class"] and obs[idx[f"stores_region_{r}"]] == 0.0:
                    rv = float(obs[idx[f"others_present_region_{r}"]])
                    cand.append((rv > thr, -self.base_demand[r] / (1.0 + pen * rv) if rv <= thr else rv, r))
            return min(cand)[2] if cand else 0
        if self.mode == "thresh":  # give up crowded regions: among regions with <= thr rivals present take the biggest demand (any class); else the least crowded
            p, idx = self.p, self.idx
            if tick >= p["harvest_tick"] or obs[idx["stores"]] >= p["max_stores"] or obs[idx["cash"]] < CAPEX + p["cash_buffer"]:
                return 0
            thr = p.get("thr", 1)
            cand = []
            for r in range(1, N_REGIONS):
                if self.class_rank[r] < p["max_class"] and obs[idx[f"stores_region_{r}"]] == 0.0:
                    rv = float(obs[idx[f"others_present_region_{r}"]])
                    cand.append(((rv > thr), (rv if rv > thr else 0.0), -self.base_demand[r], r))
            return min(cand)[3] if cand else 0
        if self.mode == "high":  # Orcho-v5 style: biggest demand class first, highest region number first (METRO descending, then AFFLUENT)
            p, idx = self.p, self.idx
            if tick >= p["harvest_tick"] or obs[idx["stores"]] >= p["max_stores"] or obs[idx["cash"]] < CAPEX + p["cash_buffer"]:
                return 0
            cand = [(self.class_rank[r], -r) for r in range(1, N_REGIONS) if self.class_rank[r] < p["max_class"] and obs[idx[f"stores_region_{r}"]] == 0.0]
            return -min(cand)[1] if cand else 0
        if self.mode == "steal":  # aggressive: prefer the region (METRO/AFFLUENT) where most rivals already sit, weighted by demand
            p, idx = self.p, self.idx
            if tick >= p["harvest_tick"] or obs[idx["stores"]] >= p["max_stores"] or obs[idx["cash"]] < CAPEX + p["cash_buffer"]:
                return 0
            scored = [
                (-self.base_demand[r] * (1.0 + 0.5 * float(obs[idx[f"others_present_region_{r}"]])), r)
                for r in range(1, N_REGIONS)
                if self.class_rank[r] < p["max_class"] and obs[idx[f"stores_region_{r}"]] == 0.0
            ]
            return min(scored)[1] if scored else 0
        p, idx = self.p, self.idx
        if tick >= p["harvest_tick"] or obs[idx["stores"]] >= p["max_stores"]:
            return 0
        if obs[idx["cash"]] < CAPEX + p["cash_buffer"]:
            return 0
        smart = self.mode == "ring_smart"
        rivals = lambda r: float(obs[idx[f"others_present_region_{r}"]])  # noqa: E731
        n = len(RING)
        owned = {i for i, r in enumerate(RING) if float(obs[idx[f"stores_region_{r}"]]) > 0}
        if owned and len(owned) < n:
            if smart:
                cand = sorted({(i + d) % n for i in owned for d in (1, -1)} - owned)
                # forward neighbour wins ties (Orcho's direction), otherwise the less crowded side
                fwd = {(i + 1) % n for i in owned}
                return RING[min(cand, key=lambda i: (rivals(RING[i]), 0 if i in fwd else 1, i))]
            tails = [q for q in owned if (q + 1) % n not in owned and (q - 1) % n in owned]
            tails = tails or [q for q in owned if (q + 1) % n not in owned]
            return RING[(tails[0] + 1) % n]
        empty = [r for r in AFFLUENT if float(obs[idx[f"stores_region_{r}"]]) == 0.0]
        if not empty:
            return 0
        return min(empty, key=lambda r: ((rivals(r) if smart else 0.0), r))


VARIANTS = {  # name -> (mode, param overrides, frugal opening ticks, frugal opening marketing)
    "kv-A": ("lowest", {"wage_stores": 0}, 0, None),
    "kv-B": ("ring", {"wage_stores": 0}, 0, None),
    "kv-C": ("ring_smart", {"wage_stores": 0}, 0, None),
    "kv-D": ("ring", {"wage_stores": 0}, 2, None),  # B + no research/wage in rounds 1-2
    "kv-E": ("ring", {"wage_stores": 0}, 2, 0.5),  # D + half marketing in rounds 1-2
    "kv-F": ("ring_smart", {"wage_stores": 0}, 2, None),  # C + frugal opening
    # demand / (1 + pen * rivals present): empty regions beat crowded ones (the v6 scored-expansion path)
    "kv-G": ("lowest", {"wage_stores": 0, "contest_pen": 0.5}, 2, None),
    "kv-H": ("lowest", {"wage_stores": 0, "contest_pen": 1.5}, 2, None),
    "kv-J": ("lowest", {"wage_stores": 0, "contest_pen": 3.0}, 2, None),
    "kv-H": ("lowest", {"wage_stores": 0, "contest_pen": 1.5}, 2, None),
    # kv-H + cash guard (cut marketing/assortment/research while cash is low and last tick lost money)
    "kv-K": ("lowest", {"wage_stores": 0, "contest_pen": 1.5}, 2, None, {"cash": 8000.0, "mkt": 0.5, "assort": 0.6}),
    "kv-L": ("lowest", {"wage_stores": 0, "contest_pen": 1.5}, 2, None, {"cash": 8000.0, "mkt": 0.2, "assort": 0.3}),
    "kv-M": ("lowest", {"wage_stores": 0, "contest_pen": 1.5}, 2, None, {"cash": 15000.0, "mkt": 0.5, "assort": 0.6}),
    # squeeze-resistance candidates (one change each on top of kv-H): fewer stores, a cash cushion before each new store, gentler wage rise
    "kv-Q1": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "max_stores": 10}, 2, None),
    "kv-Q2": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "max_stores": 8}, 2, None),
    "kv-Q3": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "cash_buffer": 3000.0}, 2, None),
    "kv-Q4": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "cash_buffer": 6000.0}, 2, None),
    "kv-Q5": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 2.0}, 2, None),
    # wage-rule sweep around kv-Q5 (wage = wage_spend + k_wage_h * (1 - happiness)); Q5 = k_wage_h 2 was the big squeeze-resistance win
    "kv-R1": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 1.0}, 2, None),
    "kv-R2": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 0.5}, 2, None),
    "kv-R3": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 2.0, "wage_spend": 0.5}, 2, None),
    "kv-R4": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 2.0, "wage_spend": 0.4}, 2, None),
    "kv-R5": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 3.0}, 2, None),
    # hyper-aggressive candidates on top of kv-Q5 (the wage-robust expert): undercut price, steal rivals' regions, more stores
    "kv-G1": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 2.0, "price": 1.7}, 2, None),
    "kv-G2": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 2.0, "price": 1.5}, 2, None),
    "kv-G3": ("steal", {"wage_stores": 0, "k_wage_h": 2.0}, 2, None),
    "kv-G4": ("steal", {"wage_stores": 0, "k_wage_h": 2.0, "price": 1.6}, 2, None),
    "kv-G5": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 2.0, "max_stores": 14, "max_class": 3}, 2, None),
    # region-order sweep on top of the Q5 wage rule (k_wage_h 2): the 4-seat battle showed contest_pen 1.5 skips contested METRO for weak AFFLUENT
    "kv-S0": ("lowest", {"wage_stores": 0, "k_wage_h": 2.0}, 2, None),  # biggest class first, lowest index (no contest penalty)
    "kv-S1": ("lowest", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.3}, 2, None),
    "kv-S2": ("lowest", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.6}, 2, None),
    "kv-S3": ("high", {"wage_stores": 0, "k_wage_h": 2.0}, 2, None),  # METRO descending then AFFLUENT descending, like Orcho v5
    "kv-S4": ("ring_smart", {"wage_stores": 0, "k_wage_h": 2.0}, 2, None),
    # "give up crowded regions" (thr = max rivals tolerated; max_class 3 also allows the lower LOYAL class)
    "kv-T0": ("thresh", {"wage_stores": 0, "k_wage_h": 2.0, "thr": 0}, 2, None),
    "kv-T1": ("thresh", {"wage_stores": 0, "k_wage_h": 2.0, "thr": 1}, 2, None),
    "kv-T2": ("thresh", {"wage_stores": 0, "k_wage_h": 2.0, "thr": 2}, 2, None),
    "kv-T3": ("thresh", {"wage_stores": 0, "k_wage_h": 2.0, "thr": 1, "max_class": 3}, 2, None),
    # combined: soft penalty 0.3 (S1) + give up regions with more than thr rivals
    "kv-U1": ("combo", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.3, "thr": 1}, 2, None),
    "kv-U2": ("combo", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.3, "thr": 2}, 2, None),
    # one-change sweep around kv-S1 (soft penalty 0.3, Q5 wages, frugal 2): expansion, wage base, research, opening
    "kv-X1": ("lowest", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.15}, 2, None),
    "kv-X2": ("lowest", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.2}, 2, None),
    "kv-X3": ("lowest", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.3, "wage_spend": 0.40}, 2, None),
    "kv-X4": ("lowest", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.3, "research": 0.0}, 2, None),
    "kv-X5": ("lowest", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.3, "research": 0.1}, 2, None),
    "kv-X6": ("lowest", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.3, "max_stores": 11}, 2, None),
    "kv-X7": ("lowest", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.3}, 1, None),
    "kv-X8": ("lowest", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.3}, 3, None),
    # levers measured by Yavkata (COUNTER_STRATEGY.md) on an Orcho-copy probe, now on the S1 base: lower service, wage base 0.55, 8-store cap
    "kv-Y1": ("lowest", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.3, "service": 0.7}, 2, None),
    "kv-Y2": ("lowest", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.3, "service": 0.5}, 2, None),
    "kv-Y3": ("lowest", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.3, "wage_spend": 0.55}, 2, None),
    "kv-Y4": ("lowest", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.3, "max_stores": 8}, 2, None),
    # hybrid ingredients: highest-number tie-break (Orcho v5), "biggest region nobody holds first" (Misho's owner rule)
    # hybrid: Q5 base (clonable, pen 1.5) + S1 cheap-wage/research trims; HY2 also takes Orcho's highest-number tie-break
    "kv-HY1": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 2.0, "wage_spend": 0.40, "research": 0.1}, 2, None),
    "kv-HY2": ("softhigh", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 2.0, "wage_spend": 0.40, "research": 0.1}, 2, None),    # Misho's joint-check numbers (service 0.2, research 0.1, cap 10, wage 0.45 + 4 x (1 - happiness)) on the Q5 expansion rule
    "kv-M1": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 4.0, "service": 0.2, "research": 0.1, "max_stores": 10}, 2, None),
    "kv-M2": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 2.0, "service": 0.2, "research": 0.1, "max_stores": 10}, 2, None),
    "kv-M3": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 2.0, "service": 0.5}, 2, None),    "kv-M4": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 2.0, "service": 0.2}, 2, None),
    "kv-M5": ("lowest", {"wage_stores": 0, "contest_pen": 1.5, "k_wage_h": 2.0, "max_stores": 10}, 2, None),    "kv-Z1": ("softhigh", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.3}, 2, None),
    "kv-Z2": ("softhigh", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.2}, 2, None),
    "kv-Z3": ("owner", {"wage_stores": 0, "k_wage_h": 2.0, "contest_pen": 0.3}, 2, None),
    # plan B retreat: cut spend AND stop expanding while cash is low and the last tick lost money
    "kv-N": ("lowest", {"wage_stores": 0, "contest_pen": 1.5}, 2, None, {"cash": 8000.0, "mkt": 0.5, "assort": 0.6, "stop_exp": 1}),
    "kv-O": ("lowest", {"wage_stores": 0, "contest_pen": 1.5}, 2, None, {"cash": 15000.0, "mkt": 0.3, "assort": 0.4, "stop_exp": 1}),
}


def make_expert(variant: str) -> ExpertV8:
    mode, over, ft, fm, *rest = VARIANTS[variant]
    params = {**json.loads(V6.read_text()), **over}
    return ExpertV8(params, mode, ft, fm, rest[0] if rest else None)


def register(variants: tuple[str, ...]) -> None:
    """Make the variants seatable as archetype names (this process only), the same way aggressive_npcs does."""
    from retail_simulator.envs import arena_match

    if variants[0] in arena_match.GEO_ARCHETYPE_LABELS:
        return
    base = arena_match.geo_archetypes

    def build(v: str) -> Any:
        expert = make_expert(v)

        def predict(obs: Any, *, episode_start: bool = False, **_kw: Any) -> Any:
            if episode_start:
                expert.reset()
            return expert(np.asarray(obs, dtype=np.float32).ravel())

        return predict

    def with_v8(n_regions: int, region_table: Any) -> dict[str, Any]:
        return {**base(n_regions, region_table), **{v: build(v) for v in variants}}

    arena_match.geo_archetypes = with_v8
    arena_match.GEO_ARCHETYPE_LABELS = (*arena_match.GEO_ARCHETYPE_LABELS, *variants)


FIELD = {  # the original checkpoints, untouched
    "Orcho": "agents/orcho/market-share/checkpoint.zip",
    "Yavkata-sac": "agents/Yavkata/sac/checkpoint.zip",
    "Krischo-mlp-v6": "agents/Krischo/mlp-v6/checkpoint.zip",
    "Jack": "agents/misho/jack/checkpoint.zip",
    "Jacky-T-v1.1": "runs/train300k/jacky_v11.zip",
}


import os  # noqa: E402

if os.environ.get("KV8_FIELD") == "orchos":  # the harsh field: two strong Orchos (v5 and v4) squeeze the shared regions
    FIELD = {
        "Orcho-v5": "agents/orcho/market-share/checkpoint.zip",
        "Orcho-v4": "runs/v8/orcho_v4.zip",
        "Yavkata-sac": "agents/Yavkata/sac/checkpoint.zip",
        "Jack": "agents/misho/jack/checkpoint.zip",
        "Jacky-T-v1.1": "runs/train300k/jacky_v11.zip",
    }


def play(job: tuple[str, int, int]) -> dict[str, Any]:
    variant, seed, shift = job
    from retail_simulator.envs.arena import arena_home_regions
    from retail_simulator.envs.arena_match import ArenaMatch, ArenaSeatSpec, resolve_arena_world

    register(tuple(VARIANTS))
    seats = [ArenaSeatSpec(name=variant, kind="npc", archetype=variant)]
    seats += [ArenaSeatSpec(name=n, kind="rl", checkpoint=p) for n, p in FIELD.items()]
    cfg, _, _ = resolve_arena_world(balanced=True)
    base = arena_home_regions(6)
    homes = tuple(base[(i + shift) % 6] for i in range(6))
    m = ArenaMatch(cfg, seats, seed=seed, homes=homes)
    tot = np.zeros(6)
    stores100 = 0
    for t in range(1, 1001):
        r = m.step()
        tot += [s.reward for s in r.seats]
        if t == 100:
            stores100 = r.seats[0].stores
    return {"homes": homes, "totals": tot.tolist(), "names": [s.name for s in seats], "stores100": stores100}


def cmd_eval(args: argparse.Namespace) -> None:
    seeds = [int(s) for s in args.seeds.split(",")]
    jobs = [(args.variant, s, sh) for s in seeds for sh in range(6)]
    with Pool(args.workers) as pool:
        res = pool.map(play, jobs, chunksize=1)
    names = res[0]["names"]
    totals = np.array([r["totals"] for r in res])
    ranks = np.argsort(np.argsort(-totals, axis=1), axis=1) + 1
    out = {"variant": args.variant, "seeds": seeds, "matches": len(res), "per_match": res}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{args.variant}: {len(res)} matches (seeds {seeds} x 6 home rotations)")
    print(f"{'seat':16}{'mean':>12}{'worst':>12}{'best':>12}{'wins':>6}{'rank':>7}")
    for k in np.argsort(-totals.mean(axis=0)):
        c = totals[:, k]
        print(f"{names[k]:16}{c.mean():12,.0f}{c.min():12,.0f}{c.max():12,.0f}{int((ranks[:, k] == 1).sum()):6d}{ranks[:, k].mean():7.2f}")
    # per-home view of the expert (seat 0)
    by_home: dict[int, list[float]] = {}
    for r in res:
        by_home.setdefault(r["homes"][0], []).append(r["totals"][0])
    print("expert by start home:", {h: f"{np.mean(v):,.0f}" for h, v in sorted(by_home.items())})
    s100 = [r["stores100"] for r in res]
    print(f"expert stores at round 100: min {min(s100)} mean {np.mean(s100):.1f}; matches stuck at <=3 stores: {sum(s <= 3 for s in s100)}/{len(s100)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--variant", choices=list(VARIANTS), required=True)
    e.add_argument("--seeds", default="5,6")
    e.add_argument("--workers", type=int, default=6)
    e.add_argument("--out", default="runs/v8/eval.json")
    a = ap.parse_args()
    cmd_eval(a)



