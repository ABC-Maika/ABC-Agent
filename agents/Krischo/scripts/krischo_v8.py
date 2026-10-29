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
    def __init__(self, params: dict[str, float], mode: str, frugal_ticks: int = 0, frugal_mkt: float | None = None) -> None:
        super().__init__(params)
        self.mode = mode  # "lowest" | "ring" | "ring_smart"
        self.frugal_ticks = frugal_ticks  # opening rounds with no research/wage (and optionally less marketing) to clear the 5,000 cash gate
        self.frugal_mkt = frugal_mkt

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.tick >= self.frugal_ticks:
            return super().__call__(obs)
        saved = dict(self.p)
        self.p.update(research=0.0, wage_spend=0.0, k_wage_h=0.0, k_wage=0.0)
        if self.frugal_mkt is not None:
            self.p.update(marketing=self.frugal_mkt)
        try:
            return super().__call__(obs)
        finally:
            self.p = saved

    def _expansion(self, obs: np.ndarray, tick: int) -> int:
        if self.mode == "lowest":
            return super()._expansion(obs, tick)
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
}


def make_expert(variant: str) -> ExpertV8:
    mode, over, ft, fm = VARIANTS[variant]
    params = {**json.loads(V6.read_text()), **over}
    return ExpertV8(params, mode, ft, fm)


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
