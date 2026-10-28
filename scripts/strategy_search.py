"""Search scripted (rule-based) strategies against strong lineups in the balanced arena.

Each strategy is a dict of levers + adaptive rules (see ``DEFAULT`` and ``_make``). A job is
(strategy, lineup, home rotation); jobs run in a process pool and append one JSON line each to
``runs/strategy/<tag>.jsonl``. ``summary`` aggregates a jsonl into a ranked table.

    python scripts/strategy_search.py run  <tag> <strat,strat,...> <lineup,lineup,...> [rotations] [workers]
    python scripts/strategy_search.py summary <tag>
"""

from __future__ import annotations

from collections.abc import Callable
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

OUT = Path("runs/strategy")
S = "runs/serious"
RL = {
    "krischo": "agents/Krischo/rppo-lstm/checkpoint.zip",
    "yavkata": "agents/Yavkata/recurrent-ppo/checkpoint.zip",
    "misho-jack": "agents/misho/jack/checkpoint.zip",
    "rival-a2c-s2": f"{S}/rival-a2c-s2.zip",
    "rival-a2c-s1": f"{S}/rival-a2c-s1.zip",
}

# orcho-recipe2 exactly (scripts/aggressive_npcs.py), expressed in this file's knobs.
DEFAULT: dict[str, Any] = {
    "price": 2.0, "marketing": 1.0, "assortment": 1.0, "promotion": 1.0, "service": 1.0,
    "research": 0.2, "loyalty": 0.0, "automation": 0, "warehouse": 0.0,
    "wage": 0.45,          # base wage_spend
    "wage_ps": 0.0,        # + wage_ps per store above wage_ps_from
    "wage_ps_from": 12,
    "wage_fb": 0.0,        # + wage_fb * max(0, h_target - happiness)  (feedback)
    "h_target": 0.97,
    "wage_max": 1.0,
    "wage_int": 0.0,       # integral gain: wint += wage_int * (h_target - happiness) when below target
    "wage_down": 0.0,      # per-tick downward drift of wint while happy
    "scale_cash": None,    # (lo, hi, min_frac): spend levers x clip((cash-lo)/(hi-lo), min_frac, 1)
    "neg_profit_scale": 1.0,  # extra spend multiplier when last_profit < 0
    "lean_cash": None,     # (cash_threshold, marketing, assortment): cut spend when cash below thr
    "cap": 12,             # store cap
    "expand_until": 10**9, # no expansion after this tick
    "cash_min": -1e18,     # only expand when cash >= cash_min
    "h_min": 0.0,          # only expand when happiness >= h_min
    "order": "demand",     # demand | obsdemand | steal | quiet
    "price_early": None,   # (price, until_tick)
    "mkt_late": None,      # (marketing, from_tick)
    "drop_cap_if_cash": None,  # (cash_threshold, cap) -> stop at lower cap while cash below thr
}

R2 = {}  # orcho-recipe2
# cash-adaptive: wage 0.45 + 4(1-h), research 0.3, spend levers scaled down as cash runs low
CA = {"wage_fb": 4.0, "h_target": 1.0, "research": 0.3,
      "scale_cash": (-3000.0, 2000.0, 0.3), "neg_profit_scale": 0.8}
STRATS: dict[str, dict[str, Any]] = {
    "recipe2": R2,
    # --- store cap / pace
    "cap11": {"cap": 11},
    "cap13": {"cap": 13},
    # --- wage
    "wage.6": {"wage": 0.6},
    "wage.8": {"wage": 0.8},
    "wage1.0": {"wage": 1.0},
    "wagefb": {"wage": 0.45, "wage_fb": 3.0},
    "wagefb.6": {"wage": 0.6, "wage_fb": 3.0},
    # --- region choice
    "obsdemand": {"order": "obsdemand"},
    "steal": {"order": "steal"},
    "obsdemand-share": {"order": "obsdemand_share"},
    # --- time-varying
    "cheap-early": {"price_early": (1.5, 100)},
    "mkt-late.6": {"mkt_late": (0.6, 300)},
    # --- bigger footprint with wage ramp
    "cap14-ramp": {"cap": 14, "wage_ps": 0.08},
    "cap16-ramp": {"cap": 16, "wage_ps": 0.08, "wage_fb": 3.0},
    # --- adaptive expansion
    "adapt-cap16": {"cap": 16, "cash_min": 5000.0, "h_min": 0.9, "wage_fb": 3.0},
    # --- round 2: robust to copies (no debt spiral, wage that tracks the required wage)
    "safe": {"cash_min": 6000.0},
    "wint": {"wage_int": 2.0, "wage_down": 0.0005, "h_target": 0.995},
    "safe-wint": {"cash_min": 6000.0, "wage_int": 2.0, "wage_down": 0.0005, "h_target": 0.995},
    "safe-w1": {"cash_min": 6000.0, "wage": 1.0},
    "safe-wint-od": {"cash_min": 6000.0, "wage_int": 2.0, "wage_down": 0.0005, "h_target": 0.995,
                     "order": "obsdemand"},
    "crowd": {"order": "crowd"},
    "crowd-wint": {"order": "crowd", "wage_int": 2.0, "wage_down": 0.0005, "h_target": 0.995},
    "od-wint": {"order": "obsdemand", "wage_int": 2.0, "wage_down": 0.0005, "h_target": 0.995},
    "crowd-w1": {"order": "crowd", "wage": 1.0},
    # --- coordinator ideas: cash-adaptive levers (1), class value / (1 + rivals) (2), both (3)
    "ca-b2000": {**CA, "cash_min": 7000.0},
    "ca-b0": {**CA, "cash_min": 5000.0},
    "ca-b2000-soft": {**CA, "cash_min": 7000.0, "scale_cash": (-4000.0, 0.0, 0.5), "neg_profit_scale": 0.9},
    "classval": {"order": "classval"},
    "cv-ca-b2000": {**CA, "cash_min": 7000.0, "order": "classval"},
    "cv-ca-b0": {**CA, "cash_min": 5000.0, "order": "classval"},
    "cv-fb": {"order": "classval", "wage_fb": 4.0, "h_target": 1.0, "research": 0.3},
}

# --- coordinator: in-between lever values + seeded random jitter (+-0.1) around recipe2
STRATS.update({
    "p1.92": {"price": 1.92}, "p1.95": {"price": 1.95}, "p1.85": {"price": 1.85},
    "inbetween": {"marketing": 0.93, "promotion": 0.85, "assortment": 0.9, "service": 0.9, "research": 0.15},
})
_rng = np.random.default_rng(7)
for _j in range(8):
    _d = {k: float(np.clip(DEFAULT[k] + _rng.uniform(-0.1, 0.1), lo, hi)) for k, lo, hi in (
        ("price", 0.5, 2.0), ("marketing", 0, 1), ("promotion", 0, 1), ("assortment", 0, 1),
        ("service", 0, 1), ("research", 0, 1), ("wage", 0, 1))}
    STRATS[f"jit{_j}"] = {k: round(v, 3) for k, v in _d.items()}

# jitters of the combined cash-adaptive + class-value strategy
_COMB = {**CA, "cash_min": 7000.0, "order": "classval"}
_rng2 = np.random.default_rng(11)
for _j in range(4):
    _d = {k: float(np.clip(DEFAULT[k] + _rng2.uniform(-0.1, 0.1), lo, hi)) for k, lo, hi in (
        ("price", 0.5, 2.0), ("marketing", 0, 1), ("promotion", 0, 1), ("assortment", 0, 1),
        ("service", 0, 1))}
    _d["research"] = float(np.clip(0.3 + _rng2.uniform(-0.1, 0.1), 0, 1))
    STRATS[f"cvj{_j}"] = {**_COMB, **{k: round(v, 3) for k, v in _d.items()}}

LINEUPS: dict[str, list[tuple[str, str]]] = {
    # (kind, what): ("rl", checkpoint-key) | ("r2", "") recipe2 copy | ("self", "") candidate copy
    "team": [("rl", k) for k in RL],
    "r2x2": [("r2", ""), ("r2", ""), ("rl", "krischo"), ("rl", "yavkata"), ("rl", "misho-jack")],
    "r2x3": [("r2", ""), ("r2", ""), ("r2", ""), ("rl", "krischo"), ("rl", "misho-jack")],
    "self3": [("self", ""), ("self", ""), ("r2", ""), ("rl", "krischo"), ("rl", "yavkata")],
}

CLASS_RANK = {0: 0, 1: 1, 2: 2, 3: 3}  # (r-1)%4: METRO, AFFLUENT, LOYAL, FRONTIER


def _make(params: dict[str, Any], n_regions: int, idx: dict[str, int]) -> Callable[..., Any]:
    from retail_simulator.core.encoding import encode_action
    from retail_simulator.envs import geo_archetypes as geo

    p = {**DEFAULT, **params}
    tick = [0]
    wint = [0.0]

    def pick(obs: Any) -> int:
        empty = list(geo._own_empty_regions(obs, idx=idx, candidates=range(1, n_regions)))
        if not empty:
            return 0
        order = p["order"]
        if order == "demand":
            return min(empty, key=lambda r: ((r - 1) % 4, r))
        if order == "quiet":
            return min(empty, key=lambda r: (float(obs[idx[f"others_present_region_{r}"]]) > 0, (r - 1) % 4, r))
        if order == "crowd":  # class order, but skip regions with >= 2 other retailers present
            return min(empty, key=lambda r: (float(obs[idx[f"others_present_region_{r}"]]) >= 2, (r - 1) % 4, r))
        if order == "classval":  # class demand split among the rivals already present
            val = {0: 1400.0, 1: 900.0, 2: 600.0, 3: 350.0}
            return max(empty, key=lambda r: (val[(r - 1) % 4]
                                             / (1.0 + float(obs[idx[f"others_present_region_{r}"]])), -r))
        if order == "steal":  # prefer contested regions, then class
            return min(empty, key=lambda r: (-float(obs[idx[f"others_present_region_{r}"]]), (r - 1) % 4, r))
        if order == "obsdemand":  # observed regional demand split among present others
            return max(empty, key=lambda r: (float(obs[idx[f"total_regional_demand_region_{r}"]])
                                             / (1.0 + float(obs[idx[f"others_present_region_{r}"]])), -r))
        if order == "obsdemand_share":  # observed demand x what rivals leave (1 - their share)
            return max(empty, key=lambda r: (float(obs[idx[f"total_regional_demand_region_{r}"]])
                                             * (1.0 - min(0.6, float(obs[idx[f"competitor_market_share_region_{r}"]]))), -r))
        raise ValueError(order)

    def predict(obs: Any, *, episode_start: bool = False, **_kw: Any) -> Any:
        if episode_start:
            tick[0] = 0
            wint[0] = 0.0
        t = tick[0]
        tick[0] += 1
        stores = float(obs[idx["stores"]])
        happy = float(obs[idx["employee_happiness"]])
        cash = float(obs[idx["cash"]])
        cap = p["cap"]
        if p["drop_cap_if_cash"] is not None and cash < p["drop_cap_if_cash"][0]:
            cap = min(cap, p["drop_cap_if_cash"][1])
        target = 0
        if stores < cap and t <= p["expand_until"] and cash >= p["cash_min"] and happy >= p["h_min"]:
            target = pick(obs)
        if p["wage_int"]:  # integral controller: fast up when unhappy, slow drift down when happy
            step = p["wage_int"] * (p["h_target"] - happy) if happy < p["h_target"] else -p["wage_down"]
            wint[0] = min(max(wint[0] + step, -p["wage"]), p["wage_max"])
        wage = p["wage"] + wint[0] + p["wage_ps"] * max(0.0, stores - p["wage_ps_from"]) \
            + p["wage_fb"] * max(0.0, p["h_target"] - happy)
        wage = min(wage, p["wage_max"])
        price = p["price"]
        if p["price_early"] is not None and t < p["price_early"][1]:
            price = p["price_early"][0]
        mkt = p["marketing"]
        if p["mkt_late"] is not None and t >= p["mkt_late"][1]:
            mkt = p["mkt_late"][0]
        asrt = p["assortment"]
        if p["lean_cash"] is not None and cash < p["lean_cash"][0]:
            mkt, asrt = min(mkt, p["lean_cash"][1]), min(asrt, p["lean_cash"][2])
        promo, res = p["promotion"], p["research"]
        if p["scale_cash"] is not None:
            lo, hi, fmin = p["scale_cash"]
            f = min(max((cash - lo) / (hi - lo), fmin), 1.0)
            if float(obs[idx["last_profit"]]) < 0:
                f = max(f * p["neg_profit_scale"], fmin)
            mkt, asrt, promo, res = mkt * f, asrt * f, promo * f, res * f
        return encode_action(geo._structured_action(
            price=price, marketing=mkt, assortment=asrt, promotion=promo,
            expansion_choice=target, research=res, service_level=p["service"],
            automation_tier=p["automation"], loyalty_spend=p["loyalty"], n_regions=n_regions,
            wage_spend=wage, warehouse_invest=p["warehouse"],
        ))

    return predict


def _job(args: tuple[str, str, int, str, str]) -> dict[str, Any]:
    strat, lineup, rot, tag, strat_json = args
    import torch
    torch.set_num_threads(1)
    from retail_simulator.envs import arena_match
    from retail_simulator.envs import geo_archetypes as geo
    from retail_simulator.envs.arena import arena_home_regions
    from retail_simulator.envs.arena_match import ArenaMatch, ArenaSeatSpec, resolve_arena_world

    cand = json.loads(strat_json)
    npc_params: dict[str, dict[str, Any]] = {"cand": cand}
    seats = [ArenaSeatSpec(name="cand", kind="npc", archetype="cand")]
    names = ["cand"]
    for i, (kind, what) in enumerate(LINEUPS[lineup]):
        if kind == "rl":
            seats.append(ArenaSeatSpec(name=what, kind="rl", checkpoint=RL[what]))
            names.append(what)
        else:  # each copy gets its own label so closures (tick state) are not shared
            label = f"{'r2' if kind == 'r2' else 'self'}#{i}"
            npc_params[label] = R2 if kind == "r2" else cand
            seats.append(ArenaSeatSpec(name=label, kind="npc", archetype=label))
            names.append(label)
    assert len(seats) == 6

    base = arena_match.geo_archetypes

    def roster(n_regions: int, table: Any) -> dict[str, Callable[..., Any]]:
        idx = geo._name_to_index(n_regions)
        return {**base(n_regions, table), **{k: _make(v, n_regions, idx) for k, v in npc_params.items()}}

    arena_match.geo_archetypes = roster
    arena_match.GEO_ARCHETYPE_LABELS = (*arena_match.GEO_ARCHETYPE_LABELS, *npc_params)
    cfg, _, _ = resolve_arena_world(balanced=True)
    hb = arena_home_regions(6)
    homes = tuple(hb[(i + rot) % 6] for i in range(6))
    t0 = time.time()
    m = ArenaMatch(cfg, seats, seed=rot, homes=homes)
    tot = np.zeros(6)
    share = np.zeros(6)
    happy: list[float] = []
    last = None
    for k in range(1000):
        last = m.step()
        tot += [s.reward for s in last.seats]
        if k >= 500:
            share += [s.market_share for s in last.seats]
        happy.append(last.seats[0].employee_happiness)
    share /= 500
    order = np.argsort(-tot).tolist()
    rec = {
        "tag": tag, "strat": strat, "lineup": lineup, "rot": rot,
        "names": names, "scores": tot.tolist(), "share": share.tolist(),
        "rank": order.index(0) + 1, "share_rank": int(np.argsort(-share).tolist().index(0)) + 1,
        "happy": float(np.mean(happy[100:])), "stores": last.seats[0].stores,
        "secs": round(time.time() - t0, 1),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{tag}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def _safe_job(args: tuple[str, str, int, str, str]) -> dict[str, Any] | str:
    try:
        return _job(args)
    except Exception as e:  # keep the pool alive; report and move on
        return f"FAILED {args[:3]}: {e!r}"


def run(tag: str, strats: list[str], lineups: list[str], rotations: list[int], workers: int) -> None:
    done = set()
    if (OUT / f"{tag}.jsonl").exists():  # resume: skip finished jobs
        done = {(r["strat"], r["lineup"], r["rot"]) for r in map(json.loads, open(OUT / f"{tag}.jsonl"))}
    jobs = [(s, lu, r, tag, json.dumps(STRATS[s])) for s in strats for lu in lineups for r in rotations
            if (s, lu, r) not in done]
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(_safe_job, jobs):
            if isinstance(rec, str):
                print(rec, flush=True)
            else:
                print(f"{rec['strat']:18}{rec['lineup']:7} rot{rec['rot']} score {rec['scores'][0]:>11,.0f} "
                      f"rank {rec['rank']} share {rec['share'][0]:.3f} happy {rec['happy']:.2f} "
                      f"stores {rec['stores']} ({rec['secs']}s)", flush=True)
    summary(tag)


def summary(tag: str) -> None:
    recs = [json.loads(line) for line in open(OUT / f"{tag}.jsonl")]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in recs:
        groups.setdefault((r["lineup"], r["strat"]), []).append(r)
    for lineup in sorted({k[0] for k in groups}):
        rows = [(s, g) for (lu, s), g in groups.items() if lu == lineup]
        rows.sort(key=lambda x: -np.mean([r["scores"][0] for r in x[1]]))
        print(f"\n== lineup {lineup}")
        print(f"{'strategy':18}{'n':>3}{'mean':>11}{'min':>11}{'wins':>6}{'rank':>6}{'share':>7}"
              f"{'shRk':>6}{'happy':>6}{'best opponent (mean)':>28}")
        for s, g in rows:
            sc = [r["scores"][0] for r in g]
            opp: dict[str, list[float]] = {}
            for r in g:
                for n, v in zip(r["names"][1:], r["scores"][1:]):
                    opp.setdefault(n.split("#")[0], []).append(v)
            best = max(opp, key=lambda n: np.mean(opp[n]))
            print(f"{s:18}{len(g):3}{np.mean(sc):11,.0f}{min(sc):11,.0f}"
                  f"{sum(r['rank'] == 1 for r in g):4}/{len(g)}{np.mean([r['rank'] for r in g]):6.2f}"
                  f"{np.mean([r['share'][0] for r in g]):7.3f}{np.mean([r['share_rank'] for r in g]):6.2f}"
                  f"{np.mean([r['happy'] for r in g]):6.2f}   {best} {np.mean(opp[best]):,.0f}")


def main() -> int:
    os.chdir(Path(__file__).resolve().parents[1])
    if sys.argv[1] == "summary":
        summary(sys.argv[2])
        return 0
    _, _, tag, strats, lineups, *rest = sys.argv
    rots = [int(x) for x in rest[0].split(",")] if rest else [0, 3]
    workers = int(rest[1]) if len(rest) > 1 else 7
    run(tag, strats.split(","), lineups.split(","), rots, workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
