"""Search OUR side: scripted variants of the v5 recipe scored on four different lineups (team addition).

Each candidate plays seat 0 in four lineups (newest teammates, hunter bots, 5 Krischo copies, the organizer-battle
rivals) x two home rotations. Score = candidate profit / v5-recipe profit on the same game, averaged over the games;
we also report the worst lineup so a gain in one lineup cannot hide a loss in another. The expansion knob ``k`` runs
from aggressive (k < 0: walk into crowded regions) to cautious (k > 0: avoid rivals).

    python scripts/contender_search.py --n-random 30 --out runs/contenders_round1.json
"""

from __future__ import annotations

import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K = "agents/Krischo"
LINEUPS = {
    "newest": (1, [f"{K}/mlp-v6/checkpoint.zip", f"{K}/rppo-lstm/checkpoint_v8.zip", "agents/Yavkata/sac/checkpoint.zip", "agents/misho/jacky-t/checkpoint.zip", f"{K}/rppo-lstm/checkpoint.zip"]),
    "hunters": (5, ["npc:hunter-1", "npc:hunter-2", "npc:hunter-3", "npc:hunter-4", f"{K}/rppo-lstm/checkpoint_v8.zip"]),
    "twins": (1, [f"{K}/mlp-v6/checkpoint.zip"] * 5),
    "battle": (154378, [f"{K}/rppo-lstm/checkpoint_v8.zip", f"{K}/mlp-v6/checkpoint.zip", "agents/misho/jack/checkpoint.zip", "agents/Yavkata/sac/checkpoint.zip", "agents/misho/jacky-t/checkpoint.zip"]),
    "field": (21, ["agents/orcho/v5/checkpoint.zip", f"{K}/mlp-v9/checkpoint.zip", f"{K}/mlp-v8/checkpoint.zip", "agents/Yavkata/sac/checkpoint.zip", "agents/misho/jerky/checkpoint.zip"]),
}
SHIFTS = (0, 3)
LINEUPS_V6 = {  # round 4 (--v6 4): v6's bad start (home 9, shift 2) on a fresh seed, and the hunters searched against v6
    "field9": (41, LINEUPS["field"][1], (2,)),
    "hunters6": (7, ["npc:hunter-5", "npc:hunter-6", "npc:hunter-7", "npc:hunter-8", "agents/orcho/v5/checkpoint.zip"], SHIFTS),
}
SPACE = {"price": (1.8, 2.0), "marketing": (0.8, 1.0), "promotion": (0.7, 1.0), "wage": (0.30, 0.50), "k": (-0.3, 1.5)}


def baseline() -> dict[str, Any]:
    from hunter_bots import DEFAULT

    return {**DEFAULT, "order": "score", "k": 0.0}


def random_candidate(rng: random.Random) -> dict[str, Any]:
    p = {k: round(rng.uniform(*r), 2) for k, r in SPACE.items()}
    p["cap"] = rng.choice([11, 12, 12, 12, 13, 14])
    return {**baseline(), **p}


ONE_AT_A_TIME = {  # one small change from the v5 recipe each, so a gain or loss has a single cause
    "k=-0.2": {"k": -0.2}, "k=0.2": {"k": 0.2}, "k=0.5": {"k": 0.5},
    "price=1.95": {"price": 1.95}, "price=1.9": {"price": 1.9},
    "wage=0.40": {"wage": 0.40}, "wage=0.50": {"wage": 0.50},
    "cap=11": {"cap": 11}, "cap=13": {"cap": 13},
    "mkt=0.95": {"marketing": 0.95}, "promo=0.9": {"promotion": 0.9},
}

R5 = {"order": "score", "k": 0.2, "wage": 0.40}  # orcho-recipe5: the best scripted step after v5
_ALL = {**R5, "exp_cash": 6000.0, "lean": 0.5, "crowd_price": 0.1, "research": 0.0}
V6_IDEAS = {  # situational levers on top of recipe5 (hunter_bots.DEFAULT documents each knob)
    "r5": R5,
    "r5+safe": {**R5, "exp_cash": 6000.0},
    "r5+lean": {**R5, "lean": 0.5},
    "r5+crowd": {**R5, "crowd_price": 0.1},
    "r5+war": {**R5, "war_cash": 30000.0, "war_price": 1.6},
    "r5+r0": {**R5, "research": 0.0},
    "r5+all": _ALL,
    "r5+all+war": {**_ALL, "war_cash": 30000.0, "war_price": 1.6},
    "v5+safe+lean": {"exp_cash": 6000.0, "lean": 0.5},
}
_C = {**R5, "crowd_price": 0.1}
V6_ROUND2 = {  # round 1: crowd and safe won, lean lost (cutting marketing in debt loses share faster than it saves)
    "r5+crowd": _C,
    "crowd+safe": {**_C, "exp_cash": 6000.0},
    "crowd.2": {**R5, "crowd_price": 0.2, "crowd_floor": 1.6},
    "crowd.2+safe": {**R5, "crowd_price": 0.2, "crowd_floor": 1.6, "exp_cash": 6000.0},
}
V6_ROUND3 = {  # round 2: plain crowd won, safe hurt. Teammate's levers: service 0.2 (fill rate 0.6 = the share cap,
    # so no lost sales and ~8% cheaper goods; service brings no customers), research 0.1, cap 10
    "crowd+svc": {**_C, "service": 0.2, "research": 0.1},
    "crowd+svc+cap10": {**_C, "service": 0.2, "research": 0.1, "cap": 10},
    "crowd+svc0": {**_C, "service": 0.2, "research": 0.0},
}
_V6 = {**_C, "service": 0.2, "research": 0.1}  # = hunter_bots.CONTENDERS["v6"]
V6_ROUND4 = {  # v6 lost from home 9: 4 stores in rivals' homes on round 2 with no cash, then kept undercutting in debt
    "v6": _V6,
    "v6+nodebtcut": {**_V6, "crowd_cash": 0.0},
    "v6+hold": {**_V6, "hold_crowded": 5000.0},
    "v6+k.5": {**_V6, "k": 0.5},
    "v6+k1": {**_V6, "k": 1.0},
    "v6+nodebtcut+hold": {**_V6, "crowd_cash": 0.0, "hold_crowded": 5000.0},
    "v6+nodebtcut+k.5": {**_V6, "crowd_cash": 0.0, "k": 0.5},
}


def play(job: tuple[dict[str, Any], tuple[Any, ...], int]) -> float:
    import os
    import sys

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "scripts"))
    params, spec, shift = job
    from aggressive_npcs import register
    from hunter_bots import persona_predict
    from retail_simulator.envs import arena_match
    from retail_simulator.envs.arena import arena_home_regions
    from retail_simulator.envs.arena_match import ArenaMatch, ArenaSeatSpec, resolve_arena_world

    register()
    base = arena_match.geo_archetypes
    arena_match.geo_archetypes = lambda n, t: {**base(n, t), "cand": persona_predict(params, n)}
    arena_match.GEO_ARCHETYPE_LABELS = (*arena_match.GEO_ARCHETYPE_LABELS, "cand")
    seed, rivals = spec[0], spec[1]
    seats = [ArenaSeatSpec(name="cand", kind="npc", archetype="cand")]
    for i, r in enumerate(rivals):
        seats.append(ArenaSeatSpec(name=f"r{i}", kind="npc", archetype=r[4:]) if r.startswith("npc:") else ArenaSeatSpec(name=f"r{i}", kind="rl", checkpoint=r))
    cfg, _, _ = resolve_arena_world(balanced=True)
    hb = arena_home_regions(6)
    m = ArenaMatch(cfg, seats, seed=seed, homes=tuple(hb[(i + shift) % 6] for i in range(6)))
    total = 0.0
    for _ in range(1000):
        total += m.step().seats[0].reward
    return total


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--one-at-a-time", action="store_true", help="single-knob changes from v5 instead of random candidates")
    parser.add_argument("--v6", nargs="?", const="1", help="the situational v6 ideas instead of random candidates: 1 = V6_IDEAS, 2 = V6_ROUND2, 3 = V6_ROUND3, 4 = V6_ROUND4 (+ LINEUPS_V6)")
    parser.add_argument("--n-random", type=int, default=30)
    parser.add_argument("--keep", type=int, default=5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/contenders_round1.json"))
    args = parser.parse_args()
    rng = random.Random(args.seed)
    cands = [("v5-recipe", baseline())]
    lineups = dict(LINEUPS)
    if args.v6:
        rounds = {"1": V6_IDEAS, "2": V6_ROUND2, "3": V6_ROUND3, "4": V6_ROUND4}
        cands += [(n, {**baseline(), **d}) for n, d in rounds[args.v6].items()]
        if args.v6 == "4":
            lineups.update(LINEUPS_V6)
    elif args.one_at_a_time:
        cands += [(n, {**baseline(), **d}) for n, d in ONE_AT_A_TIME.items()]
    else:
        cands += [(f"c{i}", random_candidate(rng)) for i in range(args.n_random)]
    keys = [(lu, sh) for lu, spec in lineups.items() for sh in (spec[2] if len(spec) > 2 else SHIFTS)]
    jobs = [(p, lineups[lu], sh) for _, p in cands for lu, sh in keys]
    out = []
    with Pool(args.workers, maxtasksperchild=3) as pool:  # fresh workers: a stuck one cannot hang the whole search
        for i, v in enumerate(pool.imap(play, jobs, chunksize=1), 1):
            out.append(v)
            print(f"[{i}/{len(jobs)}] games done", flush=True)
    res = np.array(out).reshape(len(cands), len(keys))
    base = res[0]
    rows = []
    for i, (name, p) in enumerate(cands):
        ratio = (res[i] - base) / np.maximum(np.abs(base), 50_000.0)  # change vs the v5 recipe in the same game
        per_lineup = {lu: float(np.mean([ratio[j] for j, (l2, _) in enumerate(keys) if l2 == lu])) for lu in lineups}
        rows.append((float(np.mean(ratio)), min(per_lineup.values()), per_lineup, name, p))
    rows.sort(key=lambda r: -(r[0] + r[1]))
    print(f"{'cand':18}{'mean':>8}{'worst':>8}" + "".join(f"{lu:>9}" for lu in lineups) + "   profit/game  worst game")
    for mean, worst, pl, name, p in rows:
        i = [n for n, _ in cands].index(name)
        print(f"{name:18}{mean:8.1%}{worst:8.1%}" + "".join(f"{pl[lu]:9.1%}" for lu in lineups) + f"  {res[i].mean():11,.0f} {res[i].min():11,.0f}")
    kept = {name: p for _, _, _, name, p in rows if name != "v5-recipe"}
    args.out.write_text(json.dumps(dict(list(kept.items())[: args.keep]), indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
