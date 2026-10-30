"""Search for scripted bots that beat our agent, and store the best as sparring partners.

Each candidate plays seat 0 against our checkpoint plus four fixed rivals (Krischo, Yavkata, cherry-picker+,
fast-follower+); fitness is (candidate score - our score), averaged over two home rotations. The best distinct
candidates are written to ``scripts/hunters.json`` and become the NPCs ``hunter-1`` ... ``hunter-K``.

    python scripts/hunter_search.py --n-random 30 --keep 4
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
US = "agents/orcho/market-share/checkpoint.zip"  # overridden by --us
SHIFTS = (0, 3, 5)
FIXED = {
    "krischo": "agents/Krischo/mlp-v6/checkpoint.zip",
    "twin": "agents/Krischo/rppo-lstm/checkpoint.zip",
}
SPACE = {
    "price": (0.9, 2.0), "marketing": (0.4, 1.0), "assortment": (0.4, 1.0), "promotion": (0.0, 1.0),
    "service": (0.2, 1.0), "research": (0.0, 0.5), "wage": (0.3, 0.7), "early_price": (0.0, 1.5),
    "crowd_price": (0.0, 0.3), "k": (-0.5, 1.5),  # v6-style situational undercut; region scoring for order "score"
}


def random_candidate(rng: random.Random) -> dict[str, Any]:
    from hunter_bots import DEFAULT, ORDERS

    p = {k: round(rng.uniform(*r), 2) for k, r in SPACE.items()}
    p["switch_tick"] = rng.choice([0, 0, 50, 100, 200])
    return {**DEFAULT, **p, "smart_pay": rng.choice([0, 1, 1]), "cap": rng.randint(5, 18), "order": rng.choice(ORDERS)}


def mutate(rng: random.Random, parent: dict[str, Any]) -> dict[str, Any]:
    from hunter_bots import ORDERS

    q = dict(parent)
    for k, (lo, hi) in SPACE.items():
        if k == "early_price" and q.get("early_price", 0) == 0:
            continue
        if rng.random() < 0.5:
            q[k] = round(min(hi, max(lo, q[k] + rng.gauss(0, 0.12 * (hi - lo)))), 2)
    if rng.random() < 0.5:
        q["cap"] = max(4, min(18, q["cap"] + rng.choice([-3, -2, -1, 1, 2, 3])))
    if rng.random() < 0.2:
        q["order"] = rng.choice(ORDERS)
    return q


def play(job: tuple[dict[str, Any], int, str]) -> tuple[float, float, int]:
    import os
    import sys

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "scripts"))
    params, shift, us_zip = job
    from aggressive_npcs import register
    from hunter_bots import persona_predict
    from retail_simulator.envs import arena_match
    from retail_simulator.envs.arena import arena_home_regions
    from retail_simulator.envs.arena_match import ArenaMatch, ArenaSeatSpec, resolve_arena_world

    register()
    base = arena_match.geo_archetypes
    arena_match.geo_archetypes = lambda n, t: {**base(n, t), "cand": persona_predict(params, n)}
    arena_match.GEO_ARCHETYPE_LABELS = (*arena_match.GEO_ARCHETYPE_LABELS, "cand")
    seats = [ArenaSeatSpec(name="cand", kind="npc", archetype="cand"), ArenaSeatSpec(name="us", kind="rl", checkpoint=us_zip)]
    seats += [ArenaSeatSpec(name=n, kind="rl", checkpoint=p) for n, p in FIXED.items()]
    seats += [ArenaSeatSpec(name=n, kind="npc", archetype=n) for n in ("cherry-picker+", "fast-follower+")]
    cfg, _, _ = resolve_arena_world(balanced=True)
    hb = arena_home_regions(6)
    m = ArenaMatch(cfg, seats, seed=0, homes=tuple(hb[(i + shift) % 6] for i in range(6)))
    tot = np.zeros(6)
    for _ in range(1000):
        tot += [s.reward for s in m.step().seats]
    return float(tot[0]), float(tot[1]), int(np.argsort(-tot).tolist().index(1)) + 1


def main() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from hunter_bots import HUNTERS_FILE, PERSONAS

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-random", type=int, default=30)
    parser.add_argument("--keep", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--us", default=US, help="the checkpoint the hunters are searched against")
    parser.add_argument("--out", type=Path, default=HUNTERS_FILE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evolve", type=int, default=0, help="mutations per parent of the current hunters.json (round 2)")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    cands = [(name, p) for name, p in PERSONAS.items()] + [(f"rand-{i}", random_candidate(rng)) for i in range(args.n_random)]
    if args.evolve:
        from hunter_bots import load_hunters

        parents = list(load_hunters().items())[:3]
        cands = list(parents) + [(f"{n}-m{j}", mutate(rng, p)) for n, p in parents for j in range(args.evolve)]
    jobs = [(p, s, args.us) for _, p in cands for s in SHIFTS]
    with Pool(args.workers) as pool:
        res = pool.map(play, jobs, chunksize=1)
    rows = []
    for i, (name, p) in enumerate(cands):
        r = res[len(SHIFTS) * i : len(SHIFTS) * (i + 1)]
        rows.append((np.mean([x[0] - x[1] for x in r]), np.mean([x[0] for x in r]), np.mean([x[1] for x in r]), np.mean([x[2] for x in r]), name, p))
    rows.sort(key=lambda r: -r[0])
    print(f"{'candidate':14}{'order':10}{'price':>6}{'cap':>4}{'bot':>11}{'us':>11}{'gap':>11}{'our rank':>9}")
    for gap, bot, us, rk, name, p in rows:
        print(f"{name:14}{p['order']:10}{p['price']:6.2f}{p['cap']:4d}{bot:11,.0f}{us:11,.0f}{gap:11,.0f}{rk:9.1f}")
    kept, seen = {}, set()
    for gap, bot, us, rk, name, p in rows:  # best distinct styles first
        key = (p["order"], round(p["price"], 1), p["cap"] // 4)
        if key in seen:
            continue
        seen.add(key)
        kept[f"hunter-{len(kept) + 1}"] = p
        if len(kept) == args.keep:
            break
    args.out.write_text(json.dumps(kept, indent=2) + "\n")
    print(f"\nwrote {args.out.name}: {list(kept)}")


if __name__ == "__main__":
    main()
