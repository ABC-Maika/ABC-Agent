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
}
SHIFTS = (0, 3)
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


def play(job: tuple[dict[str, Any], str, int]) -> float:
    import os
    import sys

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "scripts"))
    params, lineup, shift = job
    from aggressive_npcs import register
    from hunter_bots import persona_predict
    from retail_simulator.envs import arena_match
    from retail_simulator.envs.arena import arena_home_regions
    from retail_simulator.envs.arena_match import ArenaMatch, ArenaSeatSpec, resolve_arena_world

    register()
    base = arena_match.geo_archetypes
    arena_match.geo_archetypes = lambda n, t: {**base(n, t), "cand": persona_predict(params, n)}
    arena_match.GEO_ARCHETYPE_LABELS = (*arena_match.GEO_ARCHETYPE_LABELS, "cand")
    seed, rivals = LINEUPS[lineup]
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
    parser.add_argument("--n-random", type=int, default=30)
    parser.add_argument("--keep", type=int, default=5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/contenders_round1.json"))
    args = parser.parse_args()
    rng = random.Random(args.seed)
    cands = [("v5-recipe", baseline())]
    if args.one_at_a_time:
        cands += [(n, {**baseline(), **d}) for n, d in ONE_AT_A_TIME.items()]
    else:
        cands += [(f"c{i}", random_candidate(rng)) for i in range(args.n_random)]
    keys = [(lu, sh) for lu in LINEUPS for sh in SHIFTS]
    jobs = [(p, lu, sh) for _, p in cands for lu, sh in keys]
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
        per_lineup = {lu: float(np.mean([ratio[j] for j, (l2, _) in enumerate(keys) if l2 == lu])) for lu in LINEUPS}
        rows.append((float(np.mean(ratio)), min(per_lineup.values()), per_lineup, name, p))
    rows.sort(key=lambda r: -(r[0] + r[1]))
    print(f"{'cand':11}{'mean':>8}{'worst':>8}" + "".join(f"{lu:>9}" for lu in LINEUPS) + "   k    price wage cap")
    for mean, worst, pl, name, p in rows:
        print(f"{name:11}{mean:8.1%}{worst:8.1%}" + "".join(f"{pl[lu]:9.1%}" for lu in LINEUPS) + f" {p['k']:5.2f} {p['price']:5.2f} {p['wage']:4.2f} {p['cap']}")
    kept = {name: p for _, _, _, name, p in rows if name != "v5-recipe"}
    args.out.write_text(json.dumps(dict(list(kept.items())[: args.keep]), indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
