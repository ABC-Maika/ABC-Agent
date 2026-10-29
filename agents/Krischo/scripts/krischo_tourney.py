"""Final-style tournament between trained agents (zips): six seats, 1000 rounds, homes and seats rotate, line-ups re-drawn per match.

With seven entrants every match draws six of them, so each agent meets many different line-ups. Ranked by total profit over all matches,
with mean rank, wins and worst match. Own script; nothing in ``src/`` is edited. Run from the repo root:

    .venv\\Scripts\\python.exe agents\\Krischo\\scripts\\krischo_tourney.py --matches 18 --seed0 9300
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "scripts"))

import krischo_league as kl  # noqa: E402
import krischo_order as kord  # noqa: E402

ENTRANTS = {
    "Krischo-mlp-v6": "agents/Krischo/mlp-v6/checkpoint.zip",
    "Krischo-lstm": "agents/Krischo/rppo-lstm/checkpoint.zip",
    "Orcho-v4": "agents/orcho/market-share/checkpoint.zip",
    "Yavkata-rppo": "agents/Yavkata/recurrent-ppo/checkpoint.zip",
    "Yavkata-sac": "agents/Yavkata/sac/checkpoint.zip",
    "Jacky-T": "agents/misho/jacky-t/checkpoint.zip",
    "Jack": "agents/misho/jack/checkpoint.zip",
}


def play(job: tuple[list[str], int]) -> tuple[list[str], list[float]]:
    names, seed = job
    return names, kord.match(([("zip", ENTRANTS[n]) for n in names], seed))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matches", type=int, default=18)
    ap.add_argument("--seed0", type=int, default=9300)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--out", default="runs/alt/tourney.json")
    args = ap.parse_args()
    jobs = []
    for i in range(args.matches):
        seed = args.seed0 + i
        omit = list(ENTRANTS)[i % len(ENTRANTS)]  # each agent sits out equally often, so every agent plays the same number of matches
        names = [n for n in ENTRANTS if n != omit]
        random.Random(seed).shuffle(names)  # seat order; the homes come from the seed
        jobs.append((names, seed))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(play, jobs, chunksize=1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps([{"names": n, "totals": t} for n, t in res]))
    stats = {n: {"profit": [], "rank": [], "wins": 0} for n in ENTRANTS}
    for names, totals in res:
        order = sorted(range(6), key=lambda i: -totals[i])
        for pos, i in enumerate(order):
            s = stats[names[i]]
            s["profit"].append(totals[i])
            s["rank"].append(pos + 1)
            s["wins"] += pos == 0
    print(f"{'agent':<18}{'matches':>8}{'total profit':>14}{'mean':>10}{'worst':>10}{'mean rank':>10}{'wins':>6}")
    for n, s in sorted(stats.items(), key=lambda kv: -sum(kv[1]["profit"])):
        if s["profit"]:
            print(f"{n:<18}{len(s['profit']):>8}{sum(s['profit']):>14,.0f}{np.mean(s['profit']):>10,.0f}{min(s['profit']):>10,.0f}{np.mean(s['rank']):>10.2f}{s['wins']:>6}")


if __name__ == "__main__":
    main()
