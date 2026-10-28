"""Six-seat, no-NPC comparison (the final-tournament format): our agent(s) against five trained agents.

Waits for every zip to exist, then plays each of our agents against the five opponents over several
world seeds and cyclic rotations of the six METRO homes (our seat first), via ``arena_competition.py``.
Writes ``runs/compare6/summary.md`` and prints it. Own script; nothing in ``src/`` is touched.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HOMES = [1, 5, 9, 13, 17, 21]
OUT = Path("runs/compare6")


def play(job: tuple[str, list[tuple[str, str]], int, int]) -> dict:
    tag, players, seed, rot = job
    homes = HOMES[rot:] + HOMES[:rot]
    stem = f"{tag}_s{seed}_r{rot}"
    out_json, out_csv = OUT / f"{stem}.json", OUT / f"{stem}.csv"
    cmd = [sys.executable, "scripts/arena_competition.py", "--balanced", "--rounds", "1000", "--seed", str(seed)]
    for name, zip_path in players:
        cmd += ["--player", f"{name}={zip_path}"]
    cmd += ["--homes", ",".join(map(str, homes)), "--out", str(out_json), "--csv", str(out_csv)]
    subprocess.run(cmd, check=True, capture_output=True)
    return json.loads(out_json.read_text())["standings"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", action="append", required=True, help="NAME=zip (repeatable)")
    parser.add_argument("--opp", action="append", required=True, help="NAME=zip (repeatable, 5 expected)")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--wait-minutes", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    ours = [tuple(s.split("=", 1)) for s in args.ours]
    opps = [tuple(s.split("=", 1)) for s in args.opp]
    seeds = [int(s) for s in args.seeds.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)

    deadline = time.time() + 60 * args.wait_minutes
    needed = [z for _, z in ours + opps]
    while True:
        missing = [z for z in needed if not Path(z).exists()]
        if not missing:
            break
        if time.time() > deadline:
            raise SystemExit(f"timed out waiting for: {missing}")
        time.sleep(20)
    time.sleep(5)  # let the writers finish flushing

    lines = ["# Six-seat comparison (no NPCs, balanced world, 1000 rounds)", ""]
    lines.append(f"Opponents: {', '.join(n for n, _ in opps)}. Seeds {seeds}, {len(HOMES)} cyclic home rotations, our seat first.")
    for name, zip_path in ours:
        jobs = [(name, [(name, zip_path), *opps], seed, rot) for seed in seeds for rot in range(len(HOMES))]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            matches = list(pool.map(play, jobs))
        score, rank, share, stores = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        for st in matches:
            order = sorted(st, key=lambda n: -st[n]["total_score"])
            for r, n in enumerate(order, start=1):
                rank[n].append(r)
            for n, v in st.items():
                score[n].append(v["total_score"])
                share[n].append(v["final_market_share"])
                stores[n].append(v["final_stores"])
        lines += ["", f"## {name} ({len(matches)} matches)", "", "| Agent | Mean score | Min | Max | Mean rank | Wins | Final stores | Final share |", "|---|---|---|---|---|---|---|---|"]
        for n in sorted(score, key=lambda k: -np.mean(score[k])):
            lines.append(
                f"| {n} | {np.mean(score[n]):,.0f} | {np.min(score[n]):,.0f} | {np.max(score[n]):,.0f} | "
                f"{np.mean(rank[n]):.2f} | {sum(r == 1 for r in rank[n])} | {np.mean(stores[n]):.1f} | {np.mean(share[n]):.3f} |"
            )
    text = "\n".join(lines) + "\n"
    (OUT / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
