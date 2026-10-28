"""Play the same six seats over several seeds and rotated home regions, and summarise per seat.

One match proves little: starting regions move scores more than skill does at low training. This
runs every cyclic rotation of the homes for each seed (6 seats x 2 seeds = 12 matches) and reports
mean / min / max total score, wins and mean rank per seat.

    python scripts/arena_rotate.py --player me=runs/mkt/orcho-a2c.zip --player rival=<zip> ... \
        --npc cherry-picker+ --seeds 0,1 --out runs/rotate.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from aggressive_npcs import register

from retail_simulator.envs.arena import arena_home_regions
from retail_simulator.envs.arena_match import ArenaMatch, ArenaSeatSpec, resolve_arena_world

SEATS = 6  # team rule: every simulation seats exactly six agents


def main() -> int:
    npc_labels = register()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--player", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--npc", action="append", default=[], choices=list(npc_labels))
    parser.add_argument("--out", type=Path, default=Path("runs/rotate.json"))
    args = parser.parse_args()

    seats = []
    for spec in args.player:
        label, _, path = spec.partition("=")
        seats.append(ArenaSeatSpec(name=label, kind="rl", checkpoint=path))
    seats += [ArenaSeatSpec(name=n, kind="npc", archetype=n) for n in args.npc]
    if len(seats) != SEATS:
        parser.error(f"every simulation runs with exactly {SEATS} seats, got {len(seats)}")

    cfg, _, _ = resolve_arena_world(balanced=True)
    base = arena_home_regions(SEATS)
    names = [s.name for s in seats]
    scores: dict[str, list[float]] = {n: [] for n in names}
    ranks: dict[str, list[int]] = {n: [] for n in names}
    wins = dict.fromkeys(names, 0)
    matches = 0
    for seed in (int(x) for x in args.seeds.split(",")):
        for shift in range(SEATS):
            homes = tuple(base[(i + shift) % SEATS] for i in range(SEATS))
            match = ArenaMatch(cfg, seats, seed=seed, homes=homes)
            total = np.zeros(SEATS)
            for _ in range(args.rounds):
                total += [s.reward for s in match.step().seats]
            order = np.argsort(-total)
            for rank, k in enumerate(order, start=1):
                ranks[names[k]].append(rank)
            wins[names[order[0]]] += 1
            for k, n in enumerate(names):
                scores[n].append(float(total[k]))
            matches += 1
            print(f"match {matches} seed={seed} homes={homes} winner={names[order[0]]}", flush=True)

    summary = {
        n: {
            "mean_score": float(np.mean(scores[n])),
            "min_score": float(np.min(scores[n])),
            "max_score": float(np.max(scores[n])),
            "wins": wins[n],
            "mean_rank": float(np.mean(ranks[n])),
        }
        for n in names
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"matches": matches, "summary": summary, "scores": scores}, indent=2))
    print(f"\n{matches} matches ({args.rounds} rounds each)")
    print(f"{'seat':16}{'mean':>12}{'min':>12}{'max':>12}{'wins':>6}{'rank':>7}")
    for n in sorted(names, key=lambda n: -summary[n]["mean_score"]):
        s = summary[n]
        print(
            f"{n:16}{s['mean_score']:12,.0f}{s['min_score']:12,.0f}{s['max_score']:12,.0f}"
            f"{s['wins']:6d}{s['mean_rank']:7.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
