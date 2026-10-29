"""3 copies of orcho vs 3 copies of krischo (and 6 copies of orcho): who wins, by seat, by home region, by identity.

Reproduces the teammate-found effect: with several copies of one agent in a match, the winner follows the
HOME REGION, not the seat or the agent. Run from the repo root: python scripts/arena_copies_repro.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np

O = "agents/orcho/market-share/checkpoint.zip"
K = "agents/Krischo/rppo-lstm/checkpoint.zip"
ORDERS = {
    "O O O K K K": [("orcho", O), ("orcho1", O), ("orcho2", O), ("krischo", K), ("krischo1", K), ("krischo2", K)],
    "K K K O O O": [("krischo", K), ("krischo1", K), ("krischo2", K), ("orcho", O), ("orcho1", O), ("orcho2", O)],
    "O K O K O K": [("orcho", O), ("krischo", K), ("orcho1", O), ("krischo1", K), ("orcho2", O), ("krischo2", K)],
    "O O O O O O": [(f"orcho{i}", O) for i in range(6)],
}


def play(job):
    import os
    os.chdir(ROOT)
    order, shift = job
    from retail_simulator.envs.arena import arena_home_regions
    from retail_simulator.envs.arena_match import ArenaMatch, ArenaSeatSpec, resolve_arena_world

    seats = [ArenaSeatSpec(name=n, kind="rl", checkpoint=p) for n, p in ORDERS[order]]
    cfg, _, _ = resolve_arena_world(balanced=True)
    hb = arena_home_regions(6)
    homes = tuple(hb[(i + shift) % 6] for i in range(6))
    m = ArenaMatch(cfg, seats, seed=0, homes=homes)
    tot, stores10 = np.zeros(6), None
    for t in range(1000):
        r = m.step()
        tot += [s.reward for s in r.seats]
        if t == 20:
            stores10 = [s.stores for s in r.seats]
    return order, shift, homes, tot, stores10


if __name__ == "__main__":
    jobs = [(o, s) for o in ORDERS for s in range(6)]
    with Pool(6) as pool:
        res = pool.map(play, jobs, chunksize=1)
    for order in ORDERS:
        names = [n for n, _ in ORDERS[order]]
        print(f"\n=== seat order {order}   (names: {names})")
        print(f"{'rot':>3} {'homes':<22} {'winner':<10} {'winner seat':>11} {'winner home':>11}   scores by seat")
        wins_seat = np.zeros(6, int)
        for o, s, homes, tot, st in [x for x in res if x[0] == order]:
            w = int(np.argmax(tot))
            wins_seat[w] += 1
            print(f"{s:>3} {str(homes):<22} {names[w]:<10} {w:>11} {homes[w]:>11}   {[round(x / 1000) for x in tot]}k  stores@t20 {st}")
        print("wins by SEAT index:", wins_seat.tolist())
