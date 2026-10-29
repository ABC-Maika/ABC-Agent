"""Same as arena_rotate.py (6 home rotations) but plays the 6 matches in parallel (about 6x faster).

    SEED=1 python scripts/arena_rotate_parallel.py me=agents/orcho/market-share/checkpoint.zip         krischo=agents/Krischo/rppo-lstm/checkpoint.zip a=<zip> b=<zip> c=<zip> npc:cherry-picker+

Exactly 6 seats: `label=zip` for a checkpoint, `npc:<name>` for a scripted rival. SEED (default 0) is the
world seed; use seeds you never trained on for the deciding exam (guides/TRAINING_TOURNAMENT_HINTS.md).
"""
import sys
from multiprocessing import Pool

import numpy as np

SPECS = sys.argv[1:]  # name=zip (rl) or npc:name


def play(shift):
    from aggressive_npcs import register
    from retail_simulator.envs.arena import arena_home_regions
    from retail_simulator.envs.arena_match import ArenaMatch, ArenaSeatSpec, resolve_arena_world

    register()
    seats = []
    for s in SPECS:
        if s.startswith("npc:"):
            seats.append(ArenaSeatSpec(name=s[4:], kind="npc", archetype=s[4:]))
        else:
            n, _, p = s.partition("=")
            seats.append(ArenaSeatSpec(name=n, kind="rl", checkpoint=p))
    cfg, _, _ = resolve_arena_world(balanced=True)
    base = arena_home_regions(6)
    m = ArenaMatch(cfg, seats, seed=int(__import__("os").environ.get("SEED", "0")), homes=tuple(base[(i + shift) % 6] for i in range(6)))
    tot = np.zeros(6)
    for _ in range(1000):
        tot += [s.reward for s in m.step().seats]
    return [s.name for s in seats], tot


if __name__ == "__main__":
    assert len(SPECS) == 6, "exactly 6 seats"
    with Pool(6) as pool:
        res = pool.map(play, range(6))
    names = res[0][0]
    totals = np.array([t for _, t in res])
    ranks = np.argsort(np.argsort(-totals, axis=1), axis=1) + 1
    print(f"{'seat':16}{'mean':>12}{'min':>12}{'max':>12}{'wins':>6}{'rank':>7}")
    for k in np.argsort(-totals.mean(axis=0)):
        c = totals[:, k]
        print(f"{names[k]:16}{c.mean():12,.0f}{c.min():12,.0f}{c.max():12,.0f}{int((ranks[:, k] == 1).sum()):6d}{ranks[:, k].mean():7.2f}")
