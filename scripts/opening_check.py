"""Opening-speed gate: does seat 0 (first spec) reach 4 stores by round 3 and 12 by round 90, and stay at 12?

    SEED=1 python scripts/opening_check.py me=<zip> b=<zip> ... (6 specs, same format as arena_rotate_parallel.py)

Prints, for each of the 6 home rotations, stores at rounds 3 / 90 / 1000, the lowest store count after
round 90, the lowest cash, and PASS/FAIL. A fine-tuned or re-cloned agent that fails is thrown away.
"""
import os
import sys
from multiprocessing import Pool

SPECS = sys.argv[1:]


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
    m = ArenaMatch(cfg, seats, seed=int(os.environ.get("SEED", "0")), homes=tuple(base[(i + shift) % 6] for i in range(6)))
    stores, cash = [], []
    for _ in range(1000):
        me = m.step().seats[0]
        stores.append(me.stores)
        cash.append(me.cash)
    return shift, stores, cash


if __name__ == "__main__":
    assert len(SPECS) == 6, "exactly 6 seats"
    with Pool(6) as pool:
        res = sorted(pool.map(play, range(6)))
    print(f"{'home':>4}{'r3':>4}{'r90':>5}{'r1000':>7}{'min>90':>8}{'min cash':>10}  verdict")
    ok_all = True
    for shift, st, cash in res:
        ok = st[2] >= 4 and st[89] >= 12 and min(st[90:]) >= 12
        ok_all &= ok
        print(f"{shift:>4}{st[2]:>4}{st[89]:>5}{st[-1]:>7}{min(st[90:]):>8}{min(cash):>10,.0f}  {'PASS' if ok else 'FAIL'}")
    print("GATE:", "PASS" if ok_all else "FAIL")
