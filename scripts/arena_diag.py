"""Check one agent on the three things the team cares about: staff happiness, profit swings, lever use.

Seats the agent first, with the same rivals as the replay lineup, plays one match and prints:
staff happiness (mean after round 100), profit volatility (typical round-to-round change and the
share of rounds where profit flips sign) and, per continuous lever, mean / spread / whether it is
inside the 5 to 95 percent band.

    python scripts/arena_diag.py orcho-a2c-v2-s0=runs/v2/orcho-a2c-v2-s0.zip
"""

from __future__ import annotations

import argparse

import numpy as np
from aggressive_npcs import register

from retail_simulator.envs.arena_match import ArenaMatch, ArenaSeatSpec, resolve_arena_world

RIVALS = {
    "a2c-profit": "runs/algo_test2/a2c_profit.zip",
    "ppo-profit": "runs/algo_test2/ppo_profit.zip",
    "rppo-profit": "runs/algo_test2/recurrent_ppo_profit.zip",
    "ppo-weighted": "runs/algo_test2/ppo_weighted.zip",
}
RANGE = {"price_index": (0.5, 2.0)}
CONTINUOUS = (
    "price_index", "marketing", "assortment", "promotion", "research",
    "service_level", "loyalty_spend", "wage_spend", "warehouse_invest",
)


def main() -> int:
    register()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("agent", help="LABEL=PATH")
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    label, _, path = args.agent.partition("=")
    seats = [ArenaSeatSpec(name=label, kind="rl", checkpoint=path)]
    seats += [ArenaSeatSpec(name=n, kind="rl", checkpoint=p) for n, p in RIVALS.items()]
    seats += [ArenaSeatSpec(name="cherry-picker+", kind="npc", archetype="cherry-picker+")]
    cfg, _, _ = resolve_arena_world(balanced=True)
    match = ArenaMatch(cfg, seats, seed=args.seed)
    profit, happy, levers, total = [], [], [], np.zeros(len(seats))
    for _ in range(args.rounds):
        res = match.step()
        total += [s.reward for s in res.seats]
        me = res.seats[0]
        profit.append(me.profit)
        happy.append(me.employee_happiness)
        levers.append(me.levers)
    p = np.array(profit)
    d = np.diff(p)
    rank = int(np.argsort(-total).tolist().index(0)) + 1
    print(f"{label}: score {total[0]:,.0f} (rank {rank} of {len(seats)}), final stores {res.seats[0].stores}")
    print(
        f"  profit/round mean {p.mean():.0f}, typical change {np.median(abs(d)):.0f}, "
        f"sign flips {100 * np.mean(np.sign(p[1:]) != np.sign(p[:-1])):.0f}%"
    )
    print(f"  staff happiness: mean after round 100 = {np.mean(happy[100:]):.2f}")
    print(f"  {'lever':17}{'mean':>7}{'std':>7}  band 5-95%")
    in_use = 0
    for name in CONTINUOUS:
        v = np.array([lv.get(name, 0.0) for lv in levers])
        lo, hi = RANGE.get(name, (0.0, 1.0))
        frac = (v.mean() - lo) / (hi - lo)
        ok = 0.05 <= frac <= 0.95
        in_use += ok
        print(f"  {name:17}{v.mean():7.2f}{v.std():7.2f}  {'in use' if ok else 'PINNED/UNUSED'}")
    print(f"  levers in use: {in_use}/9")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
