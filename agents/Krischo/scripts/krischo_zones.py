"""Zone-by-zone head-to-head of two agents (who holds more share in each region, by region class).

Six-seat matches: agent A (seat 0), agent B (seat 1) and four fillers; the six METRO homes are rotated so each
agent starts from different places. For every region it compares the final market share of A and B.
Own script; nothing in ``src/`` is edited. Run from the repo root.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "scripts"))

import krischo_zoo as kz  # noqa: E402

from retail_simulator.core.schema import observation_schema  # noqa: E402
from retail_simulator.envs.arena import arena_balanced_config  # noqa: E402
from retail_simulator.envs.parallel_env import RetailParallelEnv  # noqa: E402

CLASSES = ("METRO", "AFFLUENT", "LOYAL", "FRONTIER")
IDX = {f.name: i for i, f in enumerate(observation_schema(25))}
FILLERS = ["agents/misho/jack/checkpoint.zip", "agents/Yavkata/recurrent-ppo/checkpoint.zip"]


def match(job: tuple[str, str, int, int]):
    a_zip, b_zip, rot, seed = job
    pols = [kz.ZipAny(a_zip), kz.ZipAny(b_zip), kz.ZipAny(FILLERS[0]), kz.ZipAny(FILLERS[1]), kz.GeoNpc("fast-follower"), kz.GeoNpc("cherry-picker")]
    homes = kz.HOMES[rot:] + kz.HOMES[:rot]
    env = RetailParallelEnv(arena_balanced_config(), n_learning_agents=6, home_regions=tuple(homes), seed=seed, mask_expansion_seats=tuple(range(6)))
    ag = env.possible_agents
    for p in pols:
        p.reset()
    obs, _ = env.reset(seed=seed)
    tot = [0.0] * 6
    for _ in range(1000):
        obs, rw, *_ = env.step({a: np.asarray(pols[i](obs[a]), dtype=np.float32).ravel() for i, a in enumerate(ag)})
        for i, a in enumerate(ag):
            tot[i] += float(rw[a])
    share = np.array([[obs[ag[s]][IDX[f"market_share_region_{r}"]] for r in range(1, 25)] for s in (0, 1)])
    stores = np.array([[obs[ag[s]][IDX[f"stores_region_{r}"]] for r in range(1, 25)] for s in (0, 1)])
    return tot[0], tot[1], share, stores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="agents/Krischo/rppo-lstm/checkpoint.zip")
    parser.add_argument("--b", default="agents/orcho/market-share/checkpoint.zip")
    parser.add_argument("--names", default="Krischo,Orcho")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    na, nb = args.names.split(",")
    jobs = [(args.a, args.b, rot, 7 + s) for s in range(args.seeds) for rot in range(6)]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(match, jobs, chunksize=1))
    pa, pb = np.mean([r[0] for r in res]), np.mean([r[1] for r in res])
    print(f"{len(res)} matches (six homes rotated, {args.seeds} seeds): mean profit {na} {pa:,.0f} vs {nb} {pb:,.0f}; {na} ahead in {sum(r[0] > r[1] for r in res)} of {len(res)}")
    sh = np.stack([r[2] for r in res])  # matches x 2 x 24
    st = np.stack([r[3] for r in res])
    print(f"\n{'zone (region class)':<22}{'regions':>8}{na + ' stores':>16}{nb + ' stores':>16}{na + ' share':>14}{nb + ' share':>14}{'we lead':>9}")
    for c, cname in enumerate(CLASSES):
        cols = [r for r in range(24) if r % 4 == c]
        a_sh, b_sh = sh[:, 0, cols].mean(), sh[:, 1, cols].mean()
        lead = np.mean(sh[:, 0, cols] > sh[:, 1, cols]) * 100
        print(f"{cname:<22}{len(cols):>8}{st[:, 0, cols].mean():>16.2f}{st[:, 1, cols].mean():>16.2f}{a_sh:>14.3f}{b_sh:>14.3f}{lead:>8.0f}%")
    only_b = np.mean((st[:, 1] > 0) & (st[:, 0] == 0)) * 24
    only_a = np.mean((st[:, 0] > 0) & (st[:, 1] == 0)) * 24
    print(f"\nregions (of 24, per match) where only {nb} has a store: {only_b:.1f}; only {na}: {only_a:.1f}")


if __name__ == "__main__":
    main()
