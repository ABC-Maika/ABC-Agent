"""Compare expansion-order strategies of the rule expert in six-seat matches against Orcho, Jack and Yavkata.

The order is set by ``contest_pen`` (expansion score = region demand / (1 + pen * rivals already present)),
``score_min`` (skip weak regions), ``max_class`` (3 = METRO, AFFLUENT, LOYAL) and ``max_stores``.
Own script; nothing in ``src/`` is edited. Run from the repo root.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "scripts"))

import krischo_league as kl  # noqa: E402
import krischo_optimize as ko  # noqa: E402

OPPS = ["agents/orcho/market-share/checkpoint.zip", "agents/misho/jack/checkpoint.zip", "agents/Yavkata/recurrent-ppo/checkpoint.zip"]
POOL = "runs/league2/pool.json"

VARIANTS = {
    "6 stores (METRO only)": {"max_stores": 6, "max_class": 1},
    "8 stores (+AFFLUENT)": {"max_stores": 8, "max_class": 2},
    "10 stores (+AFFLUENT/LOYAL)": {"max_stores": 10, "max_class": 3},
    "12 stores (+AFFLUENT/LOYAL)": {"max_stores": 12, "max_class": 3},
}


def match(job: tuple[list[tuple[str, object]], int]) -> list[float]:
    """Like ``kl.play_match`` but zips of any algorithm (Orcho, Jack, Yavkata are not RecurrentPPO) load through ``ZipAny``."""
    import krischo_zoo as kz
    from retail_simulator.envs.arena import arena_balanced_config
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    specs, seed = job
    rng = np.random.default_rng(seed)
    homes = tuple(int(h) for h in rng.permutation(kl.HOMES))
    world_seed = int(rng.integers(0, 2**31 - 1))
    env = RetailParallelEnv(arena_balanced_config(), n_learning_agents=6, home_regions=homes, seed=world_seed, mask_expansion_seats=tuple(range(6)))
    agents = env.possible_agents
    pols = [ko.Expert(p) if k == "expert" else kz.ZipAny(p) for k, p in specs]
    for p in pols:
        p.reset()
    obs, _ = env.reset(seed=world_seed)
    tot = [0.0] * 6
    for _ in range(1000):
        obs, rw, *_ = env.step({a: np.asarray(pols[i](obs[a]), dtype=np.float32).ravel() for i, a in enumerate(agents)})
        for i, a in enumerate(agents):
            tot[i] += float(rw[a])
    return tot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="agents/Krischo/rppo-lstm/expert_params_v5.json")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    base = {**ko.BASELINE, **json.loads(Path(args.params).read_text())}
    pool = [{**ko.BASELINE, **p} for p in json.loads(Path(POOL).read_text())]
    jobs, tags = [], []
    for name, over in VARIANTS.items():
        for i in range(args.n):
            opp = [("zip", z) for z in OPPS] + [("expert", e) for e in kl.draw(pool, 5000 + i, 2)]
            jobs.append(([("expert", {**base, **over}), *opp], 5000 + i))
            tags.append(name)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(match, jobs, chunksize=1))
    print(f"{'variant':<26}{'ours':>11}{'Orcho':>11}{'min':>11}{'rank':>6}{'beat Orcho':>12}")
    for name in VARIANTS:
        r = [x for x, t in zip(res, tags) if t == name]
        rank = [1 + sum(o > x[0] for o in x[1:]) for x in r]
        print(f"{name:<26}{np.mean([x[0] for x in r]):>11,.0f}{np.mean([x[1] for x in r]):>11,.0f}{min(x[0] for x in r):>11,.0f}{np.mean(rank):>6.2f}{sum(x[0] > x[1] for x in r):>9}/{len(r)}")


if __name__ == "__main__":
    main()
