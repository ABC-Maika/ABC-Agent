"""A zoo of opponent behaviours, so the agent has to read its rivals instead of memorising one family.

Types: parametric experts, the four geographic NPC archetypes, reactive price-matchers / undercutters /
punishers (they read the rivals, including us), random postures, opponents that switch behaviour mid-match,
passive ones, and frozen trained zips (ours and the reference PPO agents). ``train_opponent`` and
``test_opponent`` draw from DIFFERENT parameter ranges so the test measures generalisation.

``evalzoo`` plays a zip against five test-zoo opponents and prints a breakdown by opponent type.
Own script; nothing in ``src/`` is edited.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "scripts"))  # the organizer's scripts (train_vs_npcs)

import krischo_optimize as ko  # noqa: E402

from retail_simulator.core.encoding import flatten_action_space_layout  # noqa: E402
from retail_simulator.envs.arena import arena_balanced_config  # noqa: E402
from retail_simulator.envs.geo_archetypes import GEO_ARCHETYPE_LABELS, geo_archetypes  # noqa: E402

HOMES = [1, 5, 9, 13, 17, 21]
PRICE_SLOT = next(s.offset for s in flatten_action_space_layout().levers if s.spec.name == "price_index")


class ExpertP:
    def __init__(self, params: dict, label: str) -> None:
        self.e, self.label = ko.Expert(params), label

    def reset(self) -> None:
        self.e.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.e(obs)


class GeoNpc:
    def __init__(self, name: str) -> None:
        regions = arena_balanced_config().demand.regions
        self.fn, self.label, self.first = geo_archetypes(len(regions), regions)[name], f"npc:{name}", True

    def reset(self) -> None:
        self.first = True

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action = np.asarray(self.fn(obs, episode_start=self.first), dtype=np.float32).ravel()
        self.first = False
        return action


class Punisher:
    """Reads the rivals' perceived mean price (needs research) and undercuts anyone who prices below it."""

    def __init__(self, params: dict, margin: float, label: str) -> None:
        self.e, self.margin, self.label = ko.Expert(params), margin, label

    def reset(self) -> None:
        self.e.reset()

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action = self.e(obs)
        idx = self.e.idx
        held = [r for r in range(1, ko.N_REGIONS) if obs[idx[f"stores_region_{r}"]] > 0 and obs[idx[f"others_present_region_{r}"]] > 0]
        if held:
            rival = float(np.mean([obs[idx[f"competitor_mean_price_region_{r}"]] for r in held]))
            if rival < float(action[PRICE_SLOT]) - 0.05:
                action[PRICE_SLOT] = min(2.0, max(1.0, rival + self.margin))
        return action


class Switcher:
    """Plays one expert, then another from a random round on (a non-stationary opponent)."""

    def __init__(self, first: dict, second: dict, at: int) -> None:
        self.a, self.b, self.at, self.tick, self.label = ko.Expert(first), ko.Expert(second), at, 0, "switcher"

    def reset(self) -> None:
        self.a.reset(), self.b.reset()
        self.tick = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self.tick += 1
        return (self.a if self.tick <= self.at else self.b)(obs)


class ZipAny:
    """A frozen trained zip: RecurrentPPO (stateful) or PPO/A2C (memoryless)."""

    def __init__(self, path: str) -> None:
        import torch

        torch.set_num_threads(1)
        self.label, self.recurrent = f"zip:{Path(path).stem}", True
        try:
            from sb3_contrib import RecurrentPPO

            self.model = RecurrentPPO.load(path, device="cpu")
        except Exception:
            from stable_baselines3 import A2C, PPO

            self.recurrent = False
            for cls in (PPO, A2C):
                try:
                    self.model = cls.load(path, device="cpu")
                    break
                except Exception:
                    continue
        self.reset()

    def reset(self) -> None:
        self.state, self.first = None, np.array([True])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.recurrent:
            action, self.state = self.model.predict(obs, state=self.state, episode_start=self.first, deterministic=True)
            self.first = np.array([False])
        else:
            action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32).ravel()


def _reactive_params(rng: random.Random, held_out: bool) -> dict:
    p = dict(ko.BASELINE)
    p.update(
        price=rng.uniform(1.2, 1.9),
        w_rival=1.0,
        rival_margin=rng.uniform(-0.4, -0.2) if held_out else rng.uniform(-0.15, 0.1),
        research=0.5,
        marketing=rng.uniform(0.5, 1.0),
        assortment=rng.uniform(0.5, 1.0),
        promotion=rng.uniform(0.5, 1.0),
        max_stores=rng.choice((8, 10, 12)),
        max_class=rng.choice((2, 3, 4)),
        service=1.0,
    )
    return p


def _passive(rng: random.Random) -> dict:
    p = dict(ko.BASELINE)
    p.update(price=rng.uniform(1.0, 1.4), marketing=rng.uniform(0.0, 0.3), assortment=rng.uniform(0.0, 0.4), promotion=0.0,
             max_stores=rng.choice((1, 2, 3)), service=rng.choice((0.3, 1.0)))
    return p


def _opponent(rng: random.Random, pool: list[dict], zips: list[str], held_out: bool):
    r = rng.random()
    if r < 0.25:
        return ExpertP(rng.choice(pool), "expert")
    if r < 0.40:
        return GeoNpc(rng.choice(GEO_ARCHETYPE_LABELS))
    if r < 0.52:
        return ExpertP(_reactive_params(rng, held_out), "reactive:matcher")
    if r < 0.62:
        return Punisher(_reactive_params(rng, held_out), rng.uniform(-0.3, -0.1) if held_out else rng.uniform(-0.1, 0.0), "reactive:punisher")
    if r < 0.72:
        a, b = rng.choice(pool), ko.sample(rng)
        return Switcher(a, b, rng.choice((300, 450, 600, 750)))
    if r < 0.80:
        return ExpertP(ko.sample(rng), "random-posture")
    if r < 0.88:
        return ExpertP(_passive(rng), "passive")
    if zips:
        return ZipAny(rng.choice(zips))
    return ExpertP(rng.choice(pool), "expert")


def train_opponent(rng: random.Random, pool: list[dict], zips: list[str]):
    return _opponent(rng, pool, zips, held_out=False)


def test_opponent(rng: random.Random, pool: list[dict], zips: list[str]):
    return _opponent(rng, pool, zips, held_out=True)


def zoo_match(job: tuple[str, int, list[dict], list[str]]) -> tuple[float, int, list[str], list[float]]:
    """Our zip (seat 0) against five test-zoo opponents; returns (our score, our rank, labels, their scores)."""
    from retail_simulator.envs.parallel_env import RetailParallelEnv

    zip_path, seed, pool, ref_zips = job
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    homes = tuple(int(h) for h in nprng.permutation(HOMES))
    world_seed = int(nprng.integers(0, 2**31 - 1))
    opps = [test_opponent(rng, pool, ref_zips) for _ in range(5)]
    policies = [ZipAny(zip_path), *opps]
    env = RetailParallelEnv(arena_balanced_config(), n_learning_agents=6, home_regions=homes, seed=world_seed, mask_expansion_seats=tuple(range(6)))
    agents = env.possible_agents
    for p in policies:
        p.reset()
    obs, _ = env.reset(seed=world_seed)
    totals = [0.0] * 6
    for _ in range(1000):
        obs, rewards, _, _, _ = env.step({a: np.asarray(policies[i](obs[a]), dtype=np.float32).ravel() for i, a in enumerate(agents)})
        for i, a in enumerate(agents):
            totals[i] += float(rewards[a])
    rank = 1 + sum(o > totals[0] for o in totals[1:])
    return totals[0], rank, [o.label for o in opps], totals[1:]


def cmd_evalzoo(args: argparse.Namespace) -> None:
    pool = [{**ko.BASELINE, **p} for p in json.loads(Path(args.pool).read_text())][-8:]  # the held-out experts
    refs = [str(p) for p in sorted(Path("runs/phase4_rl_confirm").glob("ref_*/round_2.zip"))] + [str(p) for p in Path("runs/algos").glob("*.zip")]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(zoo_match, [(args.zip, 3000 + i, pool, refs) for i in range(args.n)], chunksize=1))
    ours = [r[0] for r in res]
    ranks = [r[1] for r in res]
    print(f"{args.zip}: {len(res)} matches vs the TEST zoo (held-out parameter ranges, 5 opponents each)")
    print(f"our mean {np.mean(ours):,.0f}  min {np.min(ours):,.0f}  max {np.max(ours):,.0f}  mean rank {np.mean(ranks):.2f}  wins {sum(r == 1 for r in ranks)}/{len(ranks)}  total {np.sum(ours):,.0f}")
    by_type: dict[str, list[float]] = {}
    for sc, rk, labels, theirs in res:
        for lab, t in zip(labels, theirs):
            by_type.setdefault(lab.split(":")[0] if lab.startswith(("npc", "zip")) else lab, []).append(sc - t)
    print("our margin over each opponent type (mean of ours minus theirs, matches with that type):")
    for lab, v in sorted(by_type.items(), key=lambda kv: np.mean(kv[1])):
        print(f"  {lab:<20} n={len(v):>3}  margin {np.mean(v):>10,.0f}  we beat them in {np.mean([x > 0 for x in v]) * 100:>3.0f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evalzoo")
    e.add_argument("--zip", required=True)
    e.add_argument("--pool", default="runs/league/pool.json")
    e.add_argument("--n", type=int, default=16)
    e.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    {"evalzoo": cmd_evalzoo}[args.cmd](args)


if __name__ == "__main__":
    main()
