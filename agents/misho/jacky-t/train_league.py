"""Jacky-T, league stage: continue the TQC run against the teammates' trained agents.

Seat 0 is Jacky-T (TQC, expansion-masked). The other five seats are the trained checkpoints on
``main`` (Krischo twice, Yavkata, orcho, and Jack), shuffled over the six metro homes each episode.
Each opponent is loaded with the ladder's own ``_materialize_predicts``, so a recurrent opponent
keeps its LSTM state across the match.

Reward per tick:
* the world's z-normalized weighted sum (profit 0.3, revenue 0.3, market share 0.3, loyalty 0.1);
* minus ``--penalty`` while Jacky-T is not in the top 2 by cumulative profit;
* plus ``--bonus`` while it is first and its lead over second place exceeds half of its own
  cumulative profit (second place below 50% of Jacky-T's score).
The two rank terms start after ``--rank-warmup`` ticks, when cumulative profits are not yet noise.
Scoring is still plain profit; the rank terms only shape training.

It resumes a snapshot of ``train.py``'s run (the obs scale saved beside it), trains up to ``--cap``
TOTAL steps, folds the obs scale into the network and writes a scorable zip.

    python agents/misho/jacky-t/train_league.py jacky_t --resume runs/jacky_t/ckpt/<snapshot>.zip
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
sys.path.insert(0, str(HERE))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import train as jacky  # noqa: E402  (the TQC trainer next to this file)
from gymnasium.utils import seeding  # noqa: E402
from sb3_contrib import TQC  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from retail_simulator.envs.arena import arena_balanced_config, arena_home_regions  # noqa: E402
from retail_simulator.envs.arena_match import _materialize_predicts  # noqa: E402
from retail_simulator.envs.geo_archetypes import _name_to_index as geo_name_to_index  # noqa: E402
from retail_simulator.envs.geo_archetypes import geo_archetypes  # noqa: E402
from retail_simulator.envs.parallel_env import RetailParallelEnv  # noqa: E402
from retail_simulator.envs.spaces import build_action_space, build_observation_space  # noqa: E402

N_SEATS = 6
LEARNER = 0
OPPONENTS: dict[str, str] = {  # the trained agents on main
    "krischo": "agents/Krischo/rppo-lstm/checkpoint.zip",
    "krischo-v8": "agents/Krischo/rppo-lstm/checkpoint_v8.zip",
    "yavkata": "agents/Yavkata/recurrent-ppo/checkpoint.zip",
    "orcho": "agents/orcho/market-share/checkpoint.zip",
    "jack": "agents/misho/jack/checkpoint.zip",
}


class LeagueEnv(gym.Env):
    """Jacky-T against the five trained agents, with rank-based reward shaping."""

    def __init__(
        self,
        seed: int,
        obs_scale: np.ndarray,
        weights: tuple[float, float, float, float],
        bonus: float,
        penalty: float,
        rank_warmup: int,
        ticks: int = 1000,
        neg_penalty: float = 0.0,
        pool: Sequence[str] | None = None,
        any_home: bool = False,
        prime_weight: float = 0.0,
    ) -> None:
        super().__init__()
        # ``prime_weight``: extra reward per tick for the demand-weighted market share held in the
        # METRO (1, 5, 9, ...) and AFFLUENT (2, 6, 10, ...) regions, the two biggest-demand classes
        # (mean base demand 1,400 and 900, against 600 and 350 for LOYAL and FRONTIER).
        self.prime_weight = prime_weight
        # ``any_home``: draw the six homes from all enterable regions (1..24), not only the six
        # published metro homes, so the policy is not tied to one starting geography.
        self.any_home = any_home
        # ``pool``: the five opponents (OPPONENTS keys or geo archetype names); default: all five.
        self.pool = list(pool) if pool is not None else list(OPPONENTS)
        self.neg_penalty = neg_penalty
        self.np_random, _ = seeding.np_random(seed)
        profit, revenue, share, loyalty = weights
        base_config = arena_balanced_config()
        self.config = replace(
            base_config,
            reward=replace(
                base_config.reward,
                mode="weighted",
                weight_profit=profit,
                weight_revenue=revenue,
                weight_market_share=share,
                weight_loyalty=loyalty,
            ),
        )
        self.scale = obs_scale.astype(np.float32)
        self.bonus, self.penalty, self.rank_warmup, self.ticks = bonus, penalty, rank_warmup, ticks
        self.homes = arena_home_regions(N_SEATS)
        regions = self.config.demand.regions
        obs_index = geo_name_to_index(len(regions))
        prime = [r for r in range(1, len(regions)) if (r - 1) % 4 in (0, 1)]  # METRO, AFFLUENT
        demand = np.array([regions[r].base_regional_demand for r in prime], dtype=np.float64)
        self._prime_idx = [obs_index[f"market_share_region_{r}"] for r in prime]
        self._prime_w = demand / demand.sum()
        self.observation_space = build_observation_space(self.config)
        self.action_space = jacky.base.bounded_logit_space(build_action_space())
        self._predicts: dict[str, Any] = {}

    def _predict(self, name: str) -> Any:
        if name not in OPPONENTS:  # a scripted archetype: a fresh closure, it carries a tick counter
            roster = geo_archetypes(len(self.config.demand.regions), self.config.demand.regions)
            return roster[name]
        if name not in self._predicts:
            self._predicts.update(
                _materialize_predicts(
                    {name: OPPONENTS[name]},
                    action_space=build_action_space(),
                    config=self.config,
                )
            )
        return self._predicts[name]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Any:
        super().reset(seed=seed)
        rng = random.Random(int(self.np_random.integers(0, 2**31 - 1)))
        if self.any_home:
            n_regions = len(self.config.demand.regions)
            drawn = self.np_random.choice(np.arange(1, n_regions), size=N_SEATS, replace=False)
            homes = tuple(int(h) for h in drawn)
        else:
            homes = tuple(int(h) for h in self.np_random.permutation(self.homes))
        names = list(self.pool)
        rng.shuffle(names)
        self.opponents = [self._predict(name) for name in names]
        self.world = RetailParallelEnv(
            self.config,
            n_learning_agents=N_SEATS,
            home_regions=homes,
            seed=int(self.np_random.integers(0, 2**31 - 1)),
            mask_expansion_seats=tuple(range(N_SEATS)),
        )
        self.agents_ = self.world.possible_agents
        self._obs, _ = self.world.reset(seed=int(self.np_random.integers(0, 2**31 - 1)))
        self.cum = np.zeros(N_SEATS)
        self.t = 0
        return self._learner_obs(), {"homes": homes, "opponents": names}

    def step(self, action: np.ndarray) -> Any:
        acts = {self.agents_[LEARNER]: np.asarray(action, dtype=np.float32)}
        for seat, predict in enumerate(self.opponents, start=1):
            agent = self.agents_[seat]
            acts[agent] = np.asarray(
                predict(self._obs[agent], episode_start=self.t == 0), dtype=np.float32
            ).ravel()
        self._obs, rewards, _, _, infos = self.world.step(acts)
        self.t += 1
        self.cum += [infos[a]["components"]["profit"] for a in self.agents_]
        reward = float(rewards[self.agents_[LEARNER]]) + self._rank_shaping()
        if self.prime_weight:
            shares = np.asarray(self._obs[self.agents_[LEARNER]])[self._prime_idx]
            reward += self.prime_weight * float(shares @ self._prime_w)
        rank = int((self.cum > self.cum[LEARNER]).sum()) + 1
        return self._learner_obs(), reward, False, self.t >= self.ticks, {"rank": rank}

    def _rank_shaping(self) -> float:
        if self.t < self.rank_warmup:
            return 0.0
        mine = self.cum[LEARNER]
        rivals = np.delete(self.cum, LEARNER)
        rank = int((rivals > mine).sum()) + 1
        shaping = -self.neg_penalty if mine < 0 else 0.0  # losing money overall
        if rank > 2:
            return shaping - self.penalty
        if rank == 1 and mine > 0 and mine - rivals.max() > 0.5 * mine:
            return shaping + self.bonus
        return shaping

    def _learner_obs(self) -> np.ndarray:
        return (self._obs[self.agents_[LEARNER]] / self.scale).astype(np.float32)


def make_env_fns(args: argparse.Namespace, obs_scale: np.ndarray) -> list[Any]:
    seeds = [
        int(s.generate_state(1)[0]) for s in np.random.SeedSequence(args.seed).spawn(args.n_envs)
    ]
    return [
        lambda s=s: Monitor(
            LeagueEnv(
                s,
                obs_scale,
                args.reward_weights,
                args.bonus,
                args.penalty,
                args.rank_warmup,
                neg_penalty=args.neg_penalty,
                any_home=args.any_home,
                prime_weight=args.prime_weight,
            ),
            info_keywords=("rank",),
        )
        for s in seeds
    ]


def train(args: argparse.Namespace) -> TQC:
    set_random_seed(args.seed)
    obs_scale = np.load(args.resume.parent / "obs_scale.npy")
    venv = DummyVecEnv(make_env_fns(args, obs_scale))
    venv.action_space.seed(args.seed)
    model = TQC.load(args.resume, env=venv)
    done = int(model.num_timesteps)
    remaining = args.cap - done
    if remaining <= 0:
        raise SystemExit(f"snapshot is already at {done} steps, cap is {args.cap}")
    print(f"resumed {args.resume} at {done} steps; training {remaining} more to the cap {args.cap}")
    # The replay buffer is not part of a snapshot: refill it before updating, or the first
    # updates would overfit a handful of transitions.
    model.learning_starts = done + args.learning_starts
    sample_action = None
    if args.from_actor:
        # Use the cloned actor from step one instead of SB3's random warm-up actions, and a small
        # learning rate so the first updates (critics still stale) do not undo the clone.
        from stable_baselines3.common.utils import get_schedule_fn

        model.lr_schedule = get_schedule_fn(args.lr)
        sample_action = model._sample_action
        model._sample_action = lambda ls, action_noise=None, n_envs=1: sample_action(
            0, action_noise, n_envs
        )
    ckpt_dir = Path("runs") / args.label / "league_ckpt"
    callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(ckpt_dir),
        name_prefix="jacky_t_league",
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.save(ckpt_dir / "obs_scale.npy", obs_scale)
    model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=False)
    if sample_action is not None:
        del model._sample_action  # the lambda must not be pickled into the saved zip

    # Fold the obs scale in, checking the network is unchanged on observations of this env.
    env = LeagueEnv(args.seed + 1, obs_scale, args.reward_weights, 0.0, 0.0, 0)
    scaled, _ = env.reset()
    samples = [scaled]
    for _ in range(40):
        scaled, *_ = env.step(env.action_space.sample())
        samples.append(scaled)
    scaled_samples = np.stack(samples)
    before = jacky.network_outputs(model, scaled_samples)
    jacky.fold_obs_scale(model, obs_scale)
    after = jacky.network_outputs(model, scaled_samples * obs_scale)
    drift = float(np.max(np.abs(before - after) / np.maximum(1.0, np.abs(before))))
    if drift > 1e-4:
        raise RuntimeError(f"folding the obs scale changed predictions by {drift:.2e}")
    print(f"obs scale folded into the network (max prediction drift {drift:.1e})")
    return model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jacky-T league stage (TQC vs teammates' agents).")
    parser.add_argument("label")
    parser.add_argument("--resume", type=Path, required=True, help="snapshot zip from train.py")
    parser.add_argument("--cap", type=int, default=100_000, help="TOTAL steps, snapshot included")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-starts", type=int, default=4_000)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument("--bonus", type=float, default=0.25, help="per tick, lead > half own score")
    parser.add_argument("--penalty", type=float, default=0.25, help="per tick, outside the top 2")
    parser.add_argument("--rank-warmup", type=int, default=50, help="ticks before rank terms apply")
    parser.add_argument(
        "--neg-penalty", type=float, default=0.0, help="per tick while cumulative profit is below 0"
    )
    parser.add_argument(
        "--from-actor",
        action="store_true",
        help="the snapshot is a behaviour-cloned actor: act with it from the first step (no random "
        "warm-up), keep its low noise and use a small learning rate",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--prime-weight",
        type=float,
        default=0.0,
        help="per-tick reward x the demand-weighted market share in METRO and AFFLUENT regions",
    )
    parser.add_argument(
        "--any-home",
        action="store_true",
        help="draw the six homes from all 24 enterable regions each episode (default: the six "
        "published metro homes, shuffled over the seats)",
    )
    parser.add_argument(
        "--reward-weights",
        type=lambda text: tuple(float(part) for part in text.split(",")),
        default=(0.3, 0.3, 0.3, 0.1),
    )
    parser.add_argument("--out", type=Path, default=Path("agents/misho/jacky-t/checkpoint.zip"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model = train(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
