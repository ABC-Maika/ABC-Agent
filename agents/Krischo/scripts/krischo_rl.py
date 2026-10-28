"""RL fine-tuning of Krischo's LSTM agent against the league opponents (six seats, no NPCs).

Starts from a distilled zip (e.g. checkpoint_v6.zip), plays seat 0 against five expert opponents drawn from
the training pool (the held-out test opponents are excluded) and improves the policy with RecurrentPPO on
profit. Many environments run in parallel processes. The observation scale is un-folded for training and
folded back into the LSTM input weights when saving, so every saved zip takes RAW observations.

Own script; nothing in ``src/`` is edited. Run from the repo root.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from functools import partial
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "scripts"))  # the organizer's scripts (train_vs_npcs)

import krischo_league as kl  # noqa: E402
import krischo_optimize as ko  # noqa: E402
import krischo_zoo as kz  # noqa: E402

from retail_simulator.core.schema import observation_schema  # noqa: E402
from retail_simulator.envs.arena import arena_balanced_config  # noqa: E402
from retail_simulator.envs.parallel_env import RetailParallelEnv  # noqa: E402
from retail_simulator.envs.spaces import build_action_space, build_observation_space  # noqa: E402


class LeagueEnv(gym.Env):
    """Seat 0 is the learner; seats 1..5 are expert opponents. Observations are divided by ``scale``."""

    def __init__(self, seed: int, pool: list[dict], hard_frac: float, scale: np.ndarray, zips: list[str] | None = None, orcho: str | None = None, orcho_frac: float = 0.0, bonus: float = 0.0) -> None:
        super().__init__()
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.pool, self.hard_frac, self.scale, self.zips = pool, hard_frac, scale, zips
        self.orcho, self.orcho_frac, self.bonus = orcho, orcho_frac, bonus  # bonus: shaping reward per new AFFLUENT/LOYAL store after the 6 METRO ones
        idx = {f.name: i for i, f in enumerate(observation_schema(25))}
        self.store_cols = [idx[f"stores_region_{r}"] for r in range(1, 25)]
        self.total_col = idx["stores"]
        config = arena_balanced_config()
        self.config = config
        self.observation_space = build_observation_space(config)
        self.action_space = build_action_space()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = random.Random(int(self.np_random.integers(0, 2**31 - 1)))
        homes = tuple(int(h) for h in self.np_random.permutation(kl.HOMES))
        if self.zips is not None:  # the opponent zoo: experts, NPCs, reactive, switching, passive, frozen zips
            others = [kz.train_opponent(rng, self.pool, self.zips) for _ in range(5)]
            if rng.random() < self.hard_frac:  # sometimes meet the strongest family on purpose
                for i in range(3):
                    others[i] = kz.ExpertP(kl.make_hard(rng), "hard")
        else:
            opps = [kl.make_hard(rng) for _ in range(3)] + rng.sample(self.pool, 2) if rng.random() < self.hard_frac else rng.sample(self.pool, 5)
            others = [ko.Expert(o) for o in opps]
        if self.orcho and rng.random() < self.orcho_frac:
            others[rng.randrange(5)] = kz.ZipAny(self.orcho)
        world_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.world = RetailParallelEnv(
            self.config,
            n_learning_agents=kl.N_SEATS,
            home_regions=homes,
            seed=world_seed,
            mask_expansion_seats=tuple(range(kl.N_SEATS)),
        )
        self.agents = self.world.possible_agents
        self.others = others
        for o in self.others:
            o.reset()
        self.obs, _ = self.world.reset(seed=world_seed)
        self.tick = 0
        return self._learner_obs(), {}

    def step(self, action):
        actions = {self.agents[0]: np.asarray(action, dtype=np.float32)}
        for i, a in enumerate(self.agents[1:]):
            actions[a] = np.asarray(self.others[i](self.obs[a]), dtype=np.float32).ravel()
        before = np.asarray(self.obs[self.agents[0]])
        self.obs, rewards, _, _, _ = self.world.step(actions)
        self.tick += 1
        reward = float(rewards[self.agents[0]])
        if self.bonus:
            after = np.asarray(self.obs[self.agents[0]])
            if before[self.total_col] >= 6 and after[self.total_col] <= 12:
                for r, col in enumerate(self.store_cols):  # region r+1: class (r % 4), 1 = AFFLUENT, 2 = LOYAL
                    if before[col] == 0 and after[col] > 0:
                        reward += self.bonus if r % 4 in (1, 2) else -self.bonus if r % 4 == 3 else 0.0  # FRONTIER does not pay
        return self._learner_obs(), reward, False, self.tick >= 1000, {}

    def _learner_obs(self) -> np.ndarray:
        return (np.asarray(self.obs[self.agents[0]], dtype=np.float32) / self.scale).astype(np.float32)


def make_env(seed: int, pool: list[dict], hard_frac: float, scale: np.ndarray, zips: list[str] | None = None, orcho: str | None = None, orcho_frac: float = 0.0, bonus: float = 0.0):
    from stable_baselines3.common.monitor import Monitor

    return Monitor(LeagueEnv(seed, pool, hard_frac, scale, zips, orcho, orcho_frac, bonus))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", default="agents/Krischo/rppo-lstm/checkpoint_v6.zip")
    parser.add_argument("--pool", default="runs/league2/pool.json", help="training opponents")
    parser.add_argument("--test-pool", default="runs/league/pool.json", help="its newest 8 are held out for testing")
    parser.add_argument("--params", default="agents/Krischo/rppo-lstm/expert_params_v4.json", help="expert used only to size the obs scale")
    parser.add_argument("--out", default="agents/Krischo/rppo-lstm/checkpoint_v8.zip")
    parser.add_argument("--steps", type=int, default=240_000)
    parser.add_argument("--chunk", type=int, default=80_000)
    parser.add_argument("--n-envs", type=int, default=11)
    parser.add_argument("--hard-frac", type=float, default=0.4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--log-std", type=float, default=-1.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--zoo", action="store_true", help="train against the opponent zoo instead of the expert pool only")
    parser.add_argument("--zips", nargs="*", default=["agents/Krischo/rppo-lstm/checkpoint_v4.zip", "agents/Krischo/rppo-lstm/checkpoint_v6.zip", "agents/Krischo/rppo-lstm/checkpoint_v8.zip"], help="frozen zips used as opponents in the zoo")
    parser.add_argument("--research-bias", type=float, default=0.0, help="raise the research output so rival signals are readable")
    parser.add_argument("--orcho", default=None, help="zip of Orcho's agent, used as one opponent in --orcho-frac of the matches")
    parser.add_argument("--orcho-frac", type=float, default=0.6)
    parser.add_argument("--bonus", type=float, default=0.0, help="shaping reward per new AFFLUENT/LOYAL store after the first 6 stores")
    parser.add_argument("--exp-std", type=float, default=None, help="log-std of the 32 expansion logits only (explore other regions; the rest keeps --log-std)")
    args = parser.parse_args()

    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    torch.set_num_threads(4)
    set_random_seed(args.seed)
    full_pool = [{**ko.BASELINE, **p} for p in json.loads(Path(args.pool).read_text())]
    test = [{**ko.BASELINE, **p} for p in json.loads(Path(args.test_pool).read_text())[-8:]]
    train_pool = [p for p in full_pool if p not in test]
    print(f"training opponents {len(train_pool)} (held out {len(full_pool) - len(train_pool)} test opponents)", flush=True)

    expert = {**ko.BASELINE, **json.loads(Path(args.params).read_text())}
    obs_data, _ = kl.collect_league(expert, train_pool, 8, 9000, 11)
    scale = (np.maximum(np.abs(obs_data).max(axis=(0, 1)), 1.0) * 1.5).astype(np.float32)

    zips = args.zips if args.zoo else None
    venv = SubprocVecEnv([partial(make_env, args.seed * 1000 + i, train_pool, args.hard_frac, scale, zips, args.orcho, args.orcho_frac, args.bonus) for i in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.999)
    n_steps = 256
    model = RecurrentPPO(
        "MlpLstmPolicy",
        venv,
        n_steps=n_steps,
        batch_size=args.n_envs * n_steps // 2,
        n_epochs=3,
        learning_rate=args.lr,
        gamma=0.999,
        gae_lambda=0.98,
        clip_range=0.1,
        ent_coef=0.0,
        device="cpu",
        verbose=1,
        seed=args.seed,
    )
    init = RecurrentPPO.load(args.init, device="cpu")
    model.policy.load_state_dict(init.policy.state_dict())
    pol = model.policy
    scale_t = torch.tensor(scale, dtype=torch.float32)
    lstms = [m for m in (pol.lstm_actor, pol.lstm_critic) if m is not None]
    with torch.no_grad():
        for m in lstms:  # un-fold: the env now feeds obs/scale
            m.weight_ih_l0.mul_(scale_t)
        pol.log_std.data.fill_(args.log_std)
        if args.exp_std is not None:
            from retail_simulator.core.encoding import flatten_action_space_layout

            exp = next(s for s in flatten_action_space_layout().levers if s.spec.name == "expansion")
            pol.log_std.data[exp.offset : exp.offset + exp.width] = args.exp_std
        if args.research_bias:  # shift the mean research output; RL can lower it again if it does not pay
            from retail_simulator.core.encoding import flatten_action_space_layout

            slot = next(s.offset for s in flatten_action_space_layout().levers if s.spec.name == "research")
            pol.action_net.bias[slot] += args.research_bias

    def save_folded(path: str) -> None:
        saved = [m.weight_ih_l0.data.clone() for m in lstms]
        log_std = pol.log_std.data.clone()
        with torch.no_grad():
            for m in lstms:
                m.weight_ih_l0.div_(scale_t)
            pol.log_std.data.fill_(-2.0)
        model.save(path)
        with torch.no_grad():
            for m, w in zip(lstms, saved):
                m.weight_ih_l0.data.copy_(w)
            pol.log_std.data.copy_(log_std)

    t0 = time.time()
    done = 0
    while done < args.steps:
        model.learn(total_timesteps=args.chunk, reset_num_timesteps=False)
        done += args.chunk
        path = str(Path(args.out).with_name(f"{Path(args.out).stem}_{done // 1000}k.zip"))
        save_folded(path)
        print(f"== {done} steps in {int(time.time() - t0)}s -> {path}", flush=True)
    save_folded(args.out)
    print(f"done in {int(time.time() - t0)}s -> {args.out}", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
