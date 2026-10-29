"""Equal-budget RL fine-tuning of our agents against the teammates' agents (six seats, no NPCs).

Seat 0 is the learner; seats 1..5 are frozen zips (default: Orcho v4, Yavkata RPPO, Jacky-T, Jack, Yavkata SAC) in a random order, homes rotate
every episode. Algorithms: ``ppo`` and ``a2c`` start from an MLP zip (e.g. the cloned expert), ``rppo`` from an LSTM zip. The observation scale is
un-folded for training and folded back into the first layer when saving, so every saved zip takes RAW observations.
Own script; nothing in ``src/`` is edited. Run from the repo root:

    .venv\\Scripts\\python.exe agents\\Krischo\\scripts\\krischo_ft.py --algo ppo --init agents\\Krischo\\mlp-v6\\checkpoint.zip --out runs\\ft\\ppo.zip --steps 50000
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
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "scripts"))

import krischo_league as kl  # noqa: E402
import krischo_optimize as ko  # noqa: E402
import krischo_zoo as kz  # noqa: E402

from retail_simulator.envs.arena import arena_balanced_config  # noqa: E402
from retail_simulator.envs.parallel_env import RetailParallelEnv  # noqa: E402
from retail_simulator.envs.spaces import build_action_space, build_observation_space  # noqa: E402

FIELD = [
    "agents/orcho/market-share/checkpoint.zip",
    "agents/Yavkata/recurrent-ppo/checkpoint.zip",
    "agents/misho/jacky-t/checkpoint.zip",
    "agents/misho/jack/checkpoint.zip",
    "agents/Yavkata/sac/checkpoint.zip",
]


class FieldEnv(gym.Env):
    """Seat 0 learns; seats 1..5 are the frozen field. Observations are divided by ``scale``."""

    def __init__(self, seed: int, scale: np.ndarray, field: list[str]) -> None:
        super().__init__()
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.scale = scale
        self.opps = [kz.ZipAny(p) for p in field]  # loaded once per process, reset every episode
        self.config = arena_balanced_config()
        self.observation_space = build_observation_space(self.config)
        self.action_space = build_action_space()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        homes = tuple(int(h) for h in self.np_random.permutation(kl.HOMES))
        order = list(self.np_random.permutation(len(self.opps)))
        self.others = [self.opps[i] for i in order]
        for o in self.others:
            o.reset()
        world_seed = int(self.np_random.integers(0, 2**31 - 1))
        self.world = RetailParallelEnv(self.config, n_learning_agents=kl.N_SEATS, home_regions=homes, seed=world_seed, mask_expansion_seats=tuple(range(kl.N_SEATS)))
        self.agents = self.world.possible_agents
        self.obs, _ = self.world.reset(seed=world_seed)
        self.tick = 0
        return self._obs(), {}

    def step(self, action):
        actions = {self.agents[0]: np.asarray(action, dtype=np.float32)}
        for i, a in enumerate(self.agents[1:]):
            actions[a] = np.asarray(self.others[i](self.obs[a]), dtype=np.float32).ravel()
        self.obs, rewards, _, _, _ = self.world.step(actions)
        self.tick += 1
        return self._obs(), float(rewards[self.agents[0]]), False, self.tick >= 1000, {}

    def _obs(self) -> np.ndarray:
        return (np.asarray(self.obs[self.agents[0]], dtype=np.float32) / self.scale).astype(np.float32)


def make_env(seed: int, scale: np.ndarray, field: list[str]):
    from stable_baselines3.common.monitor import Monitor

    return Monitor(FieldEnv(seed, scale, field))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--algo", choices=("ppo", "a2c", "rppo"), required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--n-envs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--log-std", type=float, default=-1.5)
    ap.add_argument("--params", default="agents/Krischo/rppo-lstm/expert_params_v6.json", help="expert used only to size the obs scale")
    ap.add_argument("--pool", default="runs/league2/pool.json")
    ap.add_argument("--field", nargs="*", default=FIELD)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from stable_baselines3 import A2C, PPO
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

    torch.set_num_threads(args.threads)
    set_random_seed(args.seed)
    pool = [{**ko.BASELINE, **p} for p in json.loads(Path(args.pool).read_text())]
    expert = {**ko.BASELINE, **json.loads(Path(args.params).read_text())}
    obs_data, _ = kl.collect_league(expert, pool, 8, 9000, 6)
    scale = (np.maximum(np.abs(obs_data).max(axis=(0, 1)), 1.0) * 1.5).astype(np.float32)
    scale_t = torch.tensor(scale, dtype=torch.float32)

    venv = SubprocVecEnv([partial(make_env, args.seed * 1000 + i, scale, args.field) for i in range(args.n_envs)])
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.999)
    common = dict(learning_rate=args.lr, gamma=0.999, gae_lambda=0.98, ent_coef=0.0, device="cpu", verbose=1, seed=args.seed)
    if args.algo == "rppo":
        from sb3_contrib import RecurrentPPO

        n_steps = 256
        init = RecurrentPPO.load(args.init, device="cpu")
        model = RecurrentPPO("MlpLstmPolicy", venv, policy_kwargs=init.policy_kwargs, n_steps=n_steps, batch_size=args.n_envs * n_steps // 2, n_epochs=3, clip_range=0.1, **common)
        first = [m.weight_ih_l0 for m in (model.policy.lstm_actor, model.policy.lstm_critic) if m is not None]
    else:
        init = PPO.load(args.init, device="cpu")
        if args.algo == "ppo":
            model = PPO("MlpPolicy", venv, policy_kwargs=init.policy_kwargs, n_steps=256, batch_size=args.n_envs * 128, n_epochs=3, clip_range=0.1, **common)
        else:
            model = A2C("MlpPolicy", venv, policy_kwargs=init.policy_kwargs, n_steps=64, **common)
        first = [model.policy.mlp_extractor.policy_net[0].weight, model.policy.mlp_extractor.value_net[0].weight]
    model.policy.load_state_dict(init.policy.state_dict())
    with torch.no_grad():
        for w in first:  # un-fold: the env now feeds obs / scale
            w.mul_(scale_t)
        model.policy.log_std.data.fill_(args.log_std)

    def save_folded(path: str) -> None:
        saved, log_std = [w.data.clone() for w in first], model.policy.log_std.data.clone()
        with torch.no_grad():
            for w in first:
                w.div_(scale_t)
            model.policy.log_std.data.fill_(-2.0)
        model.save(path)
        with torch.no_grad():
            for w, s in zip(first, saved):
                w.data.copy_(s)
            model.policy.log_std.data.copy_(log_std)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model.learn(total_timesteps=args.steps)
    save_folded(args.out)
    print(f"done {args.algo}: {model.num_timesteps} steps in {int(time.time() - t0)}s -> {args.out}", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
