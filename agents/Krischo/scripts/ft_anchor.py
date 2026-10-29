"""Anchored PPO fine-tune: train in chunks, after each chunk pull the weights halfway back to the start (BC clone) and save a folded zip."""
import argparse, sys
from pathlib import Path
import numpy as np, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import train_vs_npcs as base
from ft_generic import rescale

p = argparse.ArgumentParser()
p.add_argument("--init", type=Path, required=True); p.add_argument("--out", required=True)
p.add_argument("--chunks", type=int, default=5); p.add_argument("--chunk-steps", type=int, default=12000)
p.add_argument("--alpha", type=float, default=0.5); p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--n-envs", type=int, default=4); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
obs_scale, _ = base.estimate_obs_scale(a.seed)
ns = argparse.Namespace(seed=a.seed, n_envs=a.n_envs, episode_ticks=1000, bound_logits=False, reward="profit")
venv = VecNormalize(DummyVecEnv(base.make_env_fns(ns, obs_scale=obs_scale)), norm_obs=False, norm_reward=True)
model = PPO.load(a.init, env=venv, custom_objects={"learning_rate": a.lr, "lr_schedule": lambda _: a.lr}, device="cpu")
rescale(model, "ppo", obs_scale)
anchor = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
for c in range(1, a.chunks + 1):
    model.learn(total_timesteps=a.chunk_steps, reset_num_timesteps=False)
    with torch.no_grad():
        sd = model.policy.state_dict()
        for k, v in sd.items():
            if v.dtype.is_floating_point:
                v.copy_(anchor[k] + a.alpha * (v - anchor[k]))
    rescale(model, "ppo", 1.0 / obs_scale)
    model.save(f"{a.out}_c{c}.zip")
    rescale(model, "ppo", obs_scale)
    print(f"chunk {c} saved ({c * a.chunk_steps} steps)", flush=True)
