"""Train A2C or TQC against the NPC field (own script; ppo/sac/recurrent_ppo live in train_vs_npcs.py).

Mirrors the choices of ``train_vs_npcs.py`` so the zips are comparable:
* A2C (on-policy, PPO family): raw observations, ``VecNormalize(norm_obs=False, norm_reward=True)``, native
  action bounds (clipped Gaussian samples).
* TQC (off-policy, SAC family): argmax logits bounded to [-1, 1], fixed reward scale, observations scaled
  by a warm-up estimate and that scale folded into the ACTOR's first layer so the saved zip takes raw obs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "scripts"))  # the organizer's scripts (train_vs_npcs)

from stable_baselines3 import A2C  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402
from train_vs_npcs import SAC_REWARD_SCALE, estimate_obs_scale, make_env_fns  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label")
    parser.add_argument("--algo", choices=("a2c", "tqc"), required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-ticks", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    set_random_seed(args.seed)

    if args.algo == "a2c":
        args.bound_logits = False
        venv = VecNormalize(DummyVecEnv(make_env_fns(args)), norm_obs=False, norm_reward=True)
        venv.action_space.seed(args.seed)
        model = A2C("MlpPolicy", venv, verbose=1)
        model.learn(total_timesteps=args.steps)
    else:
        from sb3_contrib import TQC

        args.bound_logits = True
        obs_scale, samples = estimate_obs_scale(args.seed)
        venv = DummyVecEnv(make_env_fns(args, reward_scale=SAC_REWARD_SCALE, obs_scale=obs_scale))
        venv.action_space.seed(args.seed)
        model = TQC(
            "MlpPolicy", venv, buffer_size=min(args.steps, 1_000_000), gradient_steps=-1, verbose=1
        )
        model.learn(total_timesteps=args.steps)
        inv = torch.as_tensor(1.0 / obs_scale, dtype=torch.float32)
        actor = model.policy.actor
        obs_t = torch.as_tensor(samples / obs_scale, dtype=torch.float32)
        with torch.no_grad():
            before = actor(obs_t, deterministic=True)
            actor.latent_pi[0].weight[:, : len(inv)] *= inv
            after = actor(torch.as_tensor(samples, dtype=torch.float32), deterministic=True)
            drift = float(torch.max(torch.abs(before - after) / torch.clamp(before.abs(), min=1.0)))
        print(f"obs scale folded into the actor (max prediction drift {drift:.1e})")
        if drift > 1e-4:
            raise RuntimeError(f"folding the obs scale changed predictions by {drift:.2e}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
