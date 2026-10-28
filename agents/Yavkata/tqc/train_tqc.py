"""TQC agent: SAC-style off-policy learner with distributional critics (sb3-contrib ``TQC``).

Trains one learning seat against the NPC field, using the same env, bounded logits, reward scale
and observation-scale handling as the SAC path of ``scripts/train_vs_npcs.py`` (see its docstring
for why). TQC's policy and critics are laid out differently from SAC's (the critic is a list of
quantile networks), so the obs-scale fold is redone here. The saved zip takes RAW observations.

Run from the repo root:
    python agents/Yavkata/tqc/train_tqc.py --steps 300000
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sb3_contrib import TQC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from train_vs_npcs import (  # noqa: E402
    SAC_REWARD_SCALE,
    estimate_obs_scale,
    make_env_fns,
)


def fold_obs_scale(model: TQC, obs_scale: npt.NDArray[np.float32]) -> None:
    """Bake ``obs / obs_scale`` into the first Linear layer of the actor and every critic."""
    inv = torch.as_tensor(1.0 / obs_scale, dtype=torch.float32)
    policy = model.policy
    critics = [*policy.critic.q_networks, *policy.critic_target.q_networks]
    with torch.no_grad():
        for first in [policy.actor.latent_pi[0], *(net[0] for net in critics)]:
            first.weight[:, : len(inv)] *= inv


def network_outputs(model: TQC, obs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Actor action plus the critic and target-critic quantiles, flattened, for drift checks."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        action = model.policy.actor(obs_t, deterministic=True)
        quantiles = [model.policy.critic(obs_t, action), model.policy.critic_target(obs_t, action)]
    return torch.cat([action, *(q.flatten(start_dim=1) for q in quantiles)], dim=1).numpy()


def train(args: argparse.Namespace) -> TQC:
    set_random_seed(args.seed)
    obs_scale, samples = estimate_obs_scale(args.seed)
    args.bound_logits = True  # off-policy: argmax logit dims bounded to [-1, 1]
    env_kwargs: dict[str, Any] = {"reward_scale": SAC_REWARD_SCALE, "obs_scale": obs_scale}
    venv = DummyVecEnv(make_env_fns(args, **env_kwargs))
    venv.action_space.seed(args.seed)
    model = TQC(
        "MlpPolicy",
        venv,
        buffer_size=min(args.steps, 1_000_000),
        gradient_steps=-1,
        verbose=1,
    )
    model.learn(total_timesteps=args.steps)
    before = network_outputs(model, samples / obs_scale)
    fold_obs_scale(model, obs_scale)
    after = network_outputs(model, samples)
    drift = float(np.max(np.abs(before - after) / np.maximum(1.0, np.abs(before))))
    if drift > 1e-5:
        raise RuntimeError(f"folding the obs scale changed predictions by {drift:.2e}")
    print(f"obs scale folded into the network (max prediction drift {drift:.1e})")
    return model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TQC agent against the NPC field.")
    parser.add_argument("--steps", type=int, default=300_000, help="env steps summed over copies")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-ticks", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("runs/tqc.zip"))
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
