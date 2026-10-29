"""Jacky-T: TQC (MlpPolicy, sb3-contrib) trained against the NPC field with a weighted reward.

Same example env as ``scripts/train_vs_npcs.py`` (one expansion-masked learning seat against the
four geo archetypes, homes reshuffled and a fresh world seed every episode), trained the way that
script trains SAC, because TQC is SAC-style and off-policy:

* the 35 argmax-decoded logit dims are bounded to [-1, 1] (``bound_logits``); scoring takes the
  argmax, so the saved policy still scores;
* observations are divided by a per-dim scale for training, and the scale is folded into the first
  layers of the actor and the critics afterwards, so the saved zip takes the RAW 318-D observation
  the scorer feeds it;
* the replay buffer is capped at the run length (the default 1e6 transitions peaks near 11 GB).

The training reward is the world's z-normalized weighted sum (profit, revenue, market share,
loyalty). It is already about unit scale, so no extra reward scaling is applied. Scoring is still
profit.

Run from the repo root:
    python agents/misho/jacky-t/train.py jacky_t --steps 171008
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import numpy as np  # noqa: E402
import numpy.typing as npt  # noqa: E402
import torch  # noqa: E402
import train_vs_npcs as base  # noqa: E402
from sb3_contrib import TQC  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv  # noqa: E402


def weighted(env: base.NpcFieldEnv, weights: tuple[float, float, float, float]) -> base.NpcFieldEnv:
    """Train on profit/revenue/market-share/loyalty weights instead of profit alone."""
    profit, revenue, share, loyalty = weights
    env._config = replace(
        env._config,
        reward=replace(
            env._config.reward,
            mode="weighted",
            weight_profit=profit,
            weight_revenue=revenue,
            weight_market_share=share,
            weight_loyalty=loyalty,
        ),
    )
    return env


def make_env_fns(args: argparse.Namespace, obs_scale: npt.NDArray[np.float32]) -> list[Any]:
    """One env per copy, each with its own seed (cloning one env would repeat worlds)."""
    seeds = [
        int(s.generate_state(1)[0]) for s in np.random.SeedSequence(args.seed).spawn(args.n_envs)
    ]
    return [
        lambda s=s: Monitor(
            weighted(
                base.NpcFieldEnv(
                    s,
                    episode_ticks=args.episode_ticks,
                    bound_logits=True,
                    obs_scale=obs_scale,
                ),
                args.reward_weights,
            )
        )
        for s in seeds
    ]


def fold_obs_scale(model: TQC, obs_scale: npt.NDArray[np.float32]) -> None:
    """Bake ``obs / obs_scale`` into the first Linear layer of the actor and critics, in place."""
    inv = torch.as_tensor(1.0 / obs_scale, dtype=torch.float32)
    policy = model.policy
    critics = [*policy.critic.q_networks, *policy.critic_target.q_networks]
    with torch.no_grad():
        for first in [policy.actor.latent_pi[0], *(net[0] for net in critics)]:
            first.weight[:, : len(inv)] *= inv


def network_outputs(model: TQC, obs: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Actor action plus the critics' quantiles, for the fold drift check."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32)
    with torch.no_grad():
        action = model.policy.actor(obs_t, deterministic=True)
        quantiles = model.policy.critic(obs_t, action).flatten(start_dim=1)
    return torch.cat([action, quantiles], dim=1).numpy()


def train(args: argparse.Namespace) -> TQC:
    set_random_seed(args.seed)
    obs_scale, samples = base.estimate_obs_scale(args.seed)
    venv: VecEnv = DummyVecEnv(make_env_fns(args, obs_scale))
    # SB3 does not seed the action space; the random warm-up actions are drawn from it.
    venv.action_space.seed(args.seed)
    model = TQC(
        "MlpPolicy",
        venv,
        learning_rate=args.lr,
        buffer_size=min(args.steps, 1_000_000),
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gradient_steps=args.gradient_steps,  # per rollout of n_envs steps (default 2: UTD 0.5)
        seed=args.seed,
        verbose=1,
    )
    # Periodic snapshots keep an interrupted run usable. They are NOT folded (the obs scale is
    # still outside the network), so the scale is saved beside them; ``--finalize`` folds one.
    ckpt_dir = Path("runs") / args.label / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.save(ckpt_dir / "obs_scale.npy", obs_scale)
    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                save_path=str(ckpt_dir),
                name_prefix="jacky_t",
            )
        )
    model.learn(total_timesteps=args.steps, callback=callbacks)
    before = network_outputs(model, samples / obs_scale)
    fold_obs_scale(model, obs_scale)
    after = network_outputs(model, samples)
    drift = float(np.max(np.abs(before - after) / np.maximum(1.0, np.abs(before))))
    if drift > 1e-4:
        raise RuntimeError(f"folding the obs scale changed predictions by {drift:.2e}")
    print(f"obs scale folded into the network (max prediction drift {drift:.1e})")
    return model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Jacky-T (TQC) against the NPC field.")
    parser.add_argument("label", help="run name")
    parser.add_argument("--steps", type=int, default=171_008, help="env steps over all copies")
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-ticks", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("agents/misho/jacky-t/checkpoint.zip"))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-starts", type=int, default=4_000)
    parser.add_argument("--gradient-steps", type=int, default=2)
    parser.add_argument("--checkpoint-freq", type=int, default=20_000, help="0 disables")
    parser.add_argument(
        "--finalize", type=Path, help="fold the obs scale into this periodic snapshot and save --out"
    )
    parser.add_argument(
        "--reward-weights",
        type=lambda text: tuple(float(part) for part in text.split(",")),
        default=(0.3, 0.3, 0.3, 0.1),
        help="profit,revenue,market_share,loyalty (sum 1.0)",
    )
    return parser.parse_args(argv)


def stock_schedule(model: TQC) -> None:
    """Make the saved learning-rate schedule a stock SB3 object.

    ``--from-actor`` runs left ``lr_schedule`` as a ``constant_fn`` closure, which the zip stores by value
    (cloudpickle). Battle rooms reject such checkpoints: "carries custom code ... accept only stock
    checkpoints". The scorer never reads the schedule, so a stock ``FloatSchedule`` replaces it.
    """
    from stable_baselines3.common.utils import FloatSchedule

    model.lr_schedule = FloatSchedule(float(model.learning_rate) if not callable(model.learning_rate) else 3e-4)


def finalize(snapshot: Path, out: Path) -> None:
    """Fold the obs scale saved beside a periodic snapshot into it and save a scorable zip."""
    model = TQC.load(snapshot, device="cpu")
    fold_obs_scale(model, np.load(snapshot.parent / "obs_scale.npy"))
    stock_schedule(model)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"folded {snapshot} -> {out}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.finalize is not None:
        finalize(args.finalize, args.out)
        return 0
    model = train(args)
    stock_schedule(model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
