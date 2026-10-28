"""Jack: RecurrentPPO (MlpLstmPolicy) trained against five NPCs (a full six-seat match).

Thin layer over ``scripts/train_vs_npcs.py``: it reuses that script's env (one expansion-masked
learning seat, homes reshuffled and a fresh world seed every episode, raw observations, profit
reward) and only adds what the example leaves out for RecurrentPPO: LSTM size flags, PPO
hyperparameters, a learning-rate schedule, periodic checkpoints and an optional evaluation match.

The policy is the stock ``MlpLstmPolicy`` so the organizer's ladder can load the zip. The
observation and action are left untouched (318-D in, 44-D out); the LSTM carries the memory of
what rivals did, which is the only place the noisy competitor block can be tracked over time.

Run from the repo root:
    python agents/orlin/jack/train.py jack --steps 20000 --out agents/orlin/jack/checkpoint.zip
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import numpy as np  # noqa: E402
import train_vs_npcs as base  # noqa: E402
from sb3_contrib import RecurrentPPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize  # noqa: E402


N_SEATS = 6  # a full match: Jack plus five opponents, one per published metro home


class SixSeatEnv(base.NpcFieldEnv):
    """Jack against five NPCs: the four geo archetypes plus one more drawn each episode.

    The arena only drives the four geo archetypes, so the fifth seat repeats one of them
    (chosen at random per episode). Each seat gets its own ``geo_archetypes()`` call because an
    archetype closure carries a tick counter that two seats must not share.
    """

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        result = super().reset(seed=seed, options=options)
        extra = str(self.np_random.choice(base.NPC_FIELD))
        names = (*base.NPC_FIELD, extra)
        self._npcs = [
            base.geo_archetypes(len(self._regions), self._regions)[name] for name in names
        ]
        return result


def make_env_fns(args: argparse.Namespace) -> list[Callable[[], Any]]:
    """One env per copy, each with its own seed (cloning one env would repeat worlds).

    ``--env given`` is the example env from ``train_vs_npcs.py`` untouched (5 seats, 4 NPCs);
    ``--env six`` is the six-seat variant above.
    """
    seeds = [
        int(s.generate_state(1)[0]) for s in np.random.SeedSequence(args.seed).spawn(args.n_envs)
    ]
    if args.env == "given":
        base.N_SEATS = 1 + len(base.NPC_FIELD)
        env_cls: type[base.NpcFieldEnv] = base.NpcFieldEnv
    else:
        base.N_SEATS = N_SEATS  # NpcFieldEnv.reset reads this module global to build the world
        env_cls = SixSeatEnv
    return [lambda s=s: Monitor(_with_reward(env_cls(s, episode_ticks=args.episode_ticks), args))
            for s in seeds]


def _with_reward(env: base.NpcFieldEnv, args: argparse.Namespace) -> base.NpcFieldEnv:
    """Train on a weighted reward (profit, revenue, market share, loyalty) instead of profit.

    The world z-normalizes each component for learning seats and returns the weighted sum, so
    the number the learner sees is not in currency units. Scoring is still profit.
    """
    if args.reward_weights is not None:
        profit, revenue, share, loyalty = args.reward_weights
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


def linear_schedule(start: float) -> Callable[[float], float]:
    """Learning rate that decays linearly from ``start`` to 0 over the run."""
    return lambda progress_remaining: progress_remaining * start


def tensorboard_dir(path: Path) -> str | None:
    """Log to TensorBoard only when it is installed; training must not depend on it."""
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        print("tensorboard not installed: skipping TensorBoard logging")
        return None
    return str(path)


def build_model(args: argparse.Namespace, venv: VecEnv, run_dir: Path) -> RecurrentPPO:
    if args.shared_lstm and args.enable_critic_lstm:
        raise SystemExit("--shared-lstm needs --no-critic-lstm (SB3 shares one LSTM with the actor)")
    policy_kwargs = {
        "lstm_hidden_size": args.lstm_hidden_size,
        "n_lstm_layers": args.n_lstm_layers,
        "shared_lstm": args.shared_lstm,
        "enable_critic_lstm": args.enable_critic_lstm,
    }
    lr = linear_schedule(args.lr) if args.lr_schedule == "linear" else args.lr
    return RecurrentPPO(
        "MlpLstmPolicy",
        venv,
        learning_rate=lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_dir(run_dir / "tb"),
        seed=args.seed,
        verbose=1,
    )


def build_callbacks(args: argparse.Namespace, run_dir: Path) -> list[BaseCallback]:
    callbacks: list[BaseCallback] = []
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(args.checkpoint_freq // args.n_envs, 1),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="jack",
            )
        )
    if args.eval_freq > 0:
        # A held-out env copy (own seed) plays full 1,000-tick matches; SB3's evaluate_policy
        # carries the LSTM state across the match and predicts deterministically, like scoring.
        eval_env = VecNormalize(
            DummyVecEnv(make_env_fns(_with(args, seed=args.seed + 10_000, n_envs=1))),
            norm_obs=False,
            norm_reward=False,
            training=False,
        )
        callbacks.append(
            EvalCallback(
                eval_env,
                n_eval_episodes=1,
                eval_freq=max(args.eval_freq // args.n_envs, 1),
                deterministic=True,
                best_model_save_path=str(run_dir / "best"),
                log_path=str(run_dir / "eval"),
            )
        )
    return callbacks


def _with(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    values = {**vars(args), **overrides}
    return argparse.Namespace(**values)


def train(args: argparse.Namespace) -> RecurrentPPO:
    set_random_seed(args.seed)
    run_dir = Path("runs") / args.label
    venv: VecEnv = DummyVecEnv(make_env_fns(args))
    # Reward scaling for training only; observations stay raw because scoring applies no stats.
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=args.gamma)
    if args.resume is not None:
        # Continue from a saved zip: weights and LSTM architecture come from the file, and the
        # step counter carries on so ``--steps`` means "this many more".
        model = RecurrentPPO.load(args.resume, env=venv, tensorboard_log=None)
        model.learn(
            total_timesteps=args.steps,
            callback=build_callbacks(args, run_dir),
            reset_num_timesteps=False,
        )
        return model
    model = build_model(args, venv, run_dir)
    model.learn(total_timesteps=args.steps, callback=build_callbacks(args, run_dir))
    return model


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Jack (RecurrentPPO) against five NPCs (a full six-seat match).")
    parser.add_argument("label", help="run name; logs and checkpoints go under runs/<label>/")
    parser.add_argument("--steps", type=int, default=20_000, help="total env steps over all copies")
    parser.add_argument("--env", choices=("six", "given"), default="six")
    parser.add_argument("--resume", type=Path, help="continue training from this zip")
    parser.add_argument(
        "--reward-weights",
        type=lambda text: tuple(float(part) for part in text.split(",")),
        help="profit,revenue,market_share,loyalty (sum 1.0) for a weighted training reward; "
        "omit for profit-only",
    )
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-ticks", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("agents/orlin/jack/checkpoint.zip"))
    # Hyperparameters (RecurrentPPO defaults, except gae_lambda 0.98 as in the reference recipes).
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-schedule", choices=("constant", "linear"), default="constant")
    parser.add_argument("--n-steps", type=int, default=128, help="rollout length per env copy")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.98)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    # LSTM architecture (stored in the zip, so the stock policy class still loads).
    parser.add_argument("--lstm-hidden-size", type=int, default=256)
    parser.add_argument("--n-lstm-layers", type=int, default=1)
    parser.add_argument("--shared-lstm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--critic-lstm", dest="enable_critic_lstm", action=argparse.BooleanOptionalAction, default=True
    )
    # Callbacks (env steps over all copies; 0 disables).
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=0, help="a full 1,000-tick match is slow")
    args = parser.parse_args(argv)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model = train(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
