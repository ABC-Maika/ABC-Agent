"""SAC league training: a six-seat match against the other teammates' real checkpoints, no NPCs.

Seats: the learner plus Orcho, Krischo, Misho's ``jack``, Yavkata's ``recurrent-ppo`` and Yavkata's own
warm-started SAC snapshot (``checkpoint_bc_warmstart.zip``). That is exactly six seats, like the final
tournament. Homes are shuffled each episode and the opponents' seats are permuted.

Same recipe as ``train_sac_bc.py`` (behaviour-cloned warm start from the scripted orcho-like policy, critic
warm-up, fixed small entropy coefficient, small opening-race penalty), but the demonstrations and all
training run in this league field. Every env copy loads its OWN predict closures: a recurrent opponent keeps
its LSTM state inside the closure, so sharing one across the env copies would mix their states.
Intermediate raw-observation checkpoints are written every ``--save-every`` steps.

Run from the repo root:
    python agents/Yavkata/sac/train_sac_league.py --steps 1000000 --out agents/Yavkata/sac/checkpoint_league.zip
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_sac_bc as bc  # noqa: E402  (also imports train_sac_field and train_vs_npcs)
import train_vs_npcs as base  # noqa: E402
from retail_simulator.envs.ladder import _materialize_predicts  # noqa: E402
from retail_simulator.envs.spaces import build_action_space  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OPPONENT_ZIPS: dict[str, Path] = {
    "Orcho": REPO / "agents/orcho/market-share/checkpoint.zip",
    "Krischo": REPO / "agents/Krischo/rppo-lstm/checkpoint.zip",
    "Jack": REPO / "agents/misho/jack/checkpoint.zip",
    "YavkataRPPO": REPO / "agents/Yavkata/recurrent-ppo/checkpoint.zip",
    "YavkataSAC": REPO / "agents/Yavkata/sac/checkpoint_bc_warmstart.zip",
}
N_SEATS = 1 + len(OPPONENT_ZIPS)
ORIGINAL_ENV = base.NpcFieldEnv  # kept before main() swaps the name


class LeagueEnv(bc.OpeningEnv):
    """Learner in seat 0 against the five real checkpoints; six seats, no NPCs."""

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        out = ORIGINAL_ENV.reset(self, seed=seed, options=options)  # builds the 6-seat world
        # Fresh closures per episode and per env copy: the LSTM state lives in them.
        predicts = _materialize_predicts(
            {name: str(path) for name, path in OPPONENT_ZIPS.items()},
            action_space=build_action_space(),
            config=self._config,
        )
        names = list(OPPONENT_ZIPS)
        order = self.np_random.permutation(len(names))
        self._npcs = [predicts[names[i]] for i in order]
        return out


class SnapshotCallback(BaseCallback):
    def __init__(self, path: Path, every: int, obs_scale: Any, samples: Any) -> None:
        super().__init__()
        self.path, self.every, self.obs_scale, self.samples = path, every, obs_scale, samples
        self.next_save = every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_save:
            self.next_save += self.every
            stem = self.path.stem + f"_{self.num_timesteps // 1000}k"
            bc.save_folded_copy(self.model, self.path.with_name(stem + ".zip"), self.obs_scale, self.samples)
        return True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAC league training vs the teammates' checkpoints, six seats.")
    p.add_argument("--steps", type=int, default=1_000_000, help="env steps summed over copies")
    p.add_argument("--demo-steps", type=int, default=32_000)
    p.add_argument("--clone-steps", type=int, default=4_000)
    p.add_argument("--critic-steps", type=int, default=3_000)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--buffer", type=int, default=300_000, help="replay buffer size (1M peaks near 11 GB)")
    p.add_argument("--save-every", type=int, default=100_000)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/sac_league.zip"))
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base.N_SEATS = N_SEATS  # NpcFieldEnv.reset seats this many (read at call time)
    bc.OpeningEnv = LeagueEnv  # the demonstrator rollouts use it too
    base.NpcFieldEnv = LeagueEnv  # make_env_fns and estimate_obs_scale resolve it by name
    set_random_seed(args.seed)
    obs_scale, samples = base.estimate_obs_scale(args.seed)
    train_args = argparse.Namespace(
        seed=args.seed, n_envs=args.n_envs, episode_ticks=1000, bound_logits=True, reward="profit"
    )
    venv = DummyVecEnv(base.make_env_fns(train_args, reward_scale=base.SAC_REWARD_SCALE, obs_scale=obs_scale))
    venv.action_space.seed(args.seed)
    model = SAC(
        "MlpPolicy", venv, buffer_size=min(args.demo_steps + args.steps, args.buffer), gradient_steps=-1,
        ent_coef=args.ent_coef, learning_starts=0, verbose=1,
    )
    bc.fill_buffer_with_demos(model, obs_scale, args.demo_steps, args.seed)
    bc.clone_actor(model, args.clone_steps)
    bc.warm_critics(model, args.critic_steps, args.ent_coef)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    bc.save_folded_copy(model, args.out.with_name(args.out.stem + "_warmstart.zip"), obs_scale, samples)
    callback = SnapshotCallback(args.out, args.save_every, obs_scale, samples)
    model.learn(total_timesteps=args.steps, reset_num_timesteps=True, callback=callback)
    bc.fold_and_check(model, obs_scale, samples)
    model.save(args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
