"""RecurrentPPO (LSTM) agent for the 25-region balanced arena.

One learning seat (expansion-masked) plays the self-check NPC field; the env is the same
``NpcFieldEnv`` as ``scripts/train_vs_npcs.py`` (shuffled metro homes and a fresh world seed per
episode, truncated at the 1000-tick match length).

Interface, as the simulator defines it (read from ``retail_simulator.core.schema``, not hard-coded):

* Observation: flat vector, 318 dims in the 25-region arena (the 112 in older docs is a smaller
  layout). It is split into SELF / MARKET / COMPETITOR groups by ``observation_schema``; the
  competitor block is noised and sharpened by the ``research`` lever.
* Action: 44 dims = 9 continuous levers (price_index, marketing, assortment, promotion, research,
  service_level, loyalty_spend, wage_spend, warehouse_invest) + a 32-logit ``expansion`` block and
  a 3-logit ``automation`` block, both argmax-decoded. Expansion is masked for our seat; automation
  is a one-way tier upgrade.
* Reward: the env's per-tick reward is the seat's profit with wage write-offs and warehouse
  penalties already subtracted, so the sum of rewards is the competition's ``total_profit``.

Scoring constraints that shape this script (README, "Submit"): the zip is loaded by plain
sb3-contrib, deterministic, with NO VecNormalize statistics and no custom classes. So

* training sees ``obs / scale`` (raw observations span cash ~1e5 down to shares ~1e-2), and
  ``fold_obs_scale`` bakes that division into the LSTM input weights afterwards: the saved zip eats
  raw observations and is verified to act identically.
* rewards are scaled for training only, via ``VecNormalize(norm_obs=False)``.
* the policy is the stock ``MlpLstmPolicy``; the LSTM state is carried across the whole match.

Run from the repo root:
    python agents/Yavkata/recurrent-ppo/train_recurrent_ppo.py --steps 1000000
    python agents/Yavkata/recurrent-ppo/train_recurrent_ppo.py --eval-only checkpoint.zip
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from train_vs_npcs import NpcFieldEnv  # noqa: E402

from retail_simulator.core.encoding import flatten_action_space_layout  # noqa: E402
from retail_simulator.core.schema import (  # noqa: E402
    ObservationGroup,
    observation_schema,
)
from retail_simulator.envs.arena import arena_balanced_config  # noqa: E402

OBS_SCALE_WARMUP_TICKS = 3000
REWARD_GAMMA = 0.995  # profit accrues over a 1000-tick match, so look far ahead


# --------------------------------------------------------------------------- observation layout


@dataclass(frozen=True)
class ObsLayout:
    """Index sets of the flat observation, taken from the schema."""

    n_dims: int
    self_idx: npt.NDArray[np.intp]
    market_idx: npt.NDArray[np.intp]
    competitor_idx: npt.NDArray[np.intp]
    names: tuple[str, ...]

    @classmethod
    def build(cls) -> ObsLayout:
        n_regions = len(arena_balanced_config().demand.regions)
        fields = observation_schema(n_regions)

        def idx(group: ObservationGroup) -> npt.NDArray[np.intp]:
            return np.array([i for i, f in enumerate(fields) if f.group is group], dtype=np.intp)

        return cls(
            n_dims=len(fields),
            self_idx=idx(ObservationGroup.SELF),
            market_idx=idx(ObservationGroup.MARKET),
            competitor_idx=idx(ObservationGroup.COMPETITOR),
            names=tuple(f.name for f in fields),
        )

    def summary(self) -> str:
        return (
            f"observation: {self.n_dims} dims = {len(self.self_idx)} Self + "
            f"{len(self.market_idx)} Market + {len(self.competitor_idx)} Competitor"
        )


def estimate_obs_scale(layout: ObsLayout, seed: int, headroom: float) -> npt.NDArray[np.float32]:
    """Per-dim divisor: peak |obs| over a random rollout, times ``headroom`` where unbounded.

    Dims whose peak stays <= 1 (shares, loyalty, utilisation, presence flags) are already unit
    scale and keep divisor 1. Dims that grow with play (cash, profit, revenue, demand, wage) get
    ``headroom`` extra room, because a trained policy earns far more than a random one.
    """
    env = NpcFieldEnv(seed, bound_logits=True)
    env.action_space.seed(seed)
    obs, _ = env.reset()
    peak = np.abs(obs)
    for _ in range(OBS_SCALE_WARMUP_TICKS):
        obs, _, _, truncated, _ = env.step(env.action_space.sample())
        peak = np.maximum(peak, np.abs(obs))
        if truncated:
            obs, _ = env.reset()
    assert len(peak) == layout.n_dims, "observation size does not match the schema"
    scale = np.where(peak > 1.0, peak * headroom, 1.0)
    return scale.astype(np.float32)


def fold_obs_scale(model: RecurrentPPO, scale: npt.NDArray[np.float32]) -> None:
    """Bake ``obs / scale`` into the LSTM input weights, in place (actor and critic).

    Both LSTMs read the raw observation through ``weight_ih_l0`` (FlattenExtractor has no
    parameters), so dividing its columns by ``scale`` is exactly equivalent to scaling the input.
    """
    inv = torch.as_tensor(1.0 / scale, dtype=torch.float32)
    policy = model.policy
    lstms = [policy.lstm_actor]
    if getattr(policy, "lstm_critic", None) is not None:
        lstms.append(policy.lstm_critic)
    with torch.no_grad():
        for lstm in lstms:
            lstm.weight_ih_l0[:, : len(inv)] *= inv


# --------------------------------------------------------------------------- action head


def init_action_head(model: RecurrentPPO) -> None:
    """Start the Gaussian action mean at each lever's default instead of 0.

    Actions are clipped to the lever bounds, so a zero mean leaves price_index (0.5..2.0) pinned
    at its floor half the time and service_level at 0. Defaults (price 1.0, service 1.0, spend
    levers 0.0, logits 0.0) are a sane place to start exploring from.
    """
    bias = torch.zeros(model.policy.action_net.bias.shape[0])
    for slot in flatten_action_space_layout().levers:
        if slot.width == 1:  # continuous lever
            bias[slot.offset] = float(slot.spec.default)
    with torch.no_grad():
        model.policy.action_net.bias.copy_(bias)


# --------------------------------------------------------------------------- training


def _make_env(seed: int, ticks: int, obs_scale: npt.NDArray[np.float32]) -> Monitor:
    """Top-level (picklable) env factory, so SubprocVecEnv works under Windows' spawn."""
    return Monitor(NpcFieldEnv(seed, episode_ticks=ticks, obs_scale=obs_scale))


def make_env_fns(args: argparse.Namespace, obs_scale: npt.NDArray[np.float32]) -> list:
    """One factory per env copy, each with its own seed (cloning one env would repeat worlds)."""
    seeds = [
        int(s.generate_state(1)[0]) for s in np.random.SeedSequence(args.seed).spawn(args.n_envs)
    ]
    return [partial(_make_env, s, args.episode_ticks, obs_scale) for s in seeds]


class BestSnapshot(BaseCallback):
    """Score the policy on fixed held-out matches every ``every`` steps; keep the best weights.

    RL runs are noisy, so the last update is often not the best one. Scoring happens on the
    scaled-input network (before the fold), on seeds the final evaluation does not use.
    """

    def __init__(
        self, obs_scale: npt.NDArray[np.float32], seeds: list[int], ticks: int, every: int
    ) -> None:
        super().__init__()
        self._obs_scale, self._seeds, self._ticks, self._every = obs_scale, seeds, ticks, every
        self._next = every
        self.best_score = -np.inf
        self.best_state: dict | None = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps >= self._next:
            self._next += self._every
            self.score()

    def score(self) -> None:
        totals = evaluate(self.model, self._seeds, self._ticks, self._obs_scale, quiet=True)
        mean = float(np.mean(totals))
        better = mean > self.best_score
        print(f"[snapshot] step {self.num_timesteps}: held-out profit {mean:,.0f}"
              f"{'  <- best' if better else ''}")
        if better:
            self.best_score = mean
            self.best_state = copy.deepcopy(self.model.policy.state_dict())

    def _on_training_end(self) -> None:
        self.score()  # the final policy is a candidate too


def train(args: argparse.Namespace, layout: ObsLayout) -> RecurrentPPO:
    set_random_seed(args.seed)
    obs_scale = estimate_obs_scale(layout, args.seed, args.headroom)

    env_fns = make_env_fns(args, obs_scale)
    venv = SubprocVecEnv(env_fns) if args.subproc else DummyVecEnv(env_fns)
    # Reward scaling only. Observations are scaled inside the env and folded into the net later.
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=REWARD_GAMMA)
    venv.action_space.seed(args.seed)

    model = RecurrentPPO(
        "MlpLstmPolicy",
        venv,
        learning_rate=lambda progress: args.lr * progress,  # linear decay to 0
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=REWARD_GAMMA,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        max_grad_norm=0.5,
        policy_kwargs={
            "lstm_hidden_size": args.lstm_hidden,
            "n_lstm_layers": 1,
            "enable_critic_lstm": True,
            "log_std_init": -1.0,
        },
        verbose=1,
    )
    init_action_head(model)
    selector = BestSnapshot(
        obs_scale,
        seeds=[5000 + i for i in range(args.select_seeds)],
        ticks=args.episode_ticks,
        every=args.snapshot_every,
    )
    model.learn(total_timesteps=args.steps, callback=selector)
    venv.close()
    if selector.best_state is not None:
        model.policy.load_state_dict(selector.best_state)
        print(f"kept the best snapshot (held-out profit {selector.best_score:,.0f})")

    # Fold the scaling in and prove the raw-observation zip behaves like the scaled-input net.
    probe = _probe_observations(obs_scale, args.seed)
    before = _deterministic_actions(model, probe / obs_scale)
    fold_obs_scale(model, obs_scale)
    after = _deterministic_actions(model, probe)
    drift = float(np.max(np.abs(before - after) / np.maximum(1.0, np.abs(before))))
    if drift > 1e-4:
        raise RuntimeError(f"folding the obs scale changed actions by {drift:.2e}")
    print(f"obs scale folded into the LSTM (max action drift {drift:.1e})")
    return model


def _probe_observations(obs_scale: npt.NDArray[np.float32], seed: int) -> npt.NDArray[np.float32]:
    """A batch of plausible raw observations (uniform up to 1x the scale) for the drift check."""
    rng = np.random.default_rng(seed)
    return (rng.uniform(0.0, 1.0, size=(32, len(obs_scale))) * obs_scale).astype(np.float32)


def _deterministic_actions(model: RecurrentPPO, obs: npt.NDArray[np.float32]) -> npt.NDArray:
    """Mean actions for a batch of single-step episodes (fresh LSTM state each row)."""
    starts = np.ones(len(obs), dtype=bool)
    actions, _ = model.predict(obs, state=None, episode_start=starts, deterministic=True)
    return actions


# --------------------------------------------------------------------------- evaluation


def evaluate(
    model: RecurrentPPO,
    seeds: list[int],
    ticks: int,
    obs_scale: npt.NDArray[np.float32] | None = None,
    quiet: bool = False,
) -> list[float]:
    """Play one full match per seed as the scorer does: deterministic, LSTM state kept.

    Observations are raw unless ``obs_scale`` is given (a net whose scale is not yet folded in).
    Returns each match's cumulative profit (the sum of the per-tick, post-penalty rewards).
    """
    totals = []
    for seed in seeds:
        env = NpcFieldEnv(seed, episode_ticks=ticks, obs_scale=obs_scale)
        obs, info = env.reset()
        state, start, total, truncated = None, np.array([True]), 0.0, False
        while not truncated:
            action, state = model.predict(
                obs[None, :], state=state, episode_start=start, deterministic=True
            )
            obs, reward, _, truncated, _ = env.step(action[0])
            start = np.array([False])
            total += reward
        totals.append(total)
        if not quiet:
            print(f"  seed {seed} homes {info['homes']}: total_profit {total:,.0f}")
    return totals


# --------------------------------------------------------------------------- cli


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train / evaluate the RecurrentPPO arena agent.")
    p.add_argument("--steps", type=int, default=1_000_000, help="total env steps over all copies")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episode-ticks", type=int, default=1000)
    p.add_argument("--n-steps", type=int, default=512, help="rollout length per env copy")
    p.add_argument(
        "--subproc",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run env copies in separate processes (default) instead of one",
    )
    p.add_argument("--snapshot-every", type=int, default=25_000, help="steps between held-out scores")
    p.add_argument("--select-seeds", type=int, default=3, help="held-out matches per snapshot score")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--lstm-hidden", type=int, default=128)
    p.add_argument(
        "--headroom", type=float, default=4.0, help="extra scale room for unbounded obs dims"
    )
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "checkpoint.zip")
    p.add_argument("--eval-seeds", type=int, default=3, help="matches to play after training")
    p.add_argument("--eval-only", type=Path, help="skip training; evaluate this checkpoint")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    layout = ObsLayout.build()
    print(layout.summary())

    if args.eval_only:
        model = RecurrentPPO.load(args.eval_only, device="cpu")
    else:
        model = train(args, layout)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.out)
        print(f"saved {args.out}")
        # Reload from disk: this is what the scorer will do.
        model = RecurrentPPO.load(args.out, device="cpu")

    print("evaluation (raw observations, deterministic, full matches):")
    totals = evaluate(model, [1000 + i for i in range(args.eval_seeds)], args.episode_ticks)
    print(f"mean total_profit {np.mean(totals):,.0f} (min {min(totals):,.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
