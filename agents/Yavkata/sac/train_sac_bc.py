"""SAC with a behaviour-cloning warm start, trained in the Orcho/Krischo field (``COUNTER_STRATEGY.md`` s.7).

Why: a SAC trained from scratch loses the cash race in ticks 2-3 (promotion ~0, price ~1.2, service ~0.05
against Orcho's 1.0 / 2.0 / 1.0), stays at 3 stores and bleeds. The scripted orcho-like policy in the same
seat earns +874k, so we start from it:
  1. roll the scripted "orcho-like" demonstrator (cap 12, wage from staff happiness) in the learner seat and
     fill the replay buffer with those transitions;
  2. clone it into the actor (MSE on the squashed action), then shrink the actor's exploration noise;
  3. warm the critics with TD updates on the same buffer, so the first SAC actor steps do not wreck the clone;
  4. train SAC with a fixed small entropy coefficient and a small reward penalty while the opening cash race
     is on (ticks 1-60, cash < one store, fewer than 5 stores).
The saved zip takes RAW observations (obs scale folded into the first layers, as in ``train_vs_npcs.py``).

Run from the repo root:
    python agents/Yavkata/sac/train_sac_bc.py --steps 150000 --out agents/Yavkata/sac/checkpoint_bc.zip
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
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_sac_field as field  # noqa: E402  (also puts scripts/ on sys.path)
import train_vs_npcs as base  # noqa: E402
from retail_simulator.core.encoding import encode_action  # noqa: E402
from retail_simulator.core.schema import observation_schema  # noqa: E402
from retail_simulator.envs.geo_archetypes import _structured_action  # noqa: E402

OPENING_TICKS = 60
OPENING_PENALTY = 0.005  # reward units (after the 1e-3 scale) per tick of the cash race
DEMO_NOISE = 0.03  # std of gaussian noise on the demonstrator's squashed action


class OpeningEnv(field.FieldEnv):
    """``FieldEnv`` plus a small penalty while the opening cash race is lost."""

    def step(self, action: npt.NDArray[np.float32]):
        obs, reward, terminated, truncated, info = super().step(action)
        raw = self._obs[self._agents[0]]
        idx = _obs_index(self._config)
        if (
            self._tick <= OPENING_TICKS
            and raw[idx["cash"]] < field.EXPANSION_CAPEX
            and raw[idx["stores"]] < 5
        ):
            reward -= OPENING_PENALTY
        return obs, reward, terminated, truncated, info


_INDEX: dict[str, int] | None = None


def _obs_index(config: Any) -> dict[str, int]:
    global _INDEX
    if _INDEX is None:
        n = len(config.demand.regions)
        _INDEX = {f.name: i for i, f in enumerate(observation_schema(n))}
    return _INDEX


def orcho_demo(config: Any, cap: int, order: Sequence[int]):
    """Scripted orcho-like closure; wage follows the recipe ``min(1, 0.45 + 4 * (1 - happiness))``."""
    n = len(config.demand.regions)
    idx = _obs_index(config)

    def predict(obs: npt.NDArray[np.float32], *, episode_start: bool = False, **_: Any):
        target = 0
        if obs[idx["stores"]] < cap and obs[idx["cash"]] >= field.EXPANSION_CAPEX:
            target = next((r for r in order if obs[idx[f"stores_region_{r}"]] == 0), 0)
        wage = min(1.0, 0.45 + 4.0 * (1.0 - float(obs[idx["employee_happiness"]])))
        return encode_action(
            _structured_action(
                price=2.0, marketing=1.0, assortment=1.0, promotion=1.0, expansion_choice=target,
                research=0.2, service_level=1.0, automation_tier=0, loyalty_spend=0.0,
                n_regions=n, wage_spend=wage,
            )
        )

    return predict


def fill_buffer_with_demos(model: SAC, obs_scale: npt.NDArray[np.float32], n_steps: int, seed: int) -> None:
    """Play the demonstrator in the learner seat and store its transitions in ``model.replay_buffer``."""
    policy = model.policy
    space = model.action_space
    rng = np.random.default_rng(seed)
    N_DEMO_ENVS = model.n_envs  # the buffer wants one row per training env
    envs = [
        OpeningEnv(seed + 1000 + i, bound_logits=True, reward_scale=base.SAC_REWARD_SCALE, obs_scale=obs_scale)
        for i in range(N_DEMO_ENVS)
    ]
    demos: list[Any] = [None] * N_DEMO_ENVS
    obs = np.zeros((N_DEMO_ENVS, model.observation_space.shape[0]), dtype=np.float32)

    def start(i: int) -> None:
        o, _ = envs[i].reset()
        obs[i] = o
        regions = envs[i]._regions
        order = sorted(range(1, len(regions)), key=lambda r: (-regions[r].base_regional_demand, r))
        demos[i] = orcho_demo(envs[i]._config, cap=12, order=order)

    for i in range(N_DEMO_ENVS):
        start(i)
    for _ in range(n_steps // N_DEMO_ENVS):
        scaled = np.zeros((N_DEMO_ENVS, space.shape[0]), dtype=np.float32)
        next_obs = np.zeros_like(obs)
        rewards = np.zeros(N_DEMO_ENVS, dtype=np.float32)
        finished = []
        for i, env in enumerate(envs):
            raw = env._obs[env._agents[0]]
            action = np.clip(demos[i](raw, episode_start=env._tick == 0), space.low, space.high)
            sa = np.clip(policy.scale_action(action) + rng.normal(0.0, DEMO_NOISE, space.shape[0]), -1.0, 1.0)
            scaled[i] = sa
            o, r, _, truncated, _ = env.step(policy.unscale_action(sa).astype(np.float32))
            next_obs[i], rewards[i] = o, r
            if truncated:
                finished.append(i)
        model.replay_buffer.add(
            obs.copy(), next_obs.copy(), scaled, rewards, np.zeros(N_DEMO_ENVS, dtype=bool),
            [{} for _ in range(N_DEMO_ENVS)],
        )
        obs[:] = next_obs
        for i in finished:  # the transition above already stored the final obs; now start the next episode
            start(i)
    print(f"demonstrator transitions in the replay buffer: {model.replay_buffer.size() * N_DEMO_ENVS}")


def clone_actor(model: SAC, steps: int, batch_size: int = 256) -> float:
    """Behaviour-clone the buffer's actions into the actor mean, then shrink the exploration noise."""
    actor = model.policy.actor
    params = [p for n, p in actor.named_parameters() if not n.startswith("log_std")]
    opt = torch.optim.Adam(params, lr=1e-3)
    loss_value = float("nan")
    for step in range(steps):
        data = model.replay_buffer.sample(batch_size)
        loss = torch.nn.functional.mse_loss(actor(data.observations, deterministic=True), data.actions)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_value = float(loss)
        if step % 500 == 0:
            print(f"  clone step {step}: mse {loss_value:.5f}", flush=True)
    with torch.no_grad():
        actor.log_std.weight.mul_(0.1)
        actor.log_std.bias.fill_(-2.5)  # std ~0.08 in the squashed space
    return loss_value


def warm_critics(model: SAC, steps: int, ent_coef: float, batch_size: int = 256) -> None:
    """TD updates on the demonstration buffer for the critics only (the actor is left as cloned)."""
    policy = model.policy
    gamma, tau = model.gamma, model.tau
    for step in range(steps):
        data = model.replay_buffer.sample(batch_size)
        with torch.no_grad():
            next_actions, next_logp = policy.actor.action_log_prob(data.next_observations)
            next_q = torch.cat(policy.critic_target(data.next_observations, next_actions), dim=1)
            next_q = next_q.min(dim=1, keepdim=True).values - ent_coef * next_logp.reshape(-1, 1)
            target = data.rewards + (1 - data.dones) * gamma * next_q
        current = policy.critic(data.observations, data.actions)
        loss = 0.5 * sum(torch.nn.functional.mse_loss(q, target) for q in current)
        policy.critic.optimizer.zero_grad()
        loss.backward()
        policy.critic.optimizer.step()
        with torch.no_grad():
            for p, tp in zip(policy.critic.parameters(), policy.critic_target.parameters()):
                tp.data.mul_(1 - tau).add_(p.data, alpha=tau)
        if step % 1000 == 0:
            print(f"  critic step {step}: td loss {float(loss):.4f}", flush=True)


def fold_and_check(model: SAC, obs_scale: npt.NDArray[np.float32], samples: npt.NDArray[np.float32]) -> None:
    before = base.network_outputs(model, samples / obs_scale)
    base.fold_obs_scale(model, obs_scale)
    after = base.network_outputs(model, samples)
    drift = float(np.max(np.abs(before - after) / np.maximum(1.0, np.abs(before))))
    if drift > 1e-5:
        raise RuntimeError(f"folding the obs scale changed predictions by {drift:.2e}")
    print(f"obs scale folded into the network (max prediction drift {drift:.1e})")


def save_folded_copy(model: SAC, path: Path, obs_scale: npt.NDArray[np.float32], samples: npt.NDArray[np.float32]) -> None:
    """Save a raw-observation copy of the current policy without touching the model that keeps training."""
    tmp = path.with_suffix(".tmp.zip")
    model.save(tmp)
    copy = SAC.load(tmp, device="cpu")
    fold_and_check(copy, obs_scale, samples)
    copy.save(path)
    tmp.unlink()
    print(f"saved {path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAC with a behaviour-cloning warm start in the Orcho/Krischo field.")
    p.add_argument("--steps", type=int, default=150_000, help="SAC env steps after the warm start")
    p.add_argument("--demo-steps", type=int, default=32_000)
    p.add_argument("--clone-steps", type=int, default=4_000)
    p.add_argument("--critic-steps", type=int, default=3_000)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--p-real", type=float, default=0.6, help="chance an Orcho/Krischo seat is the real checkpoint")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("runs/sac_bc.zip"))
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    field.P_REAL = args.p_real
    base.NpcFieldEnv = OpeningEnv  # make_env_fns and estimate_obs_scale resolve it by name
    set_random_seed(args.seed)
    obs_scale, samples = base.estimate_obs_scale(args.seed)
    train_args = argparse.Namespace(
        seed=args.seed, n_envs=args.n_envs, episode_ticks=1000, bound_logits=True, reward="profit"
    )
    venv = DummyVecEnv(
        base.make_env_fns(train_args, reward_scale=base.SAC_REWARD_SCALE, obs_scale=obs_scale)
    )
    venv.action_space.seed(args.seed)
    buffer_size = args.demo_steps + args.steps
    model = SAC(
        "MlpPolicy", venv, buffer_size=min(buffer_size, 1_000_000), gradient_steps=-1,
        ent_coef=args.ent_coef, learning_starts=0, verbose=1,
    )
    fill_buffer_with_demos(model, obs_scale, args.demo_steps, args.seed)
    clone_actor(model, args.clone_steps)
    warm_critics(model, args.critic_steps, args.ent_coef)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_folded_copy(model, args.out.with_name(args.out.stem + "_warmstart.zip"), obs_scale, samples)
    model.learn(total_timesteps=args.steps, reset_num_timesteps=True)
    fold_and_check(model, obs_scale, samples)
    model.save(args.out)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
