"""Train an A2C policy to reproduce the scripted ``orcho-recipe`` (imitation, with DAgger rounds).

Round 0: the recipe plays several matches against the NPC field and every (observation, action) pair
is recorded. Rounds 1+: the learner plays and the recipe labels the situations the learner reached,
so mistakes get corrected where they actually happen. The network is fitted with squared error on the
levers and cross-entropy on the expansion choice (the argmax decides which region opens). The
observation scaling is folded into the first layer, so the saved checkpoint reads raw observations
like any other entry. The result is an ordinary Stable-Baselines3 A2C ``.zip`` (policy_kind a2c-mlp).

    python scripts/clone_recipe.py --out runs/mkt/orcho-a2c-clone.zip
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from aggressive_npcs import _aggressive_roster, register
from retail_simulator.envs import geo_archetypes as geo
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from train_vs_npcs import NpcFieldEnv

EXPERT = "orcho-recipe"


EXPANSION = slice(4, 36)  # the 32 expansion logits in the 44-wide action vector
# price, marketing, assortment, promotion, research, service, loyalty, wage, warehouse
CONTINUOUS = [0, 1, 2, 3, 36, 37, 41, 42, 43]


def collect(
    episodes: int, ticks: int, first_seed: int, act: Callable[[np.ndarray], np.ndarray] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Record observations and the recipe's action for each. ``act`` (the learner) drives the match
    when given (DAgger); otherwise the recipe drives it. Homes and world seeds change every episode."""
    env = NpcFieldEnv(0, episode_ticks=ticks)
    xs, ys = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=first_seed + ep)
        expert = _aggressive_roster(len(env._regions), env._regions)[EXPERT]
        for t in range(ticks):
            label = expert(obs, episode_start=t == 0)
            xs.append(obs)
            ys.append(label)
            obs, *_ = env.step(label if act is None else act(obs))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def saturate(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Aim past a lever's limit when the recipe sits ON it: the env clips actions, so the policy
    lands exactly on the limit instead of 5% short of it (price 1.95 vs 2.0 costs real profit)."""
    out = y.copy()
    for d in CONTINUOUS:
        out[y[:, d] <= low[d] + 1e-6, d] = low[d] - 1.0
        out[y[:, d] >= high[d] - 1e-6, d] = high[d] + 1.0
    return out


def fit(policy: Any, x: torch.Tensor, y: torch.Tensor, epochs: int) -> float:
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    others = [i for i in range(y.shape[1]) if not (EXPANSION.start <= i < EXPANSION.stop)]
    levers = CONTINUOUS
    target_region = y[:, EXPANSION].argmax(dim=1)
    n, last = len(x), 0.0
    for _ in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, 256):
            idx = perm[i : i + 256]
            mean = policy.get_distribution(x[idx]).distribution.mean
            loss = torch.nn.functional.mse_loss(mean[:, others], y[idx][:, others])
            loss = loss + 10 * torch.nn.functional.mse_loss(mean[:, levers], y[idx][:, levers])
            loss = loss + torch.nn.functional.cross_entropy(mean[:, EXPANSION], target_region[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        last = total / n
    return last


def main() -> int:
    register()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--dagger-episodes", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("runs/mkt/orcho-a2c-clone.zip"))
    args = parser.parse_args()

    x, y = collect(args.episodes, args.ticks, 0)
    # The recipe reads only the seat's own store counts. Feeding the network anything else lets it
    # lean on features (rival blocks) that look different when the rivals are trained agents.
    names = geo._name_to_index(25)
    relevant = [names["stores"]] + [names[f"stores_region_{r}"] for r in range(1, 25)]
    keep = np.zeros(x.shape[1], dtype=np.float32)
    keep[relevant] = 1.0
    scale = np.maximum(np.abs(x).max(axis=0), 1.0).astype(np.float32)
    venv = DummyVecEnv([lambda: Monitor(NpcFieldEnv(0))])
    model = A2C(
        "MlpPolicy", venv, ent_coef=0.0, verbose=0,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [64, 64]}},
    )
    policy = model.policy

    def act(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            dist = policy.get_distribution(torch.as_tensor(obs / scale * keep)[None].float())
            return dist.distribution.mean[0].numpy()

    low, high = venv.action_space.low, venv.action_space.high
    loss = fit(policy, torch.as_tensor(x / scale * keep), torch.as_tensor(saturate(y, low, high)), args.epochs)
    print(f"round 0: {len(x)} samples, loss {loss:.4f}", flush=True)
    for rnd in range(1, args.dagger_rounds + 1):
        nx, ny = collect(args.dagger_episodes, args.ticks, 1000 * rnd, act)
        x, y = np.concatenate([x, nx]), np.concatenate([y, ny])
        agree = float(np.mean(ny[:, EXPANSION].argmax(1) == np.array([act(o)[EXPANSION].argmax() for o in nx[::10]]).repeat(10)[: len(ny)]))
        loss = fit(policy, torch.as_tensor(x / scale * keep), torch.as_tensor(saturate(y, low, high)), args.epochs)
        print(f"round {rnd}: {len(x)} samples, loss {loss:.4f}, expansion agreement before refit {agree:.0%}", flush=True)
    n = len(x)

    # Fold the observation scale into the first layers, so raw observations work at scoring time.
    with torch.no_grad():
        for seq in (policy.mlp_extractor.policy_net, policy.mlp_extractor.value_net):
            seq[0].weight /= torch.as_tensor(scale)[None, :]
            seq[0].weight *= torch.as_tensor(keep)[None, :]  # ignore every input the recipe ignores
    raw = torch.as_tensor(x[:2000])
    with torch.no_grad():
        pred = policy.get_distribution(raw).distribution.mean.numpy()
    same = float(np.mean(pred[:, EXPANSION].argmax(1) == y[:2000][:, EXPANSION].argmax(1)))
    print(f"after folding: expansion choice matches the recipe on {same:.0%} of 2000 samples")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    meta = {"algo": "a2c", "reward": "imitation", "steps": int(n), "seed": 0, "ent_coef": 0.0}
    args.out.with_suffix(".json").write_text(json.dumps(meta) + "\n")
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
