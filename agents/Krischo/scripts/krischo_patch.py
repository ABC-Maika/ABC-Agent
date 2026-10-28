"""Patch a trained RecurrentPPO zip in place of retraining it.

* ``--wage`` shifts the mean of the ``wage_spend`` output (a constant bias on the output layer).
* ``--expansion`` re-fits ONLY the 32 expansion logits (rows of the output layer) on states the agent itself
  visits, labelled by the rule expert (DAgger-style), so it stops fixating on one already-owned region.
  Everything else in the network (LSTM, other levers) stays exactly as it is.

Own script; nothing in ``src/`` is edited. Run from the repo root.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "scripts"))

import krischo_league as kl  # noqa: E402
import krischo_optimize as ko  # noqa: E402

from retail_simulator.core.encoding import flatten_action_space_layout  # noqa: E402
from retail_simulator.envs.arena import arena_balanced_config  # noqa: E402
from retail_simulator.envs.parallel_env import RetailParallelEnv  # noqa: E402

LAYOUT = {s.spec.name: s for s in flatten_action_space_layout().levers}
EXP = LAYOUT["expansion"]


def collect(job: tuple[str, dict, list[dict], int]) -> tuple[np.ndarray, np.ndarray]:
    """The agent plays seat 0; its output-layer input (the latent) is recorded with the expert's expansion label."""
    zip_path, teacher_params, opps, seed = job
    from sb3_contrib import RecurrentPPO

    model = RecurrentPPO.load(zip_path, device="cpu")
    seen: list[np.ndarray] = []
    model.policy.action_net.register_forward_hook(lambda m, i, o: seen.append(i[0].detach().cpu().numpy().reshape(-1).copy()))
    rng = np.random.default_rng(seed)
    homes = tuple(int(h) for h in rng.permutation(kl.HOMES))
    world_seed = int(rng.integers(0, 2**31 - 1))
    env = RetailParallelEnv(arena_balanced_config(), n_learning_agents=6, home_regions=homes, seed=world_seed, mask_expansion_seats=tuple(range(6)))
    agents = env.possible_agents
    teacher, others = ko.Expert(teacher_params), [ko.Expert(o) for o in opps]
    obs, _ = env.reset(seed=world_seed)
    state, first, labels = None, np.array([True]), []
    for _ in range(1000):
        o0 = np.asarray(obs[agents[0]], dtype=np.float32)
        labels.append(int(np.argmax(teacher(o0)[EXP.offset : EXP.offset + EXP.width])))
        action, state = model.predict(o0, state=state, episode_start=first, deterministic=True)
        first = np.array([False])
        acts = {agents[0]: np.asarray(action, dtype=np.float32).ravel()}
        for i, a in enumerate(agents[1:]):
            acts[a] = np.asarray(others[i](obs[a]), dtype=np.float32).ravel()
        obs, _, _, _, _ = env.step(acts)
    return np.stack(seen[-1000:]), np.asarray(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default="agents/Krischo/rppo-lstm/checkpoint.zip")
    parser.add_argument("--out", default="agents/Krischo/rppo-lstm/checkpoint_patched.zip")
    parser.add_argument("--teacher", default="agents/Krischo/rppo-lstm/expert_params_v4.json")
    parser.add_argument("--pool", default="runs/league2/pool.json")
    parser.add_argument("--wage", type=float, default=None, help="target mean wage_spend (e.g. 0.5)")
    parser.add_argument("--expansion", action="store_true", help="re-fit the expansion head")
    parser.add_argument("--matches", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--nonzero-weight", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()

    import torch
    from sb3_contrib import RecurrentPPO

    torch.set_num_threads(4)
    model = RecurrentPPO.load(args.zip, device="cpu")
    head = model.policy.action_net

    if args.wage is not None:
        slot = LAYOUT["wage_spend"].offset
        with torch.no_grad():
            before = float(head.bias[slot])
            head.bias[slot] += args.wage - 0.04  # the agent's mean wage output is ~0.04
        print(f"wage_spend output bias {before:.3f} -> {float(head.bias[slot]):.3f} (target mean {args.wage})", flush=True)

    if args.expansion:
        t0 = time.time()
        teacher = {**ko.BASELINE, **json.loads(Path(args.teacher).read_text())}
        pool = [{**ko.BASELINE, **p} for p in json.loads(Path(args.pool).read_text())]
        jobs = [(args.zip, teacher, kl.draw(pool, 4000 + i, 5), 4000 + i) for i in range(args.matches)]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(collect, jobs, chunksize=1))
        x = torch.tensor(np.concatenate([r[0] for r in res]), dtype=torch.float32)
        y = torch.tensor(np.concatenate([r[1] for r in res]), dtype=torch.long)
        print(f"collected {len(y)} states in {int(time.time() - t0)}s; expert expansion labels: {int((y > 0).sum())} non-zero, {len(torch.unique(y)) - 1} distinct regions", flush=True)
        rows = slice(EXP.offset, EXP.offset + EXP.width)
        w = torch.nn.Parameter(head.weight.data[rows].clone())
        b = torch.nn.Parameter(head.bias.data[rows].clone())
        cls_w = torch.ones(EXP.width)
        cls_w[1:] = args.nonzero_weight
        opt = torch.optim.Adam([w, b], lr=1e-2)
        with torch.no_grad():
            acc0 = ((x @ w.T + b).argmax(1) == y).float().mean().item()
            nz0 = ((x @ w.T + b).argmax(1)[y > 0] == y[y > 0]).float().mean().item()
        for ep in range(args.epochs):
            loss = torch.nn.functional.cross_entropy(x @ w.T + b, y, weight=cls_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = (x @ w.T + b).argmax(1)
            acc1, nz1 = (pred == y).float().mean().item(), (pred[y > 0] == y[y > 0]).float().mean().item()
            head.weight.data[rows] = w.data
            head.bias.data[rows] = b.data
        print(f"expansion head: accuracy {acc0:.3f} -> {acc1:.3f}; on the states where the expert expands {nz0:.3f} -> {nz1:.3f} (loss {loss.item():.3f})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
