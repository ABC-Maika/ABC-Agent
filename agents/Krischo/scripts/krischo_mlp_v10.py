"""v10 variant of krischo_mlp_v8.py: the DAgger rollouts can also play against scripted NPC personas (hunters, contenders,
written "npc:<name>" in --zips), so the clone sees the states where an aggressive rival squeezes it.

v8 variant of krischo_mlp.py: the teacher is the v8 expert (ring/contest-scored expansion, wage from round 1, frugal opening) and the DAgger rollouts also play against real agent zips.

Original docstring: Clone a memoryless rule expert (expert_params_v6/v7: it only reads cash, stores, happiness and the per-region store flags)
into a plain PPO MlpPolicy, then DAgger it: the MLP plays, the expert labels the states the MLP visits.

An MLP is stateless and trains in minutes on i.i.d. states, unlike the LSTM clone that needs back-propagation through 1000 steps.
The observation scale is folded into the first layer so the saved zip takes RAW observations (the ladder applies no normalisation).
Own script; nothing in ``src/`` is edited. Run from the repo root:

    .venv\\Scripts\\python.exe agents\\Krischo\\scripts\\krischo_mlp.py --params agents\\Krischo\\rppo-lstm\\expert_params_v6.json --out runs\\alt\\mlp_v6.zip
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
import krischo_v8 as kv8  # noqa: E402  (imported BEFORE the patch below: ExpertV8 subclasses the real Expert)

_BASE_EXPERT = ko.Expert


def _expert_factory(params: dict):
    """Experts whose params carry a "_v8" marker are the v8 teacher; every other expert (the pool) stays the plain v6 class."""
    v8 = params.get("_v8")
    if v8:
        return kv8.ExpertV8({k: v for k, v in params.items() if k != "_v8"}, v8["mode"], v8["frugal_ticks"], v8["frugal_mkt"])
    return _BASE_EXPERT(params)


ko.Expert = _expert_factory  # runs at import, so spawned worker processes get the patch too

from retail_simulator.envs.arena import arena_balanced_config  # noqa: E402
from retail_simulator.envs.parallel_env import RetailParallelEnv  # noqa: E402


def make_opp(o, kz, cfg):
    """An opponent for a rollout: "npc:<name>" = a scripted persona (hunter/contender/archetype), a str = a frozen zip, a dict = a v6 expert."""
    if isinstance(o, str) and o.startswith("npc:"):
        from aggressive_npcs import register

        register()
        from retail_simulator.envs import arena_match

        table = cfg.demand.regions
        predict = arena_match.geo_archetypes(len(table), table)[o[4:]]

        class _Npc:
            def __init__(self) -> None:
                self.first = True

            def __call__(self, obs):
                out = predict(obs, episode_start=self.first)
                self.first = False
                return out

        return _Npc()
    return kz.ZipAny(o) if isinstance(o, str) else ko.Expert(o)


def rollout(job: tuple[str, dict, list, int]) -> tuple[np.ndarray, np.ndarray, float]:
    """The MLP (zip) plays seat 0 against expert / zip opponents; the expert labels every state it visits."""
    import krischo_zoo as kz

    zip_path, params, opps, seed = job
    rng = np.random.default_rng(seed)
    homes = tuple(int(h) for h in rng.permutation(kl.HOMES))
    world_seed = int(rng.integers(0, 2**31 - 1))
    env = RetailParallelEnv(arena_balanced_config(), n_learning_agents=kl.N_SEATS, home_regions=homes, seed=world_seed, mask_expansion_seats=tuple(range(kl.N_SEATS)))
    agents = env.possible_agents
    learner = kz.ZipAny(zip_path)
    teacher = ko.Expert(params)
    others = [make_opp(o, kz, arena_balanced_config()) for o in opps]
    obs, _ = env.reset(seed=world_seed)
    seen, labels, score = [], [], 0.0
    for _ in range(1000):
        o0 = np.asarray(obs[agents[0]], dtype=np.float32)
        seen.append(o0)
        labels.append(teacher(o0))
        actions = {agents[0]: learner(o0)}
        for i, a in enumerate(agents[1:]):
            actions[a] = np.asarray(others[i](obs[a]), dtype=np.float32).ravel()
        obs, rewards, _, _, _ = env.step(actions)
        score += float(rewards[agents[0]])
    return np.stack(seen), np.stack(labels), score


def main() -> None:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from retail_simulator.core.encoding import flatten_action_space_layout
    from retail_simulator.core.schema import LeverKind

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", default="agents/Krischo/rppo-lstm/expert_params_v6.json")
    ap.add_argument("--pool", default="runs/league2/pool.json")
    ap.add_argument("--out", default="runs/alt/mlp_v6.zip")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--rollouts", type=int, default=12)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60, help="passes over the data for the first fit")
    ap.add_argument("--epochs-iter", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--exp-weight", type=float, default=40.0)
    ap.add_argument("--net", default="256,256", help="hidden sizes of the actor MLP")
    ap.add_argument("--orcho", default=None, help="zip of an opponent to include in most rollouts")
    ap.add_argument("--orcho-frac", type=float, default=0.3)
    ap.add_argument("--zips", nargs="*", default=[], help="real agent zips used as opponents in DAgger rollouts")
    ap.add_argument("--zips-frac", type=float, default=0.7)
    ap.add_argument("--zips-k", type=int, default=3, help="how many zip opponents per rollout")
    ap.add_argument("--init", default=None, help="folded checkpoint zip whose actor weights start the clone (e.g. mlp-v9)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    params = {**ko.BASELINE, **json.loads(Path(args.params).read_text())}
    pool = [{**ko.BASELINE, **p} for p in json.loads(Path(args.pool).read_text())]
    t0 = time.time()
    obs, act = kl.collect_league(params, pool, args.episodes, 5000, args.workers)
    print(f"expert data {obs.shape} ({int(time.time() - t0)}s)", flush=True)
    obs, act = obs.reshape(-1, obs.shape[-1]), act.reshape(-1, act.shape[-1])
    scale = np.maximum(np.abs(obs).max(axis=0), 1.0) * 1.5

    arch = [int(x) for x in args.net.split(",")]
    model = PPO("MlpPolicy", DummyVecEnv([lambda: ko.NpcFieldEnv(0)]), policy_kwargs=dict(net_arch=dict(pi=arch, vf=[64, 64]), activation_fn=torch.nn.ReLU), seed=0, device="cpu", verbose=0)
    pol = model.policy
    layout = flatten_action_space_layout()
    disc = [(s.offset, s.width) for s in layout.levers if s.spec.kind is LeverKind.DISCRETE]
    disc_cols = {c for off, w in disc for c in range(off, off + w)}
    cont = [c for c in range(act.shape[1]) if c not in disc_cols]
    if args.init:
        src = PPO.load(args.init, device="cpu").policy
        pol.mlp_extractor.policy_net.load_state_dict(src.mlp_extractor.policy_net.state_dict())
        pol.action_net.load_state_dict(src.action_net.state_dict())
        with torch.no_grad():
            pol.mlp_extractor.policy_net[0].weight.mul_(torch.tensor(scale, dtype=torch.float32))
    params_train = [*pol.mlp_extractor.policy_net.parameters(), *pol.action_net.parameters()]
    opt = torch.optim.Adam(params_train, lr=args.lr)

    def actor(x: torch.Tensor) -> torch.Tensor:
        return pol.action_net(pol.mlp_extractor.forward_actor(pol.extract_features(x)))

    def fit(o: np.ndarray, a: np.ndarray, epochs: int, lr: float) -> tuple[float, float]:
        x = torch.tensor(o / scale, dtype=torch.float32)
        y = torch.tensor(a, dtype=torch.float32)
        for g in opt.param_groups:
            g["lr"] = lr
        n = x.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, args.batch):
                idx = perm[i : i + args.batch]
                out = actor(x[idx])
                loss = torch.nn.functional.mse_loss(out[:, cont], y[idx][:, cont])
                for off, w in disc:
                    weight = torch.ones(w)
                    if w > 3:
                        weight[1:] = args.exp_weight
                    loss = loss + torch.nn.functional.cross_entropy(out[:, off : off + w], y[idx][:, off : off + w].argmax(1), weight=weight)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_train, 1.0)
                opt.step()
        with torch.no_grad():
            out = actor(x[: min(n, 60000)])
            acc = np.mean([(out[:, off : off + w].argmax(1) == y[: out.shape[0], off : off + w].argmax(1)).float().mean().item() for off, w in disc])
            mae = (out[:, cont] - y[: out.shape[0]][:, cont]).abs().mean().item()
        return mae, acc

    def save_folded(path: str) -> None:
        first = pol.mlp_extractor.policy_net[0]
        saved, log_std = first.weight.data.clone(), pol.log_std.data.clone()
        with torch.no_grad():
            first.weight.mul_(torch.tensor(1.0 / scale, dtype=torch.float32))
            pol.log_std.data.fill_(-2.0)
        model.save(path)
        with torch.no_grad():
            first.weight.data.copy_(saved)
            pol.log_std.data.copy_(log_std)

    mae, acc = fit(obs, act, args.epochs, args.lr)
    save_folded(args.out)
    stem = args.out[:-4]
    save_folded(f"{stem}_it0.zip")
    print(f"iter 0: continuous MAE {mae:.4f}, discrete argmax accuracy {acc:.4f} -> {args.out} ({int(time.time() - t0)}s)", flush=True)
    for it in range(1, args.iters + 1):
        jobs = []
        for e in range(args.rollouts):
            seed = 7000 + 100 * it + e
            r = random.Random(seed)
            if args.zips and r.random() < args.zips_frac:
                k = min(args.zips_k, len(args.zips), 5)
                opps = [*r.sample(args.zips, k), *kl.draw(pool, seed, 5 - k)]
            elif args.orcho and r.random() < args.orcho_frac:
                opps = [args.orcho, *kl.draw(pool, seed, 4)]
            else:
                opps = kl.draw(pool, seed, 5)
            jobs.append((args.out, params, opps, seed))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(rollout, jobs, chunksize=1))
        obs = np.concatenate([obs, *[r[0] for r in res]])
        act = np.concatenate([act, *[r[1] for r in res]])
        print(f"iter {it}: MLP mean score in its own rollouts {np.mean([r[2] for r in res]):,.0f} (min {np.min([r[2] for r in res]):,.0f}); data {obs.shape[0]} states ({int(time.time() - t0)}s)", flush=True)
        mae, acc = fit(obs, act, args.epochs_iter, args.lr * 0.5)
        save_folded(args.out)
        save_folded(f"{stem}_it{it}.zip")
        print(f"iter {it}: continuous MAE {mae:.4f}, discrete argmax accuracy {acc:.4f} -> {stem}_it{it}.zip ({int(time.time() - t0)}s)", flush=True)
    print(f"done in {int(time.time() - t0)}s", flush=True)


if __name__ == "__main__":
    main()


