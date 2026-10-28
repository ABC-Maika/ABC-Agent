"""League of smarter opponents for Krischo's agent (six seats, no NPCs, balanced world).

``league``   generations of co-evolution on parametric experts. Each generation:
             1. search OUR expert against five opponents drawn from the pool (fitness = mean + min, so robust),
             2. keep it only if it beats the previous best on held-out seeds,
             3. search a BEST-RESPONSE opponent that maximises its own score against our current best,
             4. add the winners to the pool (opponents get smarter every generation).
``evalzip``  play a trained zip (our LSTM clone) against the strongest opponents of the pool.

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
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "scripts"))  # the organizer's scripts (train_vs_npcs)

import krischo_optimize as ko  # noqa: E402

from retail_simulator.envs.arena import arena_balanced_config  # noqa: E402
from retail_simulator.envs.parallel_env import RetailParallelEnv  # noqa: E402

HOMES = [1, 5, 9, 13, 17, 21]
N_SEATS = 6
OUT = Path("runs/league")


class ZipPolicy:
    """Stateful, deterministic wrapper of a RecurrentPPO zip (like the ladder's predict closure)."""

    def __init__(self, path: str) -> None:
        import torch
        from sb3_contrib import RecurrentPPO

        torch.set_num_threads(1)
        self.model = RecurrentPPO.load(path, device="cpu")
        self.reset()

    def reset(self) -> None:
        self.state = None
        self.first = np.array([True])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action, self.state = self.model.predict(obs, state=self.state, episode_start=self.first, deterministic=True)
        self.first = np.array([False])
        return np.asarray(action, dtype=np.float32).ravel()


def _policy(spec: tuple[str, Any]):
    kind, payload = spec
    return ko.Expert(payload) if kind == "expert" else ZipPolicy(payload)


def play_match(job: tuple[list[tuple[str, Any]], int]) -> list[float]:
    """One 1000-round six-seat match; returns every seat's total profit."""
    specs, seed = job
    rng = np.random.default_rng(seed)
    homes = tuple(int(h) for h in rng.permutation(HOMES))
    world_seed = int(rng.integers(0, 2**31 - 1))
    env = RetailParallelEnv(
        arena_balanced_config(),
        n_learning_agents=N_SEATS,
        home_regions=homes,
        seed=world_seed,
        mask_expansion_seats=tuple(range(N_SEATS)),
    )
    agents = env.possible_agents
    policies = [_policy(s) for s in specs]
    for p in policies:
        p.reset()
    obs, _ = env.reset(seed=world_seed)
    totals = [0.0] * N_SEATS
    for _ in range(1000):
        actions = {a: np.asarray(policies[i](obs[a]), dtype=np.float32).ravel() for i, a in enumerate(agents)}
        obs, rewards, _, _, _ = env.step(actions)
        for i, a in enumerate(agents):
            totals[i] += float(rewards[a])
    return totals


def record_match(job: tuple[dict, list[dict], int]) -> tuple[np.ndarray, np.ndarray]:
    """Seat 0 plays ``params``; returns its (obs, action) per round so a policy can be cloned from real league states."""
    params, opps, seed = job
    rng = np.random.default_rng(seed)
    homes = tuple(int(h) for h in rng.permutation(HOMES))
    world_seed = int(rng.integers(0, 2**31 - 1))
    env = RetailParallelEnv(
        arena_balanced_config(),
        n_learning_agents=N_SEATS,
        home_regions=homes,
        seed=world_seed,
        mask_expansion_seats=tuple(range(N_SEATS)),
    )
    agents = env.possible_agents
    policies = [ko.Expert(params), *[ko.Expert(o) for o in opps]]
    obs, _ = env.reset(seed=world_seed)
    seen, acted = [], []
    for _ in range(1000):
        actions = {a: np.asarray(policies[i](obs[a]), dtype=np.float32).ravel() for i, a in enumerate(agents)}
        seen.append(np.asarray(obs[agents[0]], dtype=np.float32))
        acted.append(actions[agents[0]])
        obs, _, _, _, _ = env.step(actions)
    return np.stack(seen), np.stack(acted)


def collect_league(params: dict, pool: list[dict], n_eps: int, seed0: int, workers: int = 11) -> tuple[np.ndarray, np.ndarray]:
    jobs = [(params, draw(pool, seed0 + e, 5), seed0 + e) for e in range(n_eps)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(record_match, jobs, chunksize=1))
    return np.stack([r[0] for r in res]), np.stack([r[1] for r in res])


def dagger_rollout(job: tuple[str, dict, list[dict], int]) -> tuple[np.ndarray, np.ndarray, float]:
    """The LEARNER (zip) plays seat 0 against expert opponents; the expert labels every state it visits."""
    zip_path, params, opps, seed = job
    rng = np.random.default_rng(seed)
    homes = tuple(int(h) for h in rng.permutation(HOMES))
    world_seed = int(rng.integers(0, 2**31 - 1))
    env = RetailParallelEnv(
        arena_balanced_config(),
        n_learning_agents=N_SEATS,
        home_regions=homes,
        seed=world_seed,
        mask_expansion_seats=tuple(range(N_SEATS)),
    )
    agents = env.possible_agents
    learner = ZipPolicy(zip_path)
    teacher = ko.Expert(params)
    others = [ko.Expert(o) for o in opps]
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


def make_hard(rng: random.Random) -> dict:
    """An opponent of the family that beat us: undercuts our price, max marketing, 12 stores, never harvests."""
    p = dict(ko.BASELINE)
    p.update(
        price=rng.uniform(1.6, 1.95),
        marketing=rng.uniform(0.9, 1.0),
        assortment=rng.uniform(0.9, 1.0),
        promotion=0.91,
        max_stores=rng.choice((10, 12, 14)),
        max_class=rng.choice((3, 4)),
        cash_buffer=rng.uniform(0.0, 6000.0),
        wage_spend=0.05,
        warehouse=0.18,
        harvest_tick=1000,
        k_price=rng.choice((0.0, 0.03)),
        service=1.0,
    )
    return p


def cmd_dagger(args: argparse.Namespace) -> None:
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from retail_simulator.core.encoding import flatten_action_space_layout
    from retail_simulator.core.schema import LeverKind

    torch.set_num_threads(args.threads)
    params = {**ko.BASELINE, **json.loads(Path(args.params).read_text())}
    pool = [{**ko.BASELINE, **p} for p in json.loads(Path(args.pool).read_text())]
    t0 = time.time()
    hard_rng = random.Random(123)
    hard = [make_hard(hard_rng) for _ in range(12)]
    obs, act = collect_league(params, pool + hard if args.hard_frac > 0 else pool, args.episodes, 5000, args.workers)
    print(f"initial expert data {obs.shape} ({int(time.time() - t0)}s)", flush=True)
    scale = np.maximum(np.abs(obs).max(axis=(0, 1)), 1.0) * 1.5
    model = RecurrentPPO("MlpLstmPolicy", DummyVecEnv([lambda: ko.NpcFieldEnv(0)]), seed=0, device="cpu", verbose=0)
    pol = model.policy
    hidden = pol.lstm_actor.hidden_size
    layout = flatten_action_space_layout()
    disc = [(s.offset, s.width) for s in layout.levers if s.spec.kind is LeverKind.DISCRETE]
    disc_cols = {c for off, w in disc for c in range(off, off + w)}
    cont = [c for c in range(act.shape[2]) if c not in disc_cols]
    train_params = [*pol.lstm_actor.parameters(), *pol.mlp_extractor.policy_net.parameters(), *pol.action_net.parameters()]

    def fit(obs_a: np.ndarray, act_a: np.ndarray, epochs: int, lr: float) -> float:
        n_ep, horizon, dim = obs_a.shape
        x = torch.tensor((obs_a / scale).reshape(-1, dim), dtype=torch.float32)
        y = torch.tensor(act_a.reshape(-1, act_a.shape[2]), dtype=torch.float32)
        starts = torch.zeros(n_ep * horizon)
        opt = torch.optim.Adam(train_params, lr=lr)
        loss_v = 0.0
        for _ in range(epochs):
            feats = pol.extract_features(x)
            latent, _ = pol._process_sequence(feats, (torch.zeros(1, n_ep, hidden), torch.zeros(1, n_ep, hidden)), starts, pol.lstm_actor)
            out = pol.action_net(pol.mlp_extractor.forward_actor(latent))
            loss = torch.nn.functional.mse_loss(out[:, cont], y[:, cont])
            for off, w in disc:
                weight = torch.ones(w)
                if w > 3:
                    weight[1:] = args.exp_weight
                loss = loss + torch.nn.functional.cross_entropy(out[:, off : off + w], y[:, off : off + w].argmax(1), weight=weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            opt.step()
            loss_v = loss.item()
        return loss_v

    def save_folded(path: str) -> None:
        inv = torch.tensor(1.0 / scale, dtype=torch.float32)
        lstms = [m for m in (pol.lstm_actor, pol.lstm_critic) if m is not None]
        saved = [m.weight_ih_l0.data.clone() for m in lstms]
        log_std = pol.log_std.data.clone()
        with torch.no_grad():
            for m in lstms:
                m.weight_ih_l0.mul_(inv)
            pol.log_std.data.fill_(-2.0)
        model.save(path)
        with torch.no_grad():  # restore the normalised-input weights so training can continue
            for m, w in zip(lstms, saved):
                m.weight_ih_l0.data.copy_(w)
            pol.log_std.data.copy_(log_std)

    out_zip = args.out
    loss = fit(obs, act, args.epochs, args.lr)
    save_folded(out_zip)
    print(f"iter 0: loss {loss:.4f} -> {out_zip} ({int(time.time() - t0)}s)", flush=True)
    for it in range(1, args.iters + 1):
        jobs = []
        for e in range(args.rollouts):
            seed = 7000 + 100 * it + e
            r = random.Random(seed)
            opps = draw(hard, seed, 3) + draw(pool, seed, 2) if r.random() < args.hard_frac else draw(pool, seed, 5)
            jobs.append((out_zip, params, opps, seed))
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            res = list(ex.map(dagger_rollout, jobs, chunksize=1))
        new_obs = np.stack([r[0] for r in res])
        new_act = np.stack([r[1] for r in res])
        obs, act = np.concatenate([obs, new_obs]), np.concatenate([act, new_act])
        scale = np.maximum(scale, np.abs(new_obs).max(axis=(0, 1)) * 1.1)  # never shrink; data may exceed it
        print(f"iter {it}: learner mean score in its own rollouts {np.mean([r[2] for r in res]):,.0f} (min {np.min([r[2] for r in res]):,.0f}); data {obs.shape[0]} episodes ({int(time.time() - t0)}s)", flush=True)
        loss = fit(obs, act, args.epochs_iter, args.lr * 0.5)
        save_folded(out_zip)
        print(f"iter {it}: refit loss {loss:.4f} -> {out_zip} ({int(time.time() - t0)}s)", flush=True)
    print(f"done in {int(time.time() - t0)}s", flush=True)


def draw(pool: list[dict], seed: int, k: int) -> list[dict]:
    """The same opponents for every candidate on a given seed (fair comparison)."""
    return [pool[i] for i in random.Random(seed).sample(range(len(pool)), k)]


def evaluate(ex: ProcessPoolExecutor, cands: list[dict], seeds: list[int], make_specs, idx: int) -> list[dict]:
    jobs = [(make_specs(c, s), s) for c in cands for s in seeds]
    res = list(ex.map(play_match, jobs, chunksize=1))
    out = []
    for i, c in enumerate(cands):
        rs = res[i * len(seeds) : (i + 1) * len(seeds)]
        mine = [r[idx] for r in rs]
        rank = [1 + sum(o > r[idx] for j, o in enumerate(r) if j != idx) for r in rs]
        out.append(
            {
                "params": c,
                "mean": float(np.mean(mine)),
                "min": float(np.min(mine)),
                "fit": float(0.5 * np.mean(mine) + 0.5 * np.min(mine)),
                "rank": float(np.mean(rank)),
                "wins": int(sum(r == 1 for r in rank)),
            }
        )
    return out


def local_search(ex, cands, seeds, make_specs, idx, rounds, kids, rng, log, tag) -> list[dict]:
    res = sorted(evaluate(ex, cands, seeds, make_specs, idx), key=lambda r: -r["fit"])
    top = res[:3]
    print(f"  {tag} stage 1: best fit {top[0]['fit']:,.0f} (mean {top[0]['mean']:,.0f}, min {top[0]['min']:,.0f})", flush=True)
    for rnd in range(rounds):
        new = [ko.mutate(p["params"], rng) for p in top for _ in range(kids)]
        top = sorted(top + evaluate(ex, new, seeds, make_specs, idx), key=lambda r: -r["fit"])[:3]
        print(f"  {tag} round {rnd + 1}: best fit {top[0]['fit']:,.0f} (mean {top[0]['mean']:,.0f}, min {top[0]['min']:,.0f})", flush=True)
    log[tag] = top
    return top


def initial_pool(rng: random.Random) -> list[dict]:
    d = Path("agents/Krischo/rppo-lstm")
    v1 = {**ko.BASELINE, **json.loads((d / "expert_params_v1.json").read_text())}
    v2 = {**ko.BASELINE, **json.loads((d / "expert_params_v2.json").read_text())}
    pool = [dict(ko.BASELINE), v1, v2]
    for base in (v1, v2):
        for _ in range(4):
            kid = ko.mutate(base, rng)
            kid["price"] = rng.uniform(1.0, 2.0)  # different pricing postures: undercutters and premium
            pool.append(kid)
    for p in pool:
        p["service"] = 1.0
    return pool


def cmd_league(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    if args.pool_in:  # continue from an earlier league: its pool and its best expert
        pool = [{**ko.BASELINE, **p} for p in json.loads(Path(args.pool_in).read_text())]
        best = {**ko.BASELINE, **json.loads(Path(args.best_in).read_text())}
    else:
        pool = initial_pool(rng)
        best = dict(pool[2])  # start from the dynamic v2 expert
    log: dict[str, Any] = {"generations": []}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for g in range(args.generations):
            print(f"=== generation {g + 1}/{args.generations} (pool {len(pool)}) {int(time.time() - t0)}s", flush=True)
            seeds = [300 + 10 * g + i for i in range(args.train_eps)]
            held = [600 + 10 * g + i for i in range(args.val_eps)]
            ours = lambda c, s, pool=pool: [("expert", c), *[("expert", o) for o in draw(pool, s, 5)]]  # noqa: E731
            cands = [dict(best)] + [ko.mutate(best, rng) for _ in range(args.n_cands)] + [ko.sample(rng) for _ in range(args.n_random)]
            for _ in range(args.n_dyn):  # variants that read rivals and contested regions (research on)
                kid = dict(best)
                for k in ko.DYNAMIC_KEYS:
                    kid[k] = rng.uniform(*ko.SPACE[k])
                kid["research"] = rng.uniform(0.1, 0.5)
                cands.append(ko.mutate(kid, rng))
            gen_log: dict[str, Any] = {}
            top = local_search(ex, cands, seeds, ours, 0, args.rounds, args.kids, rng, gen_log, "ours")
            check = evaluate(ex, [best, top[0]["params"]], held, ours, 0)
            improved = check[1]["fit"] > check[0]["fit"]
            print(f"  held-out: previous best fit {check[0]['fit']:,.0f} (mean {check[0]['mean']:,.0f}) vs new {check[1]['fit']:,.0f} (mean {check[1]['mean']:,.0f}) -> {'keep new' if improved else 'keep previous'}", flush=True)
            gen_log["held_out"] = check
            if improved:
                best = dict(top[0]["params"])
            # best-response opponent: maximises ITS score against our current best plus four pool members
            br_specs = lambda c, s, pool=pool, b=dict(best): [("expert", b), ("expert", c), *[("expert", o) for o in draw(pool, s, 4)]]  # noqa: E731
            br_cands = [ko.sample(rng) for _ in range(args.n_random)] + [ko.mutate(p, rng) for p in pool[-4:] for _ in range(4)]
            for c in br_cands:
                c["service"] = 1.0
            br_top = local_search(ex, br_cands, seeds, br_specs, 1, args.rounds, args.kids, rng, gen_log, "best-response")
            vs_us = evaluate(ex, [br_top[0]["params"]], held, br_specs, 1)[0]
            ours_vs_br = evaluate(ex, [best], held, br_specs, 0)[0]
            print(f"  best-response opponent on held-out seeds: its mean {vs_us['mean']:,.0f}; ours in the same matches {ours_vs_br['mean']:,.0f}", flush=True)
            gen_log["br_held"] = {"br": vs_us, "ours": ours_vs_br}
            pool.append(dict(best))  # our current best also becomes an opponent (self-play)
            pool.extend(dict(p["params"]) for p in br_top[:2])
            log["generations"].append(gen_log)
            (out / "league.json").write_text(json.dumps({"log": log, "pool": pool, "best": best}, indent=1, default=float))
            Path(args.best_out).write_text(json.dumps(best, indent=1))  # best so far, safe to use even if interrupted
            (out / "pool.json").write_text(json.dumps(pool, indent=1))
    Path(args.best_out).write_text(json.dumps(best, indent=1))
    (out / "pool.json").write_text(json.dumps(pool, indent=1))
    print(f"done in {int(time.time() - t0)}s; best expert -> {args.best_out}", flush=True)


def cmd_tournament(args: argparse.Namespace) -> None:
    """Final-style ranking: 12 entrants, groups of six re-drawn every round, rank by aggregate profit."""
    d = Path("agents/Krischo/rppo-lstm")
    pool1 = [{**ko.BASELINE, **p} for p in json.loads(Path(args.pool).read_text())][-8:]
    expert = {**ko.BASELINE, **json.loads((d / "expert_params_v4.json").read_text())}
    entrants: dict[str, tuple[str, Any]] = {
        "Krischo v8 (zip)": ("zip", str(d / "checkpoint_v8.zip")),
        "Krischo v6 (zip)": ("zip", str(d / "checkpoint_v6.zip")),
        "Krischo v4 (zip)": ("zip", str(d / "checkpoint_v4.zip")),
        "Expert v4 (rules)": ("expert", expert),
    }
    for i, p in enumerate(pool1, start=1):
        entrants[f"Opp{i} p{p['price']:.2f} s{p['max_stores']} m{p['marketing']:.2f}"] = ("expert", p)
    names = list(entrants)
    rng = random.Random(args.seed)
    jobs, groups = [], []
    for rnd in range(args.rounds):
        order = names[:]
        rng.shuffle(order)
        for g in range(len(order) // 6):
            grp = order[6 * g : 6 * g + 6]
            groups.append(grp)
            jobs.append(([entrants[n] for n in grp], 2000 + rnd * 10 + g))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(play_match, jobs, chunksize=1))
    total = {n: 0.0 for n in names}
    scores = {n: [] for n in names}
    ranks = {n: [] for n in names}
    for grp, r in zip(groups, res):
        for i, n in enumerate(grp):
            total[n] += r[i]
            scores[n].append(r[i])
            ranks[n].append(1 + sum(o > r[i] for j, o in enumerate(r) if j != i))
    print(f"Tournament: {len(names)} entrants, {len(res)} matches of six, each entrant plays {len(scores[names[0]])} matches")
    print(f"{'#':>2} {'Entrant':<38} {'Total profit':>14} {'Mean':>10} {'Min':>10} {'Avg rank':>9} {'Wins':>5}")
    for pos, n in enumerate(sorted(names, key=lambda k: -total[k]), start=1):
        print(f"{pos:>2} {n:<38} {total[n]:>14,.0f} {np.mean(scores[n]):>10,.0f} {np.min(scores[n]):>10,.0f} {np.mean(ranks[n]):>9.2f} {sum(r == 1 for r in ranks[n]):>5}")


def cmd_evalzip(args: argparse.Namespace) -> None:
    pool = json.loads(Path(args.pool).read_text())
    smart = pool[-args.smart :]  # the newest entries are the best-response opponents and our own bests
    seeds = [900 + i for i in range(args.n)]
    jobs = []
    for s in seeds:
        rng = random.Random(s)
        opp = rng.sample(smart, 5) if len(smart) >= 5 else smart + rng.sample(pool, 5 - len(smart))
        jobs.append(([("zip", args.zip), *[("expert", o) for o in opp]], s))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(play_match, jobs, chunksize=1))
    ours = [r[0] for r in res]
    rank = [1 + sum(o > r[0] for o in r[1:]) for r in res]
    best_opp = [max(r[1:]) for r in res]
    print(f"{args.zip} vs {args.smart} newest league opponents, {len(res)} matches")
    print(f"our mean {np.mean(ours):,.0f}  min {np.min(ours):,.0f}  max {np.max(ours):,.0f}  mean rank {np.mean(rank):.2f}  wins {sum(r == 1 for r in rank)}/{len(rank)}")
    print(f"best opponent per match, mean {np.mean(best_opp):,.0f}")
    print("opponent means:", [f"{np.mean([r[i] for r in res]):,.0f}" for i in range(1, 6)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("league")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=11)
    s.add_argument("--generations", type=int, default=3)
    s.add_argument("--n-cands", type=int, default=16)
    s.add_argument("--n-random", type=int, default=8)
    s.add_argument("--train-eps", type=int, default=4)
    s.add_argument("--val-eps", type=int, default=8)
    s.add_argument("--rounds", type=int, default=2)
    s.add_argument("--kids", type=int, default=6)
    s.add_argument("--best-out", default="agents/Krischo/rppo-lstm/expert_params_v3.json")
    s.add_argument("--out-dir", default="runs/league")
    s.add_argument("--pool-in", default=None, help="pool.json of an earlier league to continue from")
    s.add_argument("--best-in", default=None, help="expert params json to start from (with --pool-in)")
    s.add_argument("--n-dyn", type=int, default=0, help="variants of the best with random rival-reading parameters")
    d = sub.add_parser("dagger")
    d.add_argument("--params", default="agents/Krischo/rppo-lstm/expert_params_v4.json")
    d.add_argument("--pool", default="runs/league2/pool.json")
    d.add_argument("--out", default="agents/Krischo/rppo-lstm/checkpoint_v6.zip")
    d.add_argument("--episodes", type=int, default=30)
    d.add_argument("--rollouts", type=int, default=16)
    d.add_argument("--iters", type=int, default=3)
    d.add_argument("--epochs", type=int, default=120)
    d.add_argument("--epochs-iter", type=int, default=50)
    d.add_argument("--lr", type=float, default=2e-3)
    d.add_argument("--exp-weight", type=float, default=40.0)
    d.add_argument("--hard-frac", type=float, default=0.0, help="share of rollouts played against the family that beats us")
    d.add_argument("--threads", type=int, default=10)
    d.add_argument("--workers", type=int, default=11)
    t = sub.add_parser("tournament")
    t.add_argument("--pool", default="runs/league/pool.json", help="its newest 8 entries are the opponents")
    t.add_argument("--rounds", type=int, default=12)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--workers", type=int, default=11)
    e = sub.add_parser("evalzip")
    e.add_argument("--zip", required=True)
    e.add_argument("--pool", default="runs/league/pool.json")
    e.add_argument("--smart", type=int, default=8)
    e.add_argument("--n", type=int, default=12)
    e.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    {"league": cmd_league, "evalzip": cmd_evalzip, "dagger": cmd_dagger, "tournament": cmd_tournament}[args.cmd](args)


if __name__ == "__main__":
    main()
