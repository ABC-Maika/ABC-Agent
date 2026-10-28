"""Krischo's optimisation pipeline: search a rule-based expert, then clone it into a RecurrentPPO zip.

``search``  random + local search over a parametric expert (levers + expansion rule) against the NPC
            field (the practice field of ``train_vs_npcs.py``), validated on held-out seeds.
``clone``   behaviour-clone the best expert into a RecurrentPPO (MlpLstmPolicy) and fold the
            observation scale into the LSTM input weights, so the saved zip takes RAW observations.

Own script: nothing in ``src/`` is edited. Run from the repo root.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "scripts"))  # the organizer's scripts (train_vs_npcs)

from train_vs_npcs import NpcFieldEnv  # noqa: E402

from retail_simulator.core.encoding import (  # noqa: E402
    StructuredAction,
    encode_action,
    flatten_action_space_layout,
)
from retail_simulator.core.schema import LeverKind, observation_schema  # noqa: E402
from retail_simulator.envs.arena import arena_balanced_config  # noqa: E402

N_REGIONS = 25
CAPEX = 5000.0
OUT_DIR = Path("runs/krischo_opt")

# cherry-picker's posture plus its METRO-only expansion: the strongest practice NPC, our reference.
BASELINE: dict[str, float] = {
    "price": 1.5,
    "marketing": 0.4,
    "assortment": 0.45,
    "promotion": 0.2,
    "service": 1.0,
    "research": 0.0,
    "wage_spend": 0.0,
    "warehouse": 0.0,
    "max_stores": 6,
    "max_class": 1,  # 1 = METRO only, 2 = + AFFLUENT, 3 = + LOYAL, 4 = all
    "cash_buffer": 0.0,
    "prefer_uncontested": 0,
    "harvest_tick": 1000,
    "harvest_cut": 1.0,
    # dynamic reactions (0 = constant lever, the earlier behaviour)
    "k_price": 0.0,  # price rises with our mean share over held regions
    "k_mkt": 0.0,  # marketing rises when share is low
    "k_wh": 0.0,  # warehouse spend rises with the last stockout rate
    "k_wage": 0.0,  # wage spend rises when employee happiness is low
    "cap_cut": 0,  # 1 = halve marketing where share sits at the 0.6 cap
    # reads rivals (needs research > 0): price toward the rivals' mean, marketing with competition, contested-region logic
    "w_rival": 0.0,  # weight of the rival-price target in the price (0 = ignore rivals)
    "rival_margin": 0.0,  # target = rivals' mean price + margin (negative = undercut)
    "k_comp": 0.0,  # marketing rises with the mean number of rivals in our regions
    "contest_pen": 0.0,  # expansion score = region demand / (1 + pen * rivals present)
    "score_min": 0.0,  # skip regions whose expansion score is below this
    "early_boost": 0.0,  # extra marketing in the opening rounds
    "early_ticks": 100,
}

SPACE: dict[str, tuple[float, float]] = {
    "price": (1.0, 2.0),
    "marketing": (0.0, 1.0),
    "assortment": (0.0, 1.0),
    "promotion": (0.0, 1.0),
    "research": (0.0, 0.5),
    "wage_spend": (0.0, 0.3),
    "warehouse": (0.0, 0.3),
    "cash_buffer": (0.0, 10000.0),
    "harvest_cut": (0.0, 1.0),
    "k_price": (0.0, 2.0),
    "k_mkt": (0.0, 2.0),
    "k_wh": (0.0, 3.0),
    "k_wage": (0.0, 1.0),
    "w_rival": (0.0, 1.0),
    "rival_margin": (-0.4, 0.4),
    "k_comp": (0.0, 1.5),
    "contest_pen": (0.0, 3.0),
    "score_min": (0.0, 700.0),
    "early_boost": (0.0, 0.5),
}
DYNAMIC_KEYS = ("k_price", "k_mkt", "k_wh", "k_wage", "w_rival", "rival_margin", "k_comp", "contest_pen", "score_min", "early_boost")
INT_CHOICES: dict[str, tuple[int, ...]] = {
    "early_ticks": (50, 100, 200, 400),
    "max_stores": (6, 8, 10, 12, 14, 16, 18),
    "max_class": (1, 2, 3, 4),
    "prefer_uncontested": (0, 1),
    "harvest_tick": (600, 700, 800, 900, 1000),
    "cap_cut": (0, 1),
}


class Expert:
    """Parametric rule-based policy: fixed lever posture plus a rule for where/when to expand."""

    def __init__(self, params: dict[str, float]) -> None:
        self.p = {**BASELINE, **params}
        self.idx = {f.name: i for i, f in enumerate(observation_schema(N_REGIONS))}
        regions = arena_balanced_config().demand.regions
        demand = sorted({regions[r].base_regional_demand for r in range(1, N_REGIONS)}, reverse=True)
        self.class_rank = {r: demand.index(regions[r].base_regional_demand) for r in range(1, N_REGIONS)}
        self.base_demand = {r: regions[r].base_regional_demand for r in range(1, N_REGIONS)}
        self.tick = 0

    def reset(self) -> None:
        self.tick = 0

    def _expansion(self, obs: np.ndarray, tick: int) -> int:
        p, idx = self.p, self.idx
        if tick >= p["harvest_tick"] or obs[idx["stores"]] >= p["max_stores"]:
            return 0
        if obs[idx["cash"]] < CAPEX + p["cash_buffer"]:
            return 0
        if p["contest_pen"] > 0 or p["score_min"] > 0:
            scored = []
            for r in range(1, N_REGIONS):
                if self.class_rank[r] >= p["max_class"] or obs[idx[f"stores_region_{r}"]] != 0.0:
                    continue
                score = self.base_demand[r] / (1.0 + p["contest_pen"] * float(obs[idx[f"others_present_region_{r}"]]))
                if score >= p["score_min"]:
                    scored.append((-score, r))
            return min(scored)[1] if scored else 0
        best, best_key = 0, None
        for r in range(1, N_REGIONS):
            if self.class_rank[r] >= p["max_class"] or obs[idx[f"stores_region_{r}"]] != 0.0:
                continue
            contested = obs[idx[f"others_present_region_{r}"]] > 0.0
            key = (self.class_rank[r], int(contested) if p["prefer_uncontested"] else 0, r)
            if best_key is None or key < best_key:
                best, best_key = r, key
        return best

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        tick = self.tick
        self.tick += 1
        p, idx = self.p, self.idx
        cut = p["harvest_cut"] if tick >= p["harvest_tick"] else 1.0
        held = [r for r in range(1, N_REGIONS) if obs[idx[f"stores_region_{r}"]] > 0]
        share = float(np.mean([obs[idx[f"market_share_region_{r}"]] for r in held])) if held else 0.0
        price = min(2.0, max(1.0, p["price"] + p["k_price"] * (share - 0.4)))
        rivals = [float(obs[idx[f"competitor_mean_price_region_{r}"]]) for r in held if obs[idx[f"others_present_region_{r}"]] > 0]
        if p["w_rival"] > 0 and rivals:
            target = min(2.0, max(1.0, float(np.mean(rivals)) + p["rival_margin"]))
            price = (1.0 - p["w_rival"]) * price + p["w_rival"] * target
        intensity = float(np.mean([obs[idx[f"others_present_region_{r}"]] for r in held])) if held else 0.0
        early = p["early_boost"] if tick < p["early_ticks"] else 0.0
        marketing = min(1.0, max(0.0, p["marketing"] + p["k_mkt"] * (0.4 - share) + p["k_comp"] * min(intensity / 5.0, 1.0) + early))
        if p["cap_cut"] and share >= 0.55:
            marketing *= 0.5
        warehouse = min(1.0, p["warehouse"] + p["k_wh"] * float(obs[idx["last_stockout_rate"]]))
        wage = min(1.0, p["wage_spend"] + p["k_wage"] * max(0.0, 0.5 - float(obs[idx["employee_happiness"]])))
        return encode_action(
            StructuredAction(
                levers={
                    "price_index": price,
                    "marketing": marketing * cut,
                    "assortment": p["assortment"] * cut,
                    "promotion": p["promotion"],
                    "expansion": float(self._expansion(obs, tick)),
                    "research": p["research"],
                    "service_level": p["service"],
                    "automation": 0.0,
                    "loyalty_spend": 0.0,
                    "wage_spend": wage,
                    "warehouse_invest": warehouse,
                }
            )
        )


def run_episode(params: dict[str, float], seed: int) -> dict[str, float]:
    """One 1000-round match of the expert in seat 0 against the NPC field."""
    env = NpcFieldEnv(seed)
    expert = Expert(params)
    obs, _ = env.reset()
    expert.reset()
    total, min_cash, truncated = 0.0, 1e18, False
    idx = expert.idx
    while not truncated:
        obs, reward, _, truncated, _ = env.step(expert(obs))
        total += reward
        min_cash = min(min_cash, float(obs[idx["cash"]]))
    held = [r for r in range(1, N_REGIONS) if obs[idx[f"stores_region_{r}"]] > 0]
    share = float(np.mean([obs[idx[f"market_share_region_{r}"]] for r in held])) if held else 0.0
    return {"score": total, "stores": float(obs[idx["stores"]]), "share": share, "min_cash": min_cash}


def _task(args: tuple[dict[str, float], int]) -> dict[str, float]:
    return run_episode(*args)


def evaluate(pool: ProcessPoolExecutor, params_list: list[dict[str, float]], seeds: list[int]) -> list[dict[str, Any]]:
    jobs = [(p, s) for p in params_list for s in seeds]
    results = list(pool.map(_task, jobs, chunksize=1))
    out = []
    for i, p in enumerate(params_list):
        rs = results[i * len(seeds) : (i + 1) * len(seeds)]
        scores = [r["score"] for r in rs]
        out.append(
            {
                "params": p,
                "mean": float(np.mean(scores)),
                "min": float(np.min(scores)),
                "stores": float(np.mean([r["stores"] for r in rs])),
                "share": float(np.mean([r["share"] for r in rs])),
                "neg_cash_eps": int(sum(r["min_cash"] < 0 for r in rs)),
            }
        )
    return out


def sample(rng: random.Random) -> dict[str, float]:
    p: dict[str, float] = {k: rng.uniform(*v) for k, v in SPACE.items()}
    p.update({k: rng.choice(v) for k, v in INT_CHOICES.items()})
    p["service"] = 1.0
    return p


def mutate(p: dict[str, float], rng: random.Random) -> dict[str, float]:
    q = dict(p)
    for k in rng.sample(list(SPACE) + list(INT_CHOICES), 2):
        if k in SPACE:
            lo, hi = SPACE[k]
            q[k] = min(hi, max(lo, q[k] + rng.gauss(0, 0.15 * (hi - lo))))
        else:
            q[k] = rng.choice(INT_CHOICES[k])
    return q


def fmt(r: dict[str, Any]) -> str:
    p = r["params"]
    keys = ["price", "marketing", "assortment", "promotion", "research", "wage_spend", "warehouse", "max_stores", "max_class", "cash_buffer", "prefer_uncontested", "harvest_tick", "harvest_cut"]
    ps = " ".join(f"{k}={p[k]:.2f}" if isinstance(p[k], float) else f"{k}={p[k]}" for k in keys)
    return f"mean={r['mean']:>11,.0f} min={r['min']:>11,.0f} stores={r['stores']:.1f} share={r['share']:.3f} negcash={r['neg_cash_eps']} | {ps}"


def cmd_search(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train_seeds = list(range(100, 100 + args.train_eps))
    held_out = list(range(200, 200 + args.val_eps))
    log: dict[str, Any] = {"train_seeds": train_seeds, "held_out": held_out}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cands = [dict(BASELINE)] + [sample(rng) for _ in range(args.n_random)]
        if args.seed_params:
            prev = {**BASELINE, **json.loads(Path(args.seed_params).read_text())}
            cands.append(dict(prev))  # the earlier best, so the search can only improve on it
            for _ in range(args.n_seeded):
                kid = dict(prev)
                for k in ("k_price", "k_mkt", "k_wh", "k_wage"):
                    kid[k] = rng.uniform(*SPACE[k])
                kid["cap_cut"] = rng.choice((0, 1))
                cands.append(mutate(kid, rng))
        print(f"stage 1: {len(cands)} candidates x {len(train_seeds)} episodes", flush=True)
        res = sorted(evaluate(pool, cands, train_seeds), key=lambda r: -r["mean"])
        log["stage1"] = res
        for r in res[:6]:
            print(fmt(r), flush=True)
        top = res[:3]
        for rnd in range(args.rounds):
            kids = [mutate(parent["params"], rng) for parent in top for _ in range(args.kids)]
            kid_res = evaluate(pool, kids, train_seeds)
            top = sorted(top + kid_res, key=lambda r: -r["mean"])[:3]
            print(f"round {rnd + 1}: best mean {top[0]['mean']:,.0f}", flush=True)
            log[f"round{rnd + 1}"] = kid_res
        finalists = [BASELINE] + [t["params"] for t in top]
        print(f"validation on {len(held_out)} held-out episodes", flush=True)
        val = evaluate(pool, finalists, held_out)
        log["validation"] = val
        names = ["baseline(cherry-picker-like)"] + [f"finalist{i + 1}" for i in range(len(top))]
        for n, r in zip(names, val):
            print(f"{n:<28}", fmt(r), flush=True)
    best = max(val[1:], key=lambda r: r["mean"])
    log["best"] = best
    (OUT_DIR / "search.json").write_text(json.dumps(log, indent=1))
    Path(args.best_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.best_out).write_text(json.dumps(best["params"], indent=1))
    print("best held-out:", fmt(best), flush=True)


# --------------------------------------------------------------------------- clone


def collect(params: dict[str, float], n_eps: int, seed0: int) -> tuple[np.ndarray, np.ndarray]:
    obs_all, act_all = [], []
    for e in range(n_eps):
        env = NpcFieldEnv(seed0 + e)
        expert = Expert(params)
        obs, _ = env.reset()
        expert.reset()
        o, a, truncated = [], [], False
        while not truncated:
            act = expert(obs)
            o.append(obs)
            a.append(act)
            obs, _, _, truncated, _ = env.step(act)
        obs_all.append(np.stack(o))
        act_all.append(np.stack(a))
    return np.stack(obs_all), np.stack(act_all)


def cmd_clone(args: argparse.Namespace) -> None:
    import torch
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    torch.set_num_threads(args.threads)
    params = {**BASELINE, **json.loads(Path(args.params).read_text())}
    print("collecting expert data ...", flush=True)
    if args.field == "league":  # states of six-seat matches against the league opponents
        from krischo_league import collect_league

        pool = json.loads(Path(args.pool).read_text())
        obs, act = collect_league(params, pool, args.episodes, 5000)
    else:
        obs, act = collect(params, args.episodes, 5000)
    n_ep, horizon, obs_dim = obs.shape
    print(f"dataset {obs.shape} {act.shape}", flush=True)

    scale = np.maximum(np.abs(obs).max(axis=(0, 1)), 1.0) * 1.25
    model = RecurrentPPO("MlpLstmPolicy", DummyVecEnv([lambda: NpcFieldEnv(0)]), seed=0, device="cpu", verbose=0)
    pol = model.policy
    hidden = pol.lstm_actor.hidden_size
    layout = flatten_action_space_layout()
    disc = [(s.offset, s.width) for s in layout.levers if s.spec.kind is LeverKind.DISCRETE]
    disc_cols = {c for off, w in disc for c in range(off, off + w)}
    cont = [c for c in range(act.shape[2]) if c not in disc_cols]

    x = torch.tensor((obs / scale).reshape(-1, obs_dim), dtype=torch.float32)
    y = torch.tensor(act.reshape(-1, act.shape[2]), dtype=torch.float32)
    starts = torch.zeros(n_ep * horizon)  # zero initial state == reset state, avoids the slow per-step loop
    zeros = lambda: (torch.zeros(1, n_ep, hidden), torch.zeros(1, n_ep, hidden))  # noqa: E731

    def forward() -> torch.Tensor:
        feats = pol.extract_features(x)
        latent, _ = pol._process_sequence(feats, zeros(), starts, pol.lstm_actor)
        return pol.action_net(pol.mlp_extractor.forward_actor(latent))

    params_to_train = [*pol.lstm_actor.parameters(), *pol.mlp_extractor.policy_net.parameters(), *pol.action_net.parameters()]
    opt = torch.optim.Adam(params_to_train, lr=args.lr)
    for epoch in range(args.epochs):
        out = forward()
        loss = torch.nn.functional.mse_loss(out[:, cont], y[:, cont])
        for off, w in disc:
            target = y[:, off : off + w].argmax(dim=1)
            # expansion is rare (a few opens per match): up-weight every non-no-op class
            weight = torch.ones(w)
            if w > 3:
                weight[1:] = args.exp_weight
            loss = loss + torch.nn.functional.cross_entropy(out[:, off : off + w], target, weight=weight)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_to_train, 1.0)
        opt.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:>4} loss {loss.item():.5f}", flush=True)

    with torch.no_grad():
        out = forward()
        cont_mae = (out[:, cont] - y[:, cont]).abs().mean().item()
        exp_acc = np.mean([(out[:, off : off + w].argmax(1) == y[:, off : off + w].argmax(1)).float().mean().item() for off, w in disc])
        before = out.clone()
    print(f"clone fidelity: continuous MAE {cont_mae:.4f}, discrete argmax accuracy {exp_acc:.4f}", flush=True)

    # fold the observation scale into the LSTM input weights so the zip takes raw observations
    inv = torch.tensor(1.0 / scale, dtype=torch.float32)
    with torch.no_grad():
        for lstm in (pol.lstm_actor, pol.lstm_critic):
            if lstm is not None:
                lstm.weight_ih_l0.mul_(inv)
        x_raw = torch.tensor(obs.reshape(-1, obs_dim), dtype=torch.float32)
        feats = pol.extract_features(x_raw)
        latent, _ = pol._process_sequence(feats, zeros(), starts, pol.lstm_actor)
        after = pol.action_net(pol.mlp_extractor.forward_actor(latent))
        drift = (before - after).abs().max().item()
    print(f"obs-scale fold max drift {drift:.2e}", flush=True)
    pol.log_std.data.fill_(-2.0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"saved {args.out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=10)
    s.add_argument("--n-random", type=int, default=40)
    s.add_argument("--train-eps", type=int, default=4)
    s.add_argument("--val-eps", type=int, default=12)
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--kids", type=int, default=8)
    s.add_argument("--best-out", default="agents/Krischo/rppo-lstm/expert_params.json")
    s.add_argument("--seed-params", default=None, help="json of an earlier best expert to start from")
    s.add_argument("--n-seeded", type=int, default=24, help="dynamic-lever variants of --seed-params")
    c = sub.add_parser("clone")
    c.add_argument("--params", default="agents/Krischo/rppo-lstm/expert_params.json")
    c.add_argument("--out", default="agents/Krischo/rppo-lstm/checkpoint_v1.zip")
    c.add_argument("--episodes", type=int, default=24)
    c.add_argument("--epochs", type=int, default=150)
    c.add_argument("--lr", type=float, default=2e-3)
    c.add_argument("--exp-weight", type=float, default=40.0, help="CE weight of non-no-op expansion classes")
    c.add_argument("--field", choices=("npc", "league"), default="npc", help="where the expert data is collected")
    c.add_argument("--pool", default="runs/league/pool.json")
    c.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    {"search": cmd_search, "clone": cmd_clone}[args.cmd](args)


if __name__ == "__main__":
    main()
