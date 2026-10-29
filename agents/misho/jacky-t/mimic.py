"""Behaviour-clone Orcho and KrischoV8 into Jacky-T's TQC actor.

``collect`` plays matches with a teacher (a trained checkpoint on main) in Jacky-T's seat against the
other trained agents plus a scripted archetype, and records (observation, teacher action) each tick.
``bc`` regresses Jacky-T's actor mean onto those actions, then writes an unfolded snapshot that
``train_league.py --from-actor`` can fine-tune (the obs scale is saved beside it).

    python agents/misho/jacky-t/mimic.py collect orcho --episodes 10 --out runs/mimic/orcho.npz
    python agents/misho/jacky-t/mimic.py collect krischo-v8 --episodes 10 --out runs/mimic/krischo-v8.npz
    python agents/misho/jacky-t/mimic.py bc runs/mimic/orcho.npz runs/mimic/krischo-v8.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import train_league as league  # noqa: E402
from anchored_tqc import load_demos  # noqa: E402
from sb3_contrib import TQC  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from retail_simulator.core.encoding import flatten_action_space_layout  # noqa: E402
from retail_simulator.core.schema import LeverKind  # noqa: E402
from retail_simulator.envs.arena_match import _materialize_predicts  # noqa: E402
from retail_simulator.envs.spaces import build_action_space  # noqa: E402

TICKS = 1000
SCRIPTED_FILL = "cherry-picker"  # the fifth opponent when the teacher's own checkpoint is removed


def to_env_action(native: np.ndarray) -> np.ndarray:
    """A teacher's native action in the bounded-logit space Jacky-T trains in.

    Continuous levers keep their true values. Each argmax-decoded block becomes one-hot (+1 at the
    argmax, -1 elsewhere), which decodes to the same choice.
    """
    action = np.asarray(native, dtype=np.float32).ravel().copy()
    for slot in flatten_action_space_layout().levers:
        if slot.spec.kind is LeverKind.DISCRETE:
            block = action[slot.offset : slot.offset + slot.width]
            one_hot = np.full(slot.width, -1.0, dtype=np.float32)
            one_hot[int(np.argmax(block))] = 1.0
            action[slot.offset : slot.offset + slot.width] = one_hot
    return action


def collect(args: argparse.Namespace) -> None:
    obs_scale = np.load(args.scale)
    teacher_path = league.OPPONENTS[args.teacher]
    # The teacher takes Jacky-T's seat; the other four trained agents plus one archetype fill the rest.
    pool = [name for name in league.OPPONENTS if name != args.teacher] + [SCRIPTED_FILL]
    obs_rows, act_rows, ep_profit, ep_rank = [], [], [], []
    for episode in range(args.episodes):
        # Cover every home: episode k starts from rotation (shift0 + k) of the six METRO homes.
        base = (1, 5, 9, 13, 17, 21)
        shift = (args.shift0 + episode) % 6
        homes = tuple(base[(i + shift) % 6] for i in range(6))
        env = league.LeagueEnv(
            args.seed + episode, obs_scale, (0.3, 0.3, 0.3, 0.1), 0.0, 0.0, 0, pool=pool,
            fixed_homes=homes if args.rotate else None,
        )
        teacher = _materialize_predicts(
            {args.teacher: teacher_path}, action_space=build_action_space(), config=env.config
        )[args.teacher]
        obs, info = env.reset()
        for tick in range(TICKS):
            raw = env._obs[env.agents_[league.LEARNER]]
            action = to_env_action(teacher(raw, episode_start=tick == 0))
            obs_rows.append(obs)
            act_rows.append(action)
            obs, *_ = env.step(action)
        # The teacher's outcome in this episode, stored on every row, so flawed episodes can be
        # dropped when cloning (finished outside the top 2, or lost money).
        profit = float(env.cum[league.LEARNER])
        rank = int((env.cum > env.cum[league.LEARNER]).sum()) + 1
        ep_profit += [profit] * TICKS
        ep_rank += [rank] * TICKS
        print(
            f"{args.teacher} episode {episode + 1}/{args.episodes} profit={profit:,.0f} rank={rank}"
            f" homes={info['homes']}",
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        obs=np.stack(obs_rows),
        act=np.stack(act_rows),
        ep_profit=np.array(ep_profit),
        ep_rank=np.array(ep_rank),
    )
    print(f"saved {len(obs_rows)} samples -> {args.out}")


def behaviour_clone(args: argparse.Namespace) -> None:
    # Drop the teachers' flawed episodes and lift the wage target (see ``load_demos``).
    obs, act = load_demos(
        args.data, args.max_rank, args.wage_floor, args.expand_boost, args.early_wage, args.frugal_opening
    )
    # With a research cap the saved action space carries the lower bound, and targets above it are
    # clipped to it (the scaled target is clipped to [-1, 1]).
    if args.new_net:
        # A fresh TQC with a bigger network (the default 256x256 is small for 318 inputs and 44
        # outputs). Its critics start untrained, so the RL stage needs a long critic warm-up.
        env = league.LeagueEnv(0, np.load(args.scale), (1.0, 0.0, 0.0, 0.0), 0.0, 0.0, 0,
                               research_cap=args.research_cap)
        model = TQC(
            "MlpPolicy",
            DummyVecEnv([lambda: env]),
            policy_kwargs={"net_arch": [int(x) for x in args.new_net.split(",")]},
            learning_rate=1e-4,
            buffer_size=args.buffer_size,
            learning_starts=4_000,
            batch_size=256,
            gamma=0.997,
            gradient_steps=2,
            ent_coef=f"auto_{args.ent_init}",
            seed=args.seed,
            device="cpu",
        )
    else:
        model = TQC.load(
            args.snapshot,
            device="cpu",
            custom_objects={"action_space": league.capped_action_space(args.research_cap)},
        )
    policy = model.policy
    scaled_act = np.clip(policy.scale_action(act), -1.0, 1.0)
    target = torch.as_tensor(scaled_act, dtype=torch.float32)  # to [-1, 1]
    inputs = torch.as_tensor(obs, dtype=torch.float32)
    actor = policy.actor
    params = [*actor.latent_pi.parameters(), *actor.mu.parameters()]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    rng = np.random.default_rng(0)
    n = len(inputs)
    exp_slot = next(sl for sl in flatten_action_space_layout().levers if sl.spec.name == "expansion")
    exp_lo, exp_hi = exp_slot.offset, exp_slot.offset + exp_slot.width
    for epoch in range(args.epochs):
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, args.batch_size):
            batch = order[start : start + args.batch_size]
            mean, _, _ = actor.get_action_dist_params(inputs[batch])
            pred = torch.tanh(mean)
            if args.expansion_ce:
                # The expansion choice is one of 32 (hold + regions): a regression to averaged one-hot
                # targets collapsed it to the same regions in every home (2, 6 whatever the home), so
                # that block is fit as a classification instead.
                rest = torch.ones(pred.shape[1], dtype=torch.bool)
                rest[exp_lo:exp_hi] = False
                loss = torch.nn.functional.mse_loss(pred[:, rest], target[batch][:, rest]) + (
                    args.expansion_ce
                    * torch.nn.functional.cross_entropy(
                        pred[:, exp_lo:exp_hi] * 6.0, target[batch][:, exp_lo:exp_hi].argmax(1)
                    )
                )
            else:
                loss = torch.nn.functional.mse_loss(pred, target[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        print(f"epoch {epoch + 1}/{args.epochs}: mse {total / n:.4f}", flush=True)
    # Start the RL stage from the clone with a set, constant exploration noise (std = e**log_std;
    # the old default -3 gives ~0.05, too little to discover anything) and entropy coefficient.
    with torch.no_grad():
        actor.log_std.weight.zero_()
        actor.log_std.bias.fill_(args.log_std)
        model.log_ent_coef.fill_(float(np.log(args.ent_init)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    np.save(args.out.parent / "obs_scale.npy", np.load(args.scale))
    print(f"saved {args.out} ({n} samples, {model.num_timesteps} steps)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    scale = Path("runs/jacky_t/league_ckpt/obs_scale.npy")
    c = sub.add_parser("collect")
    c.add_argument("teacher", choices=("orcho", "krischo-v8", "krischo", "kmlp6", "kmlp8"))
    c.add_argument("--episodes", type=int, default=10)
    c.add_argument("--seed", type=int, default=100)
    c.add_argument("--rotate", action="store_true", help="cycle the six home rotations")
    c.add_argument("--shift0", type=int, default=0)
    c.add_argument("--scale", type=Path, default=scale)
    c.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("bc")
    b.add_argument("data", type=Path, nargs="+")
    b.add_argument("--snapshot", type=Path, default=Path("runs/jacky_t/league_ckpt/jacky_t_league_100000_steps.zip"))
    b.add_argument("--scale", type=Path, default=scale)
    b.add_argument("--research-cap", type=float, default=None)
    b.add_argument("--max-rank", type=int, default=0, help="keep teacher episodes ranked this or better")
    b.add_argument("--wage-floor", type=float, default=0.0)
    b.add_argument("--expand-boost", type=int, default=1, help="repeat expansion rows this many times")
    b.add_argument("--early-wage", type=float, default=0.0)
    b.add_argument("--expansion-ce", type=float, default=0.0, help="weight of the expansion classification loss")
    b.add_argument("--frugal-opening", action="store_true")
    b.add_argument("--epochs", type=int, default=12)
    b.add_argument("--lr", type=float, default=1e-3)
    b.add_argument("--batch-size", type=int, default=256)
    b.add_argument("--out", type=Path, default=Path("runs/jacky_t/bc/jacky_t_bc.zip"))
    b.add_argument("--new-net", help="e.g. 512,512: clone into a fresh TQC with this net_arch")
    b.add_argument("--buffer-size", type=int, default=300_000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--log-std", type=float, default=-3.0, help="initial log std of the actor")
    b.add_argument("--ent-init", type=float, default=0.01, help="initial entropy coefficient")
    args = parser.parse_args()
    (collect if args.cmd == "collect" else behaviour_clone)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
