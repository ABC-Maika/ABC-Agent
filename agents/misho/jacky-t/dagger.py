"""DAgger data for Jerky: play Jerky's own policy, label every state it visits with a teacher's action.

The teachers' own recordings never contain debt or a collapsed opening (a good agent is never there), so
a clone has no answer in those states and falls into a bad lever mix (loyalty spend ~0.3, research ~0.04,
wage ~0.3) that it never leaves. Here the learner's checkpoint plays (with probability ``--beta`` per
round the teacher's action is executed instead), and the teacher, a stateless MLP that can answer for any
observation, is asked what it would do in each state the learner reaches.

    python agents/misho/jacky-t/dagger.py runs/cand/s8.zip --episodes 12 --out runs/dagger/it1a.npz
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import train_league as league  # noqa: E402
from mimic import to_env_action  # noqa: E402

from retail_simulator.core.encoding import flatten_action_space_layout  # noqa: E402
from retail_simulator.core.world import compute_expansion_mask  # noqa: E402
from retail_simulator.envs.arena_match import _materialize_predicts  # noqa: E402
from retail_simulator.envs.spaces import build_action_space  # noqa: E402

TICKS = 1000
TEACHERS = ("kmlp8", "kmlp6")  # stateless MLPs
POOL = ["orcho", "orcho-v5", "kmlp6", "kmlp8", "krischo", "krischo-v8", "yav-sac", "yavkata"]
HOMES = (1, 5, 9, 13, 17, 21)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("learner", type=Path, help="folded checkpoint that plays")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--shift0", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.0, help="chance per round the teacher acts")
    parser.add_argument("--scale", type=Path, default=Path("runs/jacky_t/league_ckpt/obs_scale.npy"))
    parser.add_argument("--rule-labels", action="store_true", help="expansion label from the region score")
    parser.add_argument("--retaliate", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    obs_scale = np.load(args.scale)
    exp_slot = next(sl for sl in flatten_action_space_layout().levers if sl.spec.name == "expansion")
    rng = random.Random(args.seed)
    obs_rows, act_rows, profits = [], [], []
    for episode in range(args.episodes):
        shift = (args.shift0 + episode) % 6
        homes = tuple(HOMES[(i + shift) % 6] for i in range(6))
        env = league.LeagueEnv(
            args.seed + episode, obs_scale, (1.0, 0.0, 0.0, 0.0), 0.0, 0.0, 0,
            pool=POOL, fixed_homes=homes, research_cap=0.4,
        )
        teacher_name = TEACHERS[episode % len(TEACHERS)]
        predicts = _materialize_predicts(
            {"learner": str(args.learner), "teacher": league.OPPONENTS[teacher_name]},
            action_space=build_action_space(),
            config=env.config,
        )
        obs, info = env.reset()
        for tick in range(TICKS):
            raw = env._obs[env.agents_[league.LEARNER]]
            label = to_env_action(predicts["teacher"](raw, episode_start=tick == 0))
            if args.rule_labels:
                state = env.world._state
                scores = league.region_scores(env.config, state.retailers, league.LEARNER, args.retaliate)
                legal = compute_expansion_mask(state, league.LEARNER, env.config)
                scores = np.where(legal[: len(scores)], scores, -np.inf)
                scores = np.where(league.may_open(state.retailers[league.LEARNER], scores, tick)[: len(scores)], scores, -np.inf)
                pick = int(np.argmax(scores)) if np.isfinite(scores.max()) and scores.max() > 0 else 0
                block = np.full(exp_slot.width, -1.0, dtype=np.float32)
                block[pick] = 1.0
                label[exp_slot.offset : exp_slot.offset + exp_slot.width] = block
            if rng.random() < args.beta:
                action = label
            else:
                action = np.asarray(predicts["learner"](raw, episode_start=tick == 0), dtype=np.float32).ravel()
            obs_rows.append(obs)
            act_rows.append(label)
            obs, *_ = env.step(action)
        profits.append(float(env.cum[league.LEARNER]))
        print(f"dagger {episode + 1}/{args.episodes} home {homes[0]} teacher {teacher_name} "
              f"learner profit {profits[-1]:,.0f} opponents {info['opponents']}", flush=True)
    n = len(obs_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Every row is kept when cloning: ranks/profits are set so ``load_demos``'s filter passes them.
    np.savez_compressed(args.out, obs=np.stack(obs_rows), act=np.stack(act_rows),
                        ep_rank=np.ones(n, dtype=int), ep_profit=np.ones(n))
    print(f"saved {n} samples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
