"""Teacher data from the scripted lever lab (lever_lab.py) for cloning into Jerky.

    play    the lab seat plays Jerky's seat itself: the states of a well-played game.
    dagger  Jerky's checkpoint plays and the lab labels every state it reaches with its own action. The lab
            reads only the observation (store counts, staff happiness), so it can answer in any state, debt
            included, which the trained teachers never could.

    python agents/misho/jacky-t/lab_collect.py play opt --episodes 36 --out runs/lab/play_opt.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, "scripts")

import numpy as np  # noqa: E402
import lever_lab  # noqa: E402
import train_league as league  # noqa: E402
from mimic import to_env_action  # noqa: E402

from retail_simulator.envs.arena_match import _materialize_predicts  # noqa: E402
from retail_simulator.envs.spaces import build_action_space  # noqa: E402

TICKS = 1000
HOMES = (1, 5, 9, 13, 17, 21)
SHIFTS = (0, 1, 2, 3, 4, 5, 0, 1, 2, 3)  # the four homes Jerky lost get twice the games
# No Orcho in the field: a lab seat playing Orcho's opening against Orcho splits every region with its twin.
POOL = ["orcho-v7", "orcho-v62", "orcho-v5", "kmlp10", "kmlp9", "kmlp10rl", "kmlp8", "yav-sac", "yav-sac2",
        "cherry-picker", "fast-follower", "sprawler"]  # one per family is drawn each episode (see LeagueEnv.family)
VARIANTS = {
    "opt": dict(price=2.0, service=0.2, research=0.1, loyalty=0.4, cap=10, nclass=2),
    "opt12": dict(price=2.0, service=0.2, research=0.1, loyalty=0.4, cap=12, nclass=2),  # all-in: METRO + AFFLUENT to 12 stores
    "opt12s": dict(price=2.0, service=0.2, research=0.1, loyalty=0.4, cap=12, nclass=2, collapse_price=1.0),  # + late-game price cut once restructured
    "beta": dict(price=2.0, service=0.2, research=0.1, loyalty=0.4, cap=12, nclass=2, war_cash=25000, war_price=1.8, collapse_price=1.0),  # Jerky-beta: all-in + price war with idle cash + late price cut
    "jackrec": dict(price=2.0, service=0.2, research=0.1, loyalty=0.4, cap=12, nclass=2, order="low",
                    crowd_price=0.1, crowd_floor=1.7, war_cash=25000, war_price=1.8, war_profit=300.0,
                    stuck={"wage": 0.2, "service": 0.0, "research": 0.0}),  # lowest region first + gated price war + stuck-state levers
    "low10": dict(price=2.0, service=0.2, research=0.1, loyalty=0.4, cap=10, nclass=2, order="low"),  # lowest region first, 10 stores: no negative game in 60 scripted games
    "18": dict(price=1.8, service=0.2, research=0.1, loyalty=0.4, cap=18, nclass=3),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("play", "dagger"))
    parser.add_argument("variant", choices=tuple(VARIANTS))
    parser.add_argument("--episodes", type=int, default=36)
    parser.add_argument("--learner", type=Path, default=Path("runs/cand/s9big.zip"))
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--scale", type=Path, default=Path("runs/jacky_t/league_ckpt/obs_scale.npy"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    obs_scale = np.load(args.scale)
    obs_rows, act_rows, profits = [], [], []
    for episode in range(args.episodes):
        shift = SHIFTS[(args.offset + episode) % len(SHIFTS)]
        homes = tuple(HOMES[(i + shift) % 6] for i in range(6))
        env = league.LeagueEnv(args.seed + episode, obs_scale, (1.0, 0.0, 0.0, 0.0), 0.0, 0.0, 0,
                               pool=POOL, fixed_homes=homes, research_cap=0.4)
        lab = lever_lab._lab_predict({**lever_lab.V5, **VARIANTS[args.variant]}, len(env.config.demand.regions))
        learner = None
        if args.mode == "dagger":
            learner = _materialize_predicts({"learner": str(args.learner)}, action_space=build_action_space(),
                                            config=env.config)["learner"]
        obs, info = env.reset()
        for tick in range(TICKS):
            raw = env._obs[env.agents_[league.LEARNER]]
            label = to_env_action(lab(raw, episode_start=tick == 0))
            action = label if learner is None else np.asarray(learner(raw, episode_start=tick == 0), dtype=np.float32).ravel()
            obs_rows.append(obs); act_rows.append(label)
            obs, *_ = env.step(action)
        profits.append(float(env.cum[league.LEARNER]))
        rank = int((env.cum > env.cum[league.LEARNER]).sum()) + 1
        print(f"{args.mode} {args.variant} {episode + 1}/{args.episodes} home {homes[0]} profit {profits[-1]:,.0f} rank {rank}", flush=True)
        if args.mode == "play":
            obs_rows, act_rows = obs_rows, act_rows  # kept whole; filtered by episode below
    n = len(obs_rows)
    ep_profit = np.repeat(np.array(profits), TICKS)
    ep_rank = np.ones(n, dtype=int) if args.mode == "dagger" else np.where(ep_profit > 0, 1, 9)
    if args.mode == "dagger":
        ep_profit = np.ones(n)  # every state of Jerky's own play is kept
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, obs=np.stack(obs_rows), act=np.stack(act_rows), ep_rank=ep_rank, ep_profit=ep_profit)
    print(f"saved {n} samples -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
