# retail-simulator arena — competitor edition

This is the runnable 25-region arena for training and self-checking a
reinforcement-learning entry. It ships the environment, the training CLI, and
the self-check tooling; the balance study and internal tooling behind the
world's tuning are intentionally not included.

## Setup

```bash
git clone https://github.com/pencho-dobrev/retail-simulator-arena.git
cd retail-simulator-arena

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[rl,train,recurrent]"
```

## Train

`scripts/arena_train_player.py` builds the training world, trains one PPO
player, and writes the full effective recipe to `recipe.json` in the pool
directory. Checkpoints land under `runs/phase4_rl_confirm/<label>/`, one
snapshot per round — `runs/` is gitignored, so a checkpoint exists only on
your own machine until you submit one.

Always train in the balanced world (`--world balanced`) — that is the world
the competition is played in.

Three reference recipes — they differ by objective, not by seed:

```bash
# Explorer -- profit-only reward with an entropy bonus.
python scripts/arena_train_player.py my_explorer --world balanced \
    --ent-coef 0.02 --seeds 2,2,44

# Margin -- profit-only reward (the default), no entropy bonus.
python scripts/arena_train_player.py my_margin --world balanced --seeds 0,0,42

# Share -- the weighted objective instead of profit only.
python scripts/arena_train_player.py my_share --world balanced \
    --reward weighted --seeds 1,1,43
```

## Self-check

Seat your checkpoint against the NPC field with
`scripts/arena_competition.py` and play a full match. This is practice, not how
finals are scored (see [Rules](#rules)).

```bash
python scripts/arena_competition.py --balanced --rounds 1000 \
    --player MyAgent=path/to/model.zip \
    --npc cherry-picker --npc fortress --npc fast-follower --npc sprawler \
    --out results.json --csv decisions.csv
```

## Submit

Open an entry at
https://github.com/pencho-dobrev/retail-simulator-arena/issues/new with your
checkpoint and an `entry.json`. The checkpoint is a standard Stable-Baselines3 or
sb3-contrib `.zip` written by `model.save(...)`, from one of these algorithms:

| Algorithm | Policy | Library |
| --- | --- | --- |
| PPO, A2C | `MlpPolicy` | stable-baselines3 |
| RecurrentPPO | `MlpLstmPolicy` (stateful) | sb3-contrib |
| SAC, TD3, DDPG | `MlpPolicy` | stable-baselines3 |
| TQC | `MlpPolicy` | sb3-contrib |

The tooling detects the algorithm from the zip. A RecurrentPPO policy keeps its
recurrent state across the whole match.

The observation is a fixed 318-dimensional vector and the action a 44-dimensional
continuous vector, so `CnnPolicy` and `MultiInputPolicy` do not apply, and neither do
discrete-only algorithms such as DQN.

Two things hold for every algorithm:

- Scoring always predicts deterministically, whatever the policy was trained with.
- `VecNormalize` statistics are stored outside the zip and are not applied at scoring,
  so a policy trained with normalization must bake it into the network or not use it.

## Rules

The world is fixed — `arena_balanced_config()`, not competitor-chosen.

> **Final scoring is a tournament among the entries — there are no NPCs in
> it.** The organizer groups all entrants into matches of up to six seats and
> rotates the groups, the seats and the home regions across several matches, so
> every entrant meets varied opponents and starts from varied regions. Every
> entrant plays the same number of matches, and entrants are ranked by their
> aggregate `total_profit` across them. The seed and the home rotations are
> undisclosed.

The self-check above is local practice and is not how finals are scored: the
opponents at final scoring are the other entries, so a strong result against the
NPC field is not a prediction of your rank. The published home rotation is
`1, 5, 9, 13, 17, 21` and the tools default to `--seed 0`, but an agent tuned to a
fixed seed, a fixed set of starting regions or a fixed set of opponents will not
necessarily score well. Train for robustness across seeds, starting regions and
opponents.
