# Team agents

Four teammates, four agents, one comparison. Each agent lives in its own folder so we can run them side by side and see where each is better.

## Registry: which model belongs to whom

| Owner | Agent | Checkpoint to add in the playground | Algorithm | Notes |
|---|---|---|---|---|
| Orcho | `orcho-a2c` | `agents/orcho/market-share/checkpoint.zip` | A2C `a2c-mlp` | v5: METRO then AFFLUENT opened highest number first, 12 stores, smart staff pay, imitation-trained; see `agents/orcho/market-share/WHICH_MODEL_IS_MINE.md` |
| Yavkata | `YavkataSAC` | `agents/Yavkata/sac/checkpoint.zip` | SAC `sac` | behaviour-cloned warm start from the orcho-like recipe (no self-driven SAC steps); see `agents/Yavkata/sac/NOTES.md` |

Add a row here when you commit your own checkpoint.

## Folder layout

```
agents/
  README.md                 this file
  _template/NOTES.md        copy this into your agent folder
  <your-name>/
    <agent-name>/           e.g. orlin/margin-ppo
      checkpoint.zip        FINAL checkpoint only (SB3 .zip written by model.save)
      NOTES.md              owner, algorithm, hypothesis, command, results
      recipe.json           copy of the trainer's recipe.json (arena_train_player.py only)
      entry.json            what the submission would need (see README.md "Submit")
```

`*.zip` is gitignored everywhere except `agents/`, so checkpoints reach the team through git. Commit only the final round, never every round.

## Create your agent (every teammate)

1. `git pull`, then activate the venv (Python 3.11 or 3.12) and `pip install -e ".[rl,train,recurrent]"`.
2. In Claude Code run `/new-agent <your-name> <agent-name> <recipe>`, or copy `agents/_template/NOTES.md` into `agents/<your-name>/<agent-name>/` by hand.
3. Train it (recipes below). Train in the **balanced** world only.
4. Copy the final checkpoint into your folder as `checkpoint.zip` (see below).
5. Self-check it against the NPCs (command below), and record the result in `NOTES.md`.
6. `git add agents/<your-name> && git commit && git push`.

## Suggested split (so the four agents genuinely differ)

| Agent | Recipe | Why it is interesting |
|---|---|---|
| A | PPO, profit only (the "Margin" reference) | baseline |
| B | PPO, weighted reward (the "Share" reference) | does chasing share beat chasing profit? |
| C | PPO, profit + entropy bonus 0.02 ("Explorer") | more exploration |
| D | RecurrentPPO or SAC via `train_vs_npcs.py` | memory / off-policy |

Three of the four reference recipes come with the repo, so A to C differ by objective, not by seed. D is where your own ideas go.

## Training commands

PPO recipes (`scripts/arena_train_player.py`). The checkpoint of the last round is the usable one: with the default 6 rounds that is `round_5.zip`. Never submit `round_0.zip`.

```bash
python scripts/arena_train_player.py <name>_margin   --world balanced --seeds 0,0,42
python scripts/arena_train_player.py <name>_share    --world balanced --reward weighted --seeds 1,1,43
python scripts/arena_train_player.py <name>_explorer --world balanced --ent-coef 0.02 --seeds 2,2,44
```

Output lands in `runs/phase4_rl_confirm/<label>/`. Copy it in:

```bash
cp runs/phase4_rl_confirm/<label>/round_5.zip agents/<your-name>/<agent-name>/checkpoint.zip
cp runs/phase4_rl_confirm/<label>/recipe.json agents/<your-name>/<agent-name>/recipe.json
```

Your own trainer (PPO, RecurrentPPO or SAC against the NPC field) writes the zip straight into place:

```bash
python scripts/train_vs_npcs.py <name>_rppo --algo recurrent_ppo --steps 300000 \
    --out agents/<your-name>/<agent-name>/checkpoint.zip
```

## Self-check

```bash
python scripts/arena_competition.py --balanced --rounds 1000 \
    --player <agent-name>=agents/<your-name>/<agent-name>/checkpoint.zip \
    --npc cherry-picker --npc fortress --npc fast-follower --npc sprawler \
    --out runs/check.json --csv runs/check.csv
```

The first log line must say `arena-balanced`. One seating proves nothing below first place: repeat with different `--homes`.

## Comparing the four

Run `/compare-agents`. It seats all agents in the same match (no NPCs, like the final tournament), rotates the homes over several seeds, and writes `agents/COMPARISON.md`.
To watch one match, generate the replay page: `python scripts/arena_replay.py --player A=<zip> --player B=<zip> ... --out runs/replay.html` (not yet tried with trained players).

Rules that keep the comparison fair: same world (balanced), same rounds (1000), same seeds and home rotations for everyone, no edits to `src/`.

## Before submitting (team picks ONE)

One checkpoint per team. Write `entry.json` with team, agent_name, policy_kind (declare `recurrent-ppo-lstm` honestly), contact, training_recipe and sb3_version, then open the entry form linked in `README.md`.
