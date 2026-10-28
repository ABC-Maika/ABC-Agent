# Krischo's agent

RecurrentPPO (LSTM) trained for the 25-region arena: a rule-based expert found by search and league play, distilled into the LSTM
(behaviour cloning, then DAgger), then fine-tuned with RL against a league of opponents.

- **Entry for comparisons:** `rppo-lstm/checkpoint.zip` (a copy of `checkpoint_v8.zip`). `policy_kind` is `recurrent-ppo-lstm`.
- **Notes and results:** `rppo-lstm/NOTES.md`. Full run history with every score: `.claude/RUNLOG.md`. Strategy: `.claude/STRATEGY.md`.
- **Expert parameters:** `rppo-lstm/expert_params_v1..v4.json` (the rules the zips imitate).

## Scripts (run from the repo root, venv active)
All use the organizer's code unchanged (nothing in `src/` is edited) and need `pip install -e ".[rl,train,recurrent]"`.

| Script | What it does |
|---|---|
| `scripts/krischo_optimize.py` | `search` (parametric expert search vs NPCs), `clone` (behaviour-clone an expert into an LSTM zip) |
| `scripts/krischo_league.py` | `league` (co-evolution vs smarter opponents), `dagger` (expert corrects the LSTM's own play), `evalzip` (zip vs league opponents), `tournament` (final-style ranking, 12 entrants) |
| `scripts/krischo_rl.py` | RL fine-tuning of an LSTM zip against the league (`--zoo` = opponent zoo) |
| `scripts/krischo_zoo.py` | opponent zoo (experts, NPCs, reactive, switching, passive, frozen zips); `evalzoo` tests a zip against held-out types |
| `scripts/krischo_compare.py` | six-seat, no-NPC comparison of our zips against five trained agents |
| `scripts/krischo_train_algos.py` | A2C / TQC against the NPC field (comparison baselines) |

## Compare with a teammate's agent
```bash
python scripts/arena_competition.py --balanced --rounds 1000 \
    --player Krischo=agents/Krischo/rppo-lstm/checkpoint.zip \
    --player Teammate=agents/<name>/<agent>/checkpoint.zip \
    --homes 1,5 --out runs/vs.json --csv runs/vs.csv
```
or run `/compare-agents`, which seats every `agents/*/*/checkpoint.zip` together without NPCs.

## Checkpoints
Only the best agent is kept in `rppo-lstm/`: `checkpoint.zip`, an identical copy of `checkpoint_v8.zip` (the team rule is "only the final checkpoint";
`checkpoint_v8.zip` can be deleted to keep a single 14 MB file). The earlier experiments (`checkpoint_v1..v7`, `v8_*k`, `v9*`) were moved to
`runs/archive_zips/` (gitignored); the run history in `.claude/RUNLOG.md` names them.
