# jacky-t

- **Owner:** misho
- **Algorithm / policy:** tqc (TQC, MlpPolicy, sb3-contrib)
- **Hypothesis:** an off-policy, sample-efficient learner with distributional critics (TQC) can match or beat the on-policy RecurrentPPO "Jack" at the same step budget, using the same 0.3/0.3/0.3/0.1 reward weights.
- **Training command:** `python agents/misho/jacky-t/train.py jacky_t --steps 171008`
- **sb3_version:** 2.9.2
- **Date trained:** not trained to completion (see status)

## Status: no checkpoint yet
The run was stopped by hand at about 100,000 of 171,008 steps, and `train.py` saves the model only at the end, so **no `checkpoint.zip` exists** and Jacky-T was **not evaluated**. There are no results, and Jacky-T was not compared with Jack.

Setup that was used (all in `train.py`):
- Same original 5-seat example env as Jack's last stage (one expansion-masked seat vs the four NPC archetypes, homes and world seed reshuffled each episode).
- Weighted training reward: profit 0.3, revenue 0.3, market share 0.3, loyalty 0.1 (scoring is still profit). No extra reward scaling, since the world's z-normalized reward is about unit scale.
- Off-policy handling copied from `scripts/train_vs_npcs.py`'s SAC path: argmax logit dims bounded to [-1, 1]; observation scale folded into the first network layers (checked: drift 1.2e-07), so the saved zip takes raw 318-D observations; replay buffer capped at the run length.
- 2 gradient steps per 4 env steps, because 1 per step ran at about 26 fps (over an hour). That is a lower update rate than SAC's usual setup and may undertrain it.

## Results (fill in after self-check)
| seeds / homes | score | share | stores | notes |
|---|---|---|---|---|
| none yet | | | | |

## What I learned / what to try next
- Add a periodic checkpoint callback to `train.py` so an interrupted run keeps a usable zip (the run above lost about 65 minutes of training).
- Re-run to 171,008 steps (about 1h45m at 28 fps), then play Jack and Jacky-T together as seats 5 and 6 in a six-seat match vs the four NPCs, over several home rotations.
