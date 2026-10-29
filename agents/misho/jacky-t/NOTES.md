# jacky-t (v1.1)

- **Owner:** misho
- **Algorithm / policy:** tqc (TQC, MlpPolicy, sb3-contrib)
- **Hypothesis:** v1.0 learned a loss-making discount posture (price 0.6). Behaviour-cloning the two best teammates' agents (Orcho and KrischoV8) into the TQC actor should give it their profitable posture (price about 2.0, staff paid) as a starting point.
- **Training commands (in order):**
  1. `python agents/misho/jacky-t/train.py jacky_t --steps 171008` (stopped at the 80k snapshot; 5-seat NPC env, reward 0.3/0.3/0.3/0.1)
  2. `python agents/misho/jacky-t/train_league.py jacky_t --resume runs/jacky_t/ckpt/jacky_t_80000_steps.zip --cap 100000` (20k steps vs the five trained agents on main) -> this is v1.0
  3. `python agents/misho/jacky-t/mimic.py collect orcho --episodes 10 --out runs/mimic/orcho.npz` and the same for `krischo-v8`
  4. `python agents/misho/jacky-t/mimic.py bc runs/mimic/orcho.npz runs/mimic/krischo-v8.npz` (12 epochs, MSE 0.26 -> 0.0065), then `python agents/misho/jacky-t/train.py x --finalize runs/jacky_t/bc/jacky_t_bc.zip --out agents/misho/jacky-t/checkpoint.zip` -> **this is the checkpoint (v1.1)**
- **sb3_version:** 2.9.2
- **Date trained:** 2026-10-29

v1.1 is the behaviour-cloned actor only. It has 100,000 RL steps behind its critics (v1.0's), plus a supervised fit to 20,000 recorded moves (10,000 each from Orcho and KrischoV8 playing Jacky-T's seat in six-seat matches). The two teachers are blended: they disagree on wages (about 0.5 vs 0.05), so the clone averages them there.

## Results (six-seat match, 1000 rounds, balanced world, seed 0: Jacky-T + Jack, Krischo, KrischoV8, Yavkata, Orcho)
| homes | v1.0 score | v1.1 score | v1.1 rank of 6 |
|---|---|---|---|
| 1,5,9,13,17,21 | -346,701 | +181,479 | 2 |
| 5,9,13,17,21,1 | -359,164 | +102,640 | 3 |
| 21,17,13,9,5,1 | -290,361 | +384,411 | 1 |
| 9,13,17,21,1,5 | n/a | -72,529 | 3 |
| 17,21,1,5,9,13 | n/a | +243,505 | 2 |
| 2,3,4,6,7,8 (non-metro) | n/a | +200,719 | 4 |

v1.1 mean over the six rotations is about +173,000 (profit in 5 of 6); v1.0 averaged about -332,000 over its three. Winners varied: Orcho, Krischo, KrischoV8 each won some rotations.

## Tried and rejected
A 20k-step profit-first fine-tune of the clone to 120k steps (weights 0.7/0.1/0.1/0.1, -0.25/tick while cumulative profit < 0, the top-2 rank terms, low learning rate) made it worse: -218,890 / -12,508 / -53,949 on the first three rotations (rank 6, 5, 5). Stale critics most likely pulled the actor away from the clone. Not used.

## Code that is written but NOT trained
`train_league.py --any-home` (six homes drawn from all 24 enterable regions each episode) and `--prime-weight W` (extra reward x the demand-weighted market share in METRO and AFFLUENT regions) are tested to run, but no model was trained with them. A planned 20k-step run (to 140k) was removed from the plan.

## Caveats and next steps
- The agents Jacky-T copies are also in the test field, and the result is one seed, six rotations: not a held-out test and not a predictor of the final tournament, which has other teams' agents.
- Orcho's and Krischo's own scores swing a lot depending on who is in the match (e.g. Orcho -179k to +1.08M), so ranks are noisy.
- Next: DAgger-style data (record the teacher on Jacky-T's own states), a longer fine-tune that keeps a clone loss in the actor update, and the `--any-home` / `--prime-weight` run.
