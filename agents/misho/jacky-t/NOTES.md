# jacky-t

- **Owner:** misho
- **Algorithm / policy:** tqc (TQC, MlpPolicy, sb3-contrib)
- **Hypothesis:** an off-policy, sample-efficient learner with distributional critics can match or beat the on-policy RecurrentPPO "Jack", using the same 0.3/0.3/0.3/0.1 reward weights, and learn to stay in the top 2 against the teammates' agents.
- **Training commands:**
  1. `python agents/misho/jacky-t/train.py jacky_t --steps 171008` (stopped after the 80k snapshot; 5-seat NPC env, weighted reward)
  2. `python agents/misho/jacky-t/train_league.py jacky_t --resume runs/jacky_t/ckpt/jacky_t_80000_steps.zip --cap 100000` (20k more steps against the five trained agents on main)
- **sb3_version:** 2.9.2
- **Date trained:** 2026-10-29

100,000 steps in total: 80k against the four NPC archetypes, then 20k in a six-seat league of Krischo (two checkpoints), Yavkata, orcho and Jack. Reward: the world's weighted z-score (profit 0.3, revenue 0.3, share 0.3, loyalty 0.1), minus 0.25 per tick outside the top 2 by cumulative profit, plus 0.25 per tick while first with a lead over second above half of its own score (both after tick 50). Scoring is still profit. TQC settings: logits bounded to [-1, 1], obs scale folded into the network (drift 6.6e-07), 2 gradient steps per 4 env steps.

## Results (six-seat match, 1000 rounds, balanced world, seed 0: JackyT, Jack, Krischo, KrischoV8, Yavkata, Orcho)
| homes | score | share | stores | rank of 6 |
|---|---|---|---|---|
| 1,5,9,13,17,21 | -346,701 | 0.117 | 3 | 6 |
| 5,9,13,17,21,1 | -359,164 | 0.132 | 3 | 6 |
| 21,17,13,9,5,1 | -290,361 | 0.153 | 3 | 6 |

Orcho won all three (0.8M to 1.26M, 12 stores). Jack was about break-even (4th). Jacky-T finished last every time.

## What I learned / what to try next
- Jacky-T plays price about 0.60 (near the 0.5 floor) with marketing, assortment and promotion near maximum, so it buys share (0.12 to 0.15) at a loss. Staff happiness sits near 0.01 and cash ends negative. The winner prices at 2.0.
- The rank bonus and penalty were applied for only 20k steps and did not turn this around. The deep discount posture was learned in the first 80k steps against the scripted NPCs.
- It trained against these same agents, so the result is not inflated by a held-out test; it is simply poor.
- Next: train the league stage longer from a profit-first reward, or start TQC from Jack's behaviour; consider a price floor around 1.0 in the reward (e.g. penalise negative cumulative profit directly).
