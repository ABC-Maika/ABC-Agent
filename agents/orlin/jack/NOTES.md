# jack

- **Owner:** orlin
- **Algorithm / policy:** recurrent-ppo-lstm (RecurrentPPO, MlpLstmPolicy, sb3-contrib)
- **Hypothesis:** an LSTM can track how rivals behave across ticks, which the noisy competitor block hides in a single observation.
- **Training command:** `python agents/orlin/jack/train.py jack --steps 20000 --out agents/orlin/jack/checkpoint.zip`
- **sb3_version:** 2.9.2
- **Date trained:** 2026-10-28

Trained in a six-seat match: Jack plus five NPCs (the four geo archetypes plus one extra archetype drawn per episode), homes reshuffled and a fresh world seed every episode, expansion masking on, profit reward. This is a 20k-step smoke run, far below the 300k-step reference budget.

## Results (self-check vs the four NPCs, 1000 rounds, balanced world)
| seeds / homes | score | share | stores | notes |
|---|---|---|---|---|
| seed 0, homes 1,5,9,13,17 | -12,431 | 0.045 | 3 | rank 4 of 5 |
| seed 0, homes 5,9,13,17,21 | 30,300 | 0.229 | 10 | rank 3 of 5 |

cherry-picker won both (1.12M and 1.33M). The self-check cannot seat a duplicate NPC, so it plays 5 seats, not the 6 used in training.

## What I learned / what to try next
- 20k steps is only enough to show the pipeline works; the policy is near break-even.
- Next: train for 300k+ steps, try `--ent-coef 0.02`, and re-run with more `--homes` rotations.
