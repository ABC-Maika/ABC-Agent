# jack

- **Owner:** orlin
- **Algorithm / policy:** recurrent-ppo-lstm (RecurrentPPO, MlpLstmPolicy, sb3-contrib)
- **Hypothesis:** an LSTM can track how rivals behave across ticks, which the noisy competitor block hides in a single observation.
- **Training command:**
  1. `python agents/misho/jack/train.py jack --steps 20000` (six-seat env, profit reward)
  2. `python agents/misho/jack/train.py jack --env given --resume runs/jack_20k.zip --steps 100000` (original 5-seat example env, profit reward)
  3. `python agents/misho/jack/train.py jack --env given --resume runs/jack_120k.zip --steps 50000 --reward-weights 0.3,0.3,0.3,0.1 --out runs/jack_w_growth.zip` (this is the checkpoint)
- **sb3_version:** 2.9.2
- **Date trained:** 2026-10-28

About 170k steps in total, still below the 300k-step reference budget. Homes reshuffled and a fresh world seed every episode, expansion masking on. The last 50k steps trained on a weighted reward (profit 0.3, revenue 0.3, market share 0.3, loyalty 0.1); scoring is still profit.

## Results (self-check vs the four NPCs, 1000 rounds, balanced world, seed 0)
| model / homes | score | rank of 5 |
|---|---|---|
| 20k, homes 1,5,9,13,17 | -12,431 | 4 |
| 120k, homes 1,5,9,13,17 | 793,292 | 2 |
| 170k growth, homes 1,5,9,13,17 | 2,029,363 | 1 |
| 170k growth, homes 5,9,13,17,21 | 1,538,585 | 1 |
| 170k growth, homes 21,1,5,9,13 | 1,409,597 | 1 |

Other 50k variants from the 120k model, mean score over the same three rotations: profit-only 640k (2nd), share 0.5/0.2/0.2/0.1 1.37M (1st), profit-heavy 0.7/0.1/0.1/0.1 1.03M (1st), growth 1.66M (1st). The 120k baseline averaged 672k (2nd).

The self-check plays 5 seats (Jack plus four NPCs). Cherry-picker scored about 0.9M to 1.1M in these matches.

## What I learned / what to try next
- Pure profit reward stalled after 120k steps; weighting revenue and market share into the training reward is what helped.
- Caveat: Jack trained against these same four NPCs. The final tournament has no NPCs, so test it against other teams' agents with `/compare-agents` before trusting the score.
- Next: try more weightings (e.g. higher loyalty), train to 300k+ steps, and run more `--homes` rotations and seeds.
