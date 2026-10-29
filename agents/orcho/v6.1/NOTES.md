# orcho-v6.1

v6 plus cautious expansion (`k=1.0`: avoid regions where rivals already sit), RL fine-tuned 10k steps (A2C).

## Exam (seeds 31, 32 x 6 home rotations; vs v5, Krischo v9, mlp-v8, Yavkata sac, Misho jerky)

| Version | Wins / 12 | Losses |
|---|---|---|
| v6.1 clone (scripted plan, before RL) | 11 | seed 32 home 9 |
| **v6.1 RL 10k (this checkpoint)** | 8 | homes 9 and 13, both seeds |
| v6 RL 10k | 9 | |

## Known problem
The RL fine-tune drifts the wage lever from 0.40 to 0.8-1.0: `league_finetune.py` pays 300/tick for happy staff,
about 3x the real profit per tick. High wages, then losses from ~t90, debt, then restructure to 3 stores.
`--happy-bonus` now switches it off; next version retrains with it.
