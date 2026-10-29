# orcho-v6.2

Same start as v6.1 (clone of the scripted v6.1 recipe), RL fine-tuned 10k steps (A2C) against a zoo that now holds
the teammates' newest agents: Krischo mlp-v9 and mlp-v10-rl, Yavkata sac and sac-v2, Misho jerky.

## Results (6 seats: me, v6, v5, Krischo v10-rl, Yavkata sac-v2, Misho jerky; 6 home rotations per seed)

| Seeds | Agent in "me" | Mean / game | Wins / 12 | Avg place |
|---|---|---|---|---|
| 31, 32 | v6.1 | 207k | 7 | 2.75 |
| 31, 32 | **v6.2** | 213k | **8** | **2.33** |
| 34, 35 (never used before) | v6.1 | 73k | 3 | 3.17 |
| 34, 35 (never used before) | **v6.2** | **262k** | **6** | **2.42** (best seat) |

## What we learned about this field
- Every strong agent now opens ~12 stores, so only about two seats profit per game; the others crash to about -200k.
- The crash: cash below -5000 (the credit limit) makes the game clamp every lever and the wage to 0. Sales drop
  ~65%, staff happiness goes to 0, store overhead keeps running (~750/round lost), then 3% interest per round on the debt.
- Tried and rejected (contender_search.py --v6 5 / 6): "survival mode" (cut marketing/research, stop undercutting when
  cash runs low) changes nothing; no crowd undercutting costs ~55k/game; the scripted v6.1 recipe stays the best plan.
- Happiness bonus in the RL reward: keep it (`--happy-bonus 0` with the same settings: 5 wins instead of 8 on seeds 31/32).
  v6.1 and v6.2 used identical settings; the only change is the opponent zoo, so the stronger opponents made the difference.

## Limits
Both v6.1-family agents seated in the same game crowd each other and both lose (v6 then wins 10/12). Losses still
crash to about -250k; the agent cannot close stores, so the only defence is not being the one squeezed by round ~150.
