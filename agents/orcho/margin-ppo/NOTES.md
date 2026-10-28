# margin-ppo

- **Owner:** orcho
- **Algorithm / policy:** ppo-mlp
- **Hypothesis:** Baseline: PPO with a profit-only reward (the "Margin" reference recipe). Score is profit, so train on exactly that.
- **Training command:** `python scripts/arena_train_player.py orcho_margin --world balanced --seeds 0,0,42`
- **sb3_version:** <from `unzip -p checkpoint.zip system_info.txt`>
- **Date trained:** not trained yet

## Results (fill in after self-check)
| seeds / homes | score | share | stores | notes |
|---|---|---|---|---|

## What I learned / what to try next
