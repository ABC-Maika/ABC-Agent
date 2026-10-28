# market-share

- **Owner:** orcho
- **Algorithm / policy:** ppo-mlp
- **Hypothesis:** An agent that optimises the market factor above everything else: winning regional share first, so rivals cannot take regions and profit follows from scale. Compared with the profit-only baseline, does share-first beat profit-first once seats are rotated?
- **Training command:** `python scripts/arena_train_player.py orcho_market --world balanced --reward weighted --seeds 1,1,43` (the existing "Share" recipe: profit 0.5, revenue 0.2, market share 0.2, loyalty 0.1). **Not yet share-first**: the trainer has no flag for the reward weights, so a heavier market-share weight needs a small change to `scripts/arena_train_player.py` (the weights are `weight_market_share` etc. in `RewardConfig`).
- **sb3_version:** <from `unzip -p checkpoint.zip system_info.txt`>
- **Date trained:** not trained yet

## Results (fill in after self-check)
| seeds / homes | score | share | stores | notes |
|---|---|---|---|---|

## What I learned / what to try next
