# rppo-lstm

- **Owner:** Krischo
- **Algorithm / policy:** recurrent-ppo-lstm
- **Hypothesis:** an LSTM keeps a compressed history of profit, share and rival moves, so it can learn the match phase (expand, defend, harvest) that a memoryless policy cannot see (the observation has no round counter).
- **Training command:** see "Run 0" below
- **sb3_version:** to fill in from `unzip -p checkpoint.zip system_info.txt`
- **Date trained:** 2026-10-28

## Run 0 (smoke test, default hyperparameters)
~10,000 steps with `scripts/train_vs_npcs.py` defaults: gamma 0.99, no observation-scale fold into the LSTM, 4 NPC opponents, 5 seats. This only checks that the pipeline works and gives a first self-check. It is far too short to be a real agent.

## Results (fill in after self-check)
| seeds / homes | score | share | stores | notes |
|---|---|---|---|---|
| seed 0, homes 1,5,9,13,17 (Krischo seat 0, 5 seats, 1000 rounds) | -7,067 | 0.110 | 4 | rank 4 of 5; beat only sprawler (-214,897). Leader cherry-picker 1,203,095 |

Run 0 stats: 10,240 steps in 155 s (65 fps, 4 envs on one core). Self-check match takes ~8 s.

## What I learned / what to try next
- See `.claude/STRATEGY.md` section 6 for the training checklist (gamma ~0.998, obs-scale fold, diverse opponents, checkpoints, held-out seeds).
