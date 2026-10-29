# orcho-ppo-v5 (PPO copy of orcho-a2c v5)

v5's network copied into a PPO policy, then trained 10,000 steps (test budget). Kept next to `orcho-a2c`, which stays the main entry.

| Seed (lineup: mlp-v8, mlp-v6, LSTM v8, Yavkata SAC, Jacky-T) | orcho-a2c v5 | orcho-ppo-v5 |
|---|---|---|
| 0 / 1 / 2 / 3 | 563,928 / 593,024 / 557,593 / 651,897 | 545,812 / 537,758 / 525,714 / 600,475 |
| average | 591,611 | 552,440 (-6.6%) |

Opening gate (`scripts/opening_check.py`): 5 of 6 homes pass; home 2 is slow (6 stores at round 150), like v5. Beats Krischo mlp-v8 on every seed. Not trained further; 100,000 steps only when the team says it is ready.
