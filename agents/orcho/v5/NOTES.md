# orcho-v5 (A2C, league fine-tuned 10k)

Source: `runs/league/orcho-a2c-v5-rl10k.zip` (renamed to v5). orcho-a2c v5 clone, then 3,000 critic-only + 7,000 policy steps in the league (`scripts/league_finetune.py`, A2C, gamma 0.999). 10,000 steps = test budget.

Different file from `agents/orcho/ppo-v5` (that one is the PPO copy).
