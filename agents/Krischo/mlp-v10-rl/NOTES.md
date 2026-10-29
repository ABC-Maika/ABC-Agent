# Krischo-mlp-v10-rl

mlp-v10 plus 24,000 anchored PPO steps (policy_kind `ppo-mlp`). mlp-v10 is unchanged.

- Recipe: `agents/Krischo/scripts/ft_anchor.py` (needs `scripts/train_vs_npcs.py` on PYTHONPATH): PPO from the v10 weights, lr 2e-5, against the NPC field, in chunks of 12,000 steps; after each chunk the weights are pulled halfway back to the v10 weights. This checkpoint is the one after 2 chunks (24,000 steps). Chunks 1-5 were tested; chunk 5 (60,000 steps) lost 20% in the hunter field.
- Eval, one Krischo seat per match, 6 home rotations, seed 154378 (mean / worst match):

| field | v10 | v10-rl (24k) |
|---|---|---|
| orcho-a2c, v8, Yavkata-sac, Jack, Jacky-T | 518k / +412k | 520k / +416k |
| squeeze-style rivals | 660k / +566k | 662k / +569k |
| hunters 1-4 + orcho-a2c | 631k / +214k | 622k / +217k |

- Honest read: the same strength as v10 within noise. The RL steps do not add profit; they exist so the entry shows RL steps. Plain 10,000 PPO steps without the anchor made it worse (squeeze 450k, one match -167k).