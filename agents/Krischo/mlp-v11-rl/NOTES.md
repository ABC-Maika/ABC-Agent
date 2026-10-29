# Krischo-mlp-v11-rl

mlp-v11 plus 24,000 anchored PPO steps (policy_kind `ppo-mlp`, an RL-trained checkpoint). mlp-v11 and mlp-v10-rl are unchanged.

- Recipe: `agents/Krischo/scripts/ft_anchor.py --init agents/Krischo/mlp-v11/checkpoint.zip --chunks 2` (PPO, lr 2e-5, 12,000 steps per chunk against the NPC field; after each chunk the weights are pulled halfway back to the v11 weights). This is the checkpoint after 2 chunks. mlp-v11 itself is behaviour cloning (DAgger) from expert kv-E3, see its NOTES.
- Eval, one Krischo seat per match, 6 home rotations, mean (matches under 100k):

| field | v10-rl | v11 | v11-rl |
|---|---|---|---|
| elite6 (orcho-a2c, v5, Jerky, sac-v2, sac v1), seed 154378 | 153k (2/6) | 318k (2/6) | 256k (2/6) |
| elite6, seed 2129219164 | 234k (2/6) | 269k (2/6) | 292k (2/6) |
| std (orcho-a2c, v8, sac, Jack, Jacky-T), seed 154378 | 518k (0/6) | 530k (0/6) | 528k (0/6) |

- Honest read: the best RL-trained entry of ours (elite6 average of the two seeds 274k vs 194k for v10-rl); about 7% below the pure clone v11 on that average. Fifth at homes 9 and 13 (-220k to -275k) like v11. Not tested in the hunters / squeeze fields. Still collapses in a 4-seat match. Jerky is the repo checkpoint, possibly older than the one on the site.