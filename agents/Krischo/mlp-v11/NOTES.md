# Krischo-mlp-v11

PPO MlpPolicy (policy_kind `ppo-mlp`), behaviour-cloned from expert kv-E3 and started from the mlp-v10 weights, 6 DAgger rounds, 6-seat matches only. mlp-v10 / v10-rl are unchanged.

- Teacher kv-E3 (`agents/Krischo/scripts/krischo_v8.py`, params in expert_params_v11.json): v9 expansion rule (frugal opening, demand / (1 + 1.5 x rivals)), up to 12 stores, wage 0.45 + 2 x (1 - happiness), and Yavkata's lean numbers: service 0.15, price 1.9, research 0.
- Opponents in the DAgger rollouts: hunters 1-2, contenders, orcho-a2c, orcho v5, Yavkata sac-v2 and sac v1, Jerky, Jack, Jacky-T v1.1. Trainer: `agents/Krischo/scripts/krischo_mlp_v10.py --init`.
- Eval, one Krischo seat per match, 6 home rotations:

| field | v10-rl | v11 |
|---|---|---|
| elite6 (orcho-a2c, v5, Jerky, sac-v2, sac v1), seed 154378 | 153k | 318k |
| elite6, seed 2129219164 | 234k | 269k |
| std (orcho-a2c, v8, sac, Jack, Jacky-T), seed 154378 | 518k | 530k |
| squeeze rivals, seed 154378 | 660k | 409k |
| hunters + orcho-a2c, seed 154378 | 631k | 477k |
| 4 seats (orcho-a2c, Jerky, sac-v2), seed 154378 | -286k | -278k |

- Honest read: best of ours against the real agents in a 6-seat match (first at homes 1, 5, 17, 21, +330k to +730k; fifth at homes 9 and 13, -220k to -270k). Weaker than v10-rl against scripted hunters / contenders. Still collapses in a 4-seat match (all homes ~-280k): the expert kv-E3 earns +495k there, the clone does not copy it; a variant trained with 40% 4-seat rollouts only reached -74k and lost 6-seat strength, so it was dropped.
- Jerky here is the repo checkpoint (agents/misho/jerky); the site's Jerky scored +865k in battle Z6GB-QEH5, so this one may be older. One or two seeds only.