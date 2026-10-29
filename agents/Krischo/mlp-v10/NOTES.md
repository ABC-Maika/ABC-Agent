# Krischo-mlp-v10

PPO MlpPolicy (policy_kind `ppo-mlp`), behaviour-cloned from expert kv-M3 (`agents/Krischo/scripts/krischo_v8.py`) and started from the v9 weights, then two short DAgger steps. mlp-v9 is unchanged.

- Teacher kv-M3 = v9 teacher (kv-H) with wage 0.45 + 2 x (1 - happiness) (was 4x) and service 0.5 (was 1.0). Everything else as v9: frugal opening in rounds 1-2, expansion scored by demand / (1 + 1.5 x rivals present), up to 12 stores. Params in expert_params_v10.json.
- Chain: v9 weights -> 6 DAgger rounds with teacher kv-Q5 (wage k=2) -> 6 DAgger rounds with teacher kv-M3. Each round 12 six-seat rollouts, lr 3e-4. Opponents in the rollouts: hunters 1-4, contenders, orcho-a2c, orcho v5, Yavkata-sac, Jack, Jacky-T v1.1, mlp-v6. No RL steps. Trainer: `agents/Krischo/scripts/krischo_mlp_v10.py` (`--init` = start weights).
- Eval, one Krischo seat per match, 6 home rotations, seed 154378 only (mean / worst match):

| field | v9 | v10 |
|---|---|---|
| orcho-a2c, v8, Yavkata-sac, Jack, Jacky-T | 455k / +368k | 518k / +412k |
| 4 hunters + 1 orcho-a2c ("squeeze" style rivals) | -9k / -320k | 660k / +566k |
| hunters 1-4 + orcho-a2c | 28k / -289k | 631k / +214k |

- Why: v9 loses its share when rivals take the same regions (4 of 6 matches under 100k in the hunter fields). Cheaper wage and service 0.5 keep cash positive there.
- Tried and rejected: clones of experts with the lower contest penalty (S1/U1) collapsed to -250k in 6-seat matches; store cap 10 (Misho's number) cost 35-45% here; service 0.2 is similar to 0.5.
- Caveats: one seed (154378); opponents partly the same agents it trained against; not tested on unseen opponents; not yet compared to orcho v5 in 4-seat.