# Krischo-mlp-v9

PPO MlpPolicy (policy_kind `ppo-mlp`), behaviour-cloned from the v8 rule expert (kv-H) with a longer DAgger. mlp-v8 is unchanged.

- Teacher: expert v8 = v6 + wage from round 1, frugal opening (no research/wage in rounds 1-2), expansion scored by demand / (1 + 1.5 * rivals present). `agents/Krischo/scripts/krischo_v8.py`, params in expert_params_v8.json.
- Why v9: mlp-v8 collapsed mid-game from home 13 (two strong Orchos shrank its share, profit went negative, cash ran out, the game restructured it to 3 stores: -156k to -184k). The expert itself does not collapse in that field, so the loss was clone drift, not strategy.
- Train: `python agents/Krischo/scripts/krischo_mlp_v8.py --params agents/Krischo/mlp-v8/expert_params_v8.json --out runs/v8/mlp_v9.zip --iters 12 --rollouts 24 --epochs-iter 40 --workers 12 --zips-frac 0.85 --zips-k 4 --zips <Orcho v5, Orcho v4, Yavkata-sac, Jack, Jacky-T v1.1, mlp-v6>` (348,000 labelled states; no RL steps).
- Eval, 6 home rotations x seeds 5,6,7, field Orcho v5, Orcho v4, Yavkata-sac, Jack, Jacky-T v1.1: mean 460k / 413k / 457k (mlp-v8: 374k / 284k / 320k); worst match +168k (mlp-v8: -184k); first or second from every home.
- Caveat: the opponents in that test are the same agents it trained against. Not yet tested on unseen opponents.
