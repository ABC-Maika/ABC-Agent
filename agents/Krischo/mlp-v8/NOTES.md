# Krischo-mlp-v8

PPO MlpPolicy (policy_kind `ppo-mlp`), behaviour-cloned from the v8 rule expert + 5 DAgger rounds. mlp-v6 is unchanged.

Expert v8 (agents/Krischo/scripts/krischo_v8.py, variant kv-H; params in expert_params_v8.json) = v6 plus:
- wage paid from round 1 (0.45 + 4 * (1 - happiness)) instead of "no wage until 8 stores"
- frugal opening: no research or wage in rounds 1-2, so cash stays above the 5,000 store gate on round 3
- expansion score = region demand / (1 + 1.5 * rivals present): empty regions beat crowded ones

Train: `python agents/Krischo/scripts/krischo_mlp_v8.py --params agents/Krischo/mlp-v8/expert_params_v8.json --out runs/v8/mlp_v8.zip --iters 5 --rollouts 12 --workers 10 --zips <Orcho, Yavkata-sac, Jack, Jacky-T v1.1, mlp-v6>`
Eval (6 home rotations x seeds 5,6, field = the original checkpoints): expert kv-H mean 325k, 12/12 wins, worst +176k; this MLP mean 282k / 300k, worst +101k.
