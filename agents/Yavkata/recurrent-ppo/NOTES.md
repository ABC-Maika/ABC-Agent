# recurrent-ppo

- **Owner:** Yavkata
- **Algorithm / policy:** recurrent-ppo-lstm (sb3-contrib `RecurrentPPO`, `MlpLstmPolicy`, separate actor/critic LSTM, 128 hidden)
- **Hypothesis:** Rewards (stockpile-free balanced world) depend on slowly moving state — loyalty, awareness, service score, wage happiness, warehouse load — and on noised competitor reads. An LSTM can integrate those over the 1000-tick match, so it should beat a memoryless PPO at the same budget.
- **Training command:** `python agents/Yavkata/recurrent-ppo/train_recurrent_ppo.py --steps 1000000`
- **Eval only:** `python agents/Yavkata/recurrent-ppo/train_recurrent_ppo.py --eval-only agents/Yavkata/recurrent-ppo/checkpoint.zip`
- **sb3_version:** 2.9.2
- **Date trained:** 2026-10-28 (seed 12, 300k steps; chosen over seed 13)

## Design choices
- **Env:** `NpcFieldEnv` from `scripts/train_vs_npcs.py` (one expansion-masked seat vs the 4 self-check NPCs, shuffled homes, new world seed per episode, truncation at 1000 ticks).
- **Observation:** 318 dims (the schema is the source of truth; `ObsLayout` splits it into Self / Market / Competitor). Trained on `obs / scale`; the scale is folded into the LSTM input weights before saving, so the zip takes raw observations (the scorer applies no `VecNormalize`). The script checks the fold changes actions by < 1e-4.
- **Action:** 44 dims. Gaussian mean starts at each continuous lever's default (price 1.0, service 1.0, spend levers 0) rather than 0, because actions are clipped to the lever bounds.
  Expansion (32 logits) and automation (3 logits) are argmax-decoded; PPO-family logits are left unbounded on purpose (see the docstring in `train_vs_npcs.py`).
- **Reward:** the env's per-tick profit (write-offs and warehouse penalties already subtracted) = the competition's `total_profit`. Scaled for training only with `VecNormalize(norm_obs=False)`; gamma 0.995.
- Levers not named in the original brief but present in the action space: `loyalty_spend`, `automation`.

## Status
Smoke-tested (4,096 steps): trains, folds the obs scale (drift 1e-8), saves, reloads and plays a full 1000-tick match on raw observations. Not yet trained for real. Next: a full run, then the self-check with `scripts/arena_competition.py --balanced`.

## Results (fill in after self-check)
| seeds / homes | score | share | stores | notes |
|---|---|---|---|---|

## zones variant (2026-10-30): `train_rppo_zones.py` -> `checkpoint_zones.zip` (not wired into entry.json)
sac-v2 recipe on RecurrentPPO: demonstrator (metro then affluent, cap 3, service 0.15, price 1.9, research 0) -> BC + 2 DAgger rounds -> value fit -> 50k PPO steps (lr 3e-5, clip 0.1) in the Orcho/Krischo field, best held-out snapshot.
Command: `.venv\Scripts\python.exe agents/Yavkata/recurrent-ppo/train_rppo_zones.py --steps 50000 --cap 3 --bc-epochs 60 --value-epochs 40`
Measured (eval_field full6, seeds 2,3, 12 matches): zones mean 20k (min -156k, rank 3.58, 0 wins), old checkpoint -153k (rank 5.25). Stores stay at ~2.75 all match. Orcho 1.15M / Krischo 754k in those matches, so a 3-store cap is far below sac-v2 (507k, cap 12).

## What I learned / what to try next
