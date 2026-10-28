# tqc

- **Owner:** Yavkata
- **Algorithm / policy:** tqc (sb3-contrib `TQC`, `MlpPolicy`, `policy_kind: "tqc"`)
- **Hypothesis:** The balanced world's profit signal is noisy (perceived competitor block, rival play). TQC's distributional critics drop the most optimistic value estimates and SAC-style entropy keeps exploration up, so at an equal step budget it should beat PPO and RecurrentPPO.
- **Training command:** `python agents/Yavkata/tqc/train_tqc.py --steps 300000 --out agents/Yavkata/tqc/checkpoint.zip`
- **sb3_version:** <from `unzip -p checkpoint.zip system_info.txt`>
- **Date trained:** <yyyy-mm-dd>

## Design choices
- **Env:** `NpcFieldEnv` from `scripts/train_vs_npcs.py` (one expansion-masked seat vs the 4 self-check NPCs, shuffled homes, new world seed per episode, truncation at 1000 ticks). Same setup as that script's SAC path.
- **Action:** the 35 argmax-decoded logit dims are bounded to [-1, 1] (off-policy needs a bounded space); scoring decodes them identically.
- **Observation:** all 318 dims. Trained on `obs / scale`; the scale is folded into the first layer of the actor and every critic, so the zip takes raw observations. The script aborts if the fold changes predictions by > 1e-5.
- **Reward:** per-tick profit times 1e-3 (off-policy algorithms store the raw reward, so `VecNormalize` would do nothing). `gradient_steps=-1`, `buffer_size=min(steps, 1e6)`.

## Status
Smoke-tested (2,000 steps): trains, folds the obs scale (drift 2e-7), saves. Measured speed about 27 steps/s on CPU, so 300k steps is roughly 3 hours. Not yet trained for real.

## Results (fill in after self-check)
| seeds / homes | score | share | stores | notes |
|---|---|---|---|---|

## What I learned / what to try next
