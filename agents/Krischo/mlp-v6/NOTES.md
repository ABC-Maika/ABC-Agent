# Krischo-mlp-v6

- **Owner:** Krischo
- **Algorithm / policy:** ppo-mlp (SB3 `PPO`, `MlpPolicy`, actor 256x256 ReLU)
- **Hypothesis:** the rule expert v6 is a pure function of the current observation (cash, stores, happiness, stores per region), so a stateless MLP can copy it exactly, unlike the LSTM clone, which smoothed its thresholds (no wage below 8 stores, buy only with 5,000 cash) and fell into the wage-debt spiral.
- **Training command:** `python agents/Krischo/scripts/krischo_mlp.py --params agents/Krischo/rppo-lstm/expert_params_v6.json --out runs/alt/mlp_v6.zip --episodes 60 --epochs 60 --iters 4 --rollouts 12 --epochs-iter 30 --orcho agents/orcho/market-share/checkpoint.zip --threads 4 --workers 8` (about 4 minutes; `checkpoint.zip` is the iteration noted below)
- **sb3_version:** 2.9.2
- **Date trained:** 2026-10-29

## The rules it copies (expert_params_v6.json)
12 stores (METRO first, then AFFLUENT), price 2.0, marketing, assortment and promotion 1.0, service 1.0, research 0.2, warehouse 0, no staff pay while it holds fewer than 8 stores, then 0.45 + 4 x (1 - happiness). In the scan (`runs/scan/`) cutting marketing, assortment or promotion below 1.0 sent the chain into the restructuring trap, and LOYAL stores lost money (13th store about -58 per tick).

## Results (balanced world, 1000 rounds, six seats vs Orcho v3, Jack, Yavkata, Jacky-T v1.1 and one pool expert, seeds 9100+)
| agent | matches | mean profit | worst match | rank | beat Orcho |
|---|---|---|---|---|---|
| expert v6 (rules, not a zip) | 12 | 625,203 | +488,642 | 1.17 | 10/12 |
| krischo rppo-lstm (`checkpoint.zip`) | 12 | 195,740 | -253,023 | 3.42 | 1/12 |
| **this agent** (`checkpoint.zip` = DAgger iteration 2) | 6 (seeds 9100-9105) | 574,150 | +456,607 | 1.67 | 2/6 |
| DAgger iteration 3 (not used) | 6 | 471,520 | +381,637 | 1.83 | 1/6 |

Six matches only: treat gaps between the iterations as noise. The expert row above is 12 matches, so the two means are not strictly comparable.

## What I learned / what to try next
- The LSTM clone (256, 512 units) never reached the expert (-5k to -33k); the stateless MLP reached 99% argmax accuracy after 35 s of training.
- Next: check it against the newest teammate agents and several home rotations; try expert v7 (service 0.2) as teacher after a held-out check.
