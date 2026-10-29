# jerky (v1, 100k steps)

- **Owner:** misho. **Algorithm:** TQC (`MlpPolicy`, 512x512, sb3-contrib). Research lever capped at 0.4 (Box bound in the saved action space).
- **Recipe:** `mimic.py collect` Orcho + KMlp6 (16 episodes each, kept only rank <= 2 with profit > 0), `mimic.py bc --new-net 512,512 --wage-floor 0.5 --research-cap 0.4`, then `python runs/run_league.py jerky2 --cap 300000 ... --raw-profit-scale 1e-3 --lead-shaping 1.0 --anchor-data ...` (anchored TQC, 30k critic warm-up, 13-opponent pool incl. personas). **Stopped at 100k steps by request** (planned 300k); the 100k snapshot was folded with `train.py --finalize`.
- **Reward fix:** the world's weighted reward z-scores per seat, so steady +1000 and steady -1000 profit both score about -36; Jerky trains on raw profit plus a fading lead bonus.

## Result (arena_rotate, seeds 3001,3002 x 6 rotations = 12 matches; Orcho, KMlp6, Krischo, KrischoV8, YavSAC)
| seat | mean | wins | rank |
|---|---|---|---|
| Orcho | +413,795 | 10 | 1.83 |
| KrischoV8 | +104,837 | 0 | 2.75 |
| **Jerky** | **-52,836** | **2** | **4.33** |
| KMlp6 | -82,216 | 0 | 3.42 |
| YavSAC | -83,226 | 0 | 3.92 |
| Krischo | -216,117 | 0 | 4.75 |

Jerky's two wins are both with homes starting at 21. High variance (-353k to +669k). It is not better than Orcho and was not compared with v1.1 on these seeds. Not evaluated beyond this one tournament; the 300k plan was cut short.
