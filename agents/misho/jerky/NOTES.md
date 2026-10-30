# jerky

- **Owner:** misho. **Algorithm:** TQC (`MlpPolicy`, 512x512, sb3-contrib), untrained after cloning: the weights are a behaviour clone of a scripted recipe. Research lever capped at 0.4 (Box bound in the saved action space). Stock checkpoint (`FloatSchedule`, no function saved by value), so battle rooms accept it.
- **Checkpoint here: "lowest-first".** The clone of the lever lab's lowest-region-first recipe (`agents/misho/jacky-t/lever_lab.py`, `order="low"`): price 2.0, marketing, assortment and promotion 1.0, service 0.2, research 0.1, loyalty 0.4 (inert), wage 0.45 + 4 x (1 - happiness), warehouse 0, METRO and AFFLUENT regions lowest number first, up to 10 stores. **Main still holds the previous "lab-opt" checkpoint** (same levers, METRO then AFFLUENT highest region first, which is Orcho's opening); this one is local only until pushed.
- **Recipe:** `lab_collect.py play low10` (two runs of 24 games) and `lab_collect.py dagger low10` (30 games, the old lab-opt Jerky's states relabelled by the recipe), then `mimic.py bc ... --new-net 512,512 --max-rank 2 --opening-boost 8 --expand-boost 8 --expansion-ce 0.5 --research-cap 0.4 --log-std -2 --ent-init 0.02 --epochs 30`, then `train.py --finalize`. No RL steps: every 10k-step RL session started from a clone scored lower.

## Where it stands (battle format, seed 7001, six home rotations, one agent per team)
| field | lab-opt (main) | lowest-first (this) |
|---|---|---|
| Krischo v10, Yavkata SAC, Orcho v7, sprawler, fast-follower | +384,978 (worst +304,794, 1 win) | +393,064 (worst +280,859, 0 wins, Orcho v7 wins 6 of 6) |
| Krischo v13, Yavkata SAC-v2, Orcho v7, sprawler, cherry-picker | +203,179 (worst -202,101, 1 win) | **+405,783** (worst **+178,662**, 1 win; Yavkata SAC-v2 wins 4) |

The scripted recipe (lab seat, 60 games over two seeds, six fields) never lost money, worst +184k; the Orcho-style recipe lost money in 4 of 60.

## Known weaknesses
- **Leaves Orcho alone.** Orcho v7 made +469k and won all six matches in the first field; the Orcho-style opening used to take share from it. This Jerky rarely collapses but does not beat Orcho v7.
- **Hurts Krischo v10 and sometimes Yavkata SAC-v2.** Its lowest-first order collides with Krischo v10's AFFLUENT-first opening (2, 6, 10) and with Yavkata SAC-v2 in some fields; that helps Jerky only when those are the rivals that matter.
- **Stuck games remain possible** (3 to 4 stores, restructuring, about -250 per round); none occurred in the two fields above, but the scripted seat lost -180k in a few older games with the Orcho-style order.
- **No memory, no rival identity, no rival profit.** One seed per cell: gaps under about 100k are noise.
- Tried and rejected: cash buffers, rival-aware opening switch, adaptive recipe switch, late price cuts, spoiler tactics, cap 11 or 12, Orcho v7's levers, RL fine-tuning of the clone.
