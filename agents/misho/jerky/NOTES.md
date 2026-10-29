# jerky

- **Owner:** misho. **Algorithm:** TQC (`MlpPolicy`, 512x512, sb3-contrib). Research lever capped at 0.4 (Box bound in the saved action space).
- **Checkpoint on main:** session 4 (30k RL steps on top of a behaviour clone of KMlp8 / KMlp6 / Orcho), restored after later sessions (5 to 8) scored worse. It is a stock checkpoint (no function saved by value), so battle rooms accept it.
- **Recipe:** `mimic.py collect` the teachers in Jerky's seat, `mimic.py bc --new-net 512,512`, then 10k-step sessions of `runs/run_league.py` / `train_league.py` (anchored TQC, raw-profit reward, lead shaping, cash and happiness guards). `dagger.py` relabels Jerky's own states with a teacher's actions.

## Where it stands (12 fresh matches, seeds 5001 and 5002; Orcho, KMlp6, KMlp8, Krischo, YavSAC)
| version | mean | wins | rank |
|---|---|---|---|
| session 4 (this) | -87,863 | 0 | 3.67 |
| session 7 | -124,317 | 0 | 3.83 |
| session 8 | -222,179 | 0 | 5.17 |

Against the four NPCs of the site's self-check (six home rotations) the untrained clone wins every rotation (+2.6M mean), so the NPCs are not a test.

## Known weakness
Bimodal: in about two of six homes it grows to 12 stores and wins (+560k); in the rest it is cut back to 3 stores by the world's restructuring (cash below -10,000 for 50 rounds) and loses about -280k per match. It is decided in the first ~150 rounds (crowded METRO openings, ~100 cash short of the 5,000 store gate in round 3). Cutting spend at 3 stores does not help. See the sessions' traces in the conversation notes.
