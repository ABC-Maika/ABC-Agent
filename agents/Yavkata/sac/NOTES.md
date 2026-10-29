# sac

- **Owner:** Yavkata
- **Algorithm / policy:** sac (stable-baselines3 `SAC`, `MlpPolicy`, `policy_kind: "sac"`)
- **What `checkpoint.zip` is:** the behaviour-cloned warm start of `train_sac_bc.py` (a copy of `checkpoint_bc_warmstart.zip`), saved before any self-driven SAC step. The actor imitates the scripted "orcho-like" policy (price 2.0, marketing/assortment/promotion/service 1.0, research 0.2, wage `min(1, 0.45 + 4*(1-happiness))`, biggest regions first up to 12 stores) and the critics were warmed on the same transitions. It is a SAC network, but its play comes from the demonstrator, not from SAC learning.
- **Hypothesis:** The opening is a cash race (stores 3 and 4 on ticks 2-3). SAC from scratch loses it and stays at 3 stores; starting from the winning opening recipe avoids that.
- **Training command:** `python agents/Yavkata/sac/train_sac_bc.py --steps 150000 --out agents/Yavkata/sac/checkpoint_bc.zip` (the warm start is written first; this run was stopped before the SAC phase finished, and the warm start is the checkpoint kept)
- **sb3_version:** 2.9.2
- **Date trained:** 2026-10-29

## Files
| File | What |
|---|---|
| `checkpoint.zip` | the entry: the warm start of `train_sac_bc.py` |
| `train_sac_bc.py` | recipe that produces it (the warm start is written before the SAC phase) |
| `train_sac.py`, `train_sac_field.py`, `train_sac_league.py` | earlier SAC-from-scratch / field / league trainers. Their checkpoints were deleted (the from-scratch ones lost the cash race, the league one did worse than the warm start); `train_sac_league.py` also needs a warm-start zip that no longer exists |
| `COUNTER_STRATEGY.md`, `cs_analyze.py`, `cs_probe.py` | opponent analysis (Orcho, Krischo), battle analysis, probes |

## Results (six seats, seed 0, homes 1,5,9,13,17,21, 1000 rounds, `arena_competition.py --balanced`)
| field | this agent | notes |
|---|---|---|
| vs Orcho, Krischo + 3 NPC | 571,768 (2nd, 12 stores) | Orcho 810,829; Krischo 529,184 |
| vs Orcho, Krischo, Jack, YavkataRPPO + itself as a seat | 500,827 (2nd, 10 stores) | Orcho 773,758; Krischo 394,608 |
One seating, one seed: not a rank. Final scoring has no NPCs and undisclosed seeds and homes.

## Open
- Further SAC training from the warm start made it worse in the league (50k steps: -103k, 3 stores). If trained further: lower actor learning rate, longer critic warm-up, demonstrations from Orcho/Krischo as well.
- Rotate homes and seeds before trusting any number above.
