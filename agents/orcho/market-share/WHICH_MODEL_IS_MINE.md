# Which model is Orcho's (for whoever tests it in the playground)

**Add this file:** `agents/orcho/market-share/checkpoint.zip`
**Agent name to show:** `orcho-a2c` (version 3)
**Algorithm / policy:** A2C, `MlpPolicy` (declare `a2c-mlp`). Stable-Baselines3 2.9.2.
**Entry file:** `agents/orcho/market-share/entry.json` (team and contact still say TODO: fill them in before submitting).

## What it is
A normal SB3 A2C checkpoint trained by **imitation**: the scripted recipe `orcho-recipe2` in
`scripts/aggressive_npcs.py` played matches and the A2C policy was fitted to copy it
(`scripts/clone_recipe.py --expert orcho-recipe2 --see employee_happiness`, 26,000 samples incl. 3
DAgger rounds). It reads its own store counts and its staff happiness. The recipe:
- opens the **biggest-demand regions first** (METRO, then AFFLUENT, LOYAL, FRONTIER), up to **12 stores**
- price 2.0; marketing, assortment, promotion and service 1.0; research 0.2
- **smart pay**: staff pay 0.45 while staff are happy, raised up to 1.0 as happiness drops
  (`0.45 + 4 * (1 - happiness)`). The wage staff demand rises with your market share, so a fixed
  0.45 let happiness collapse to ~2% whenever we grew big; smart pay keeps it near 100%.

## Load and play it (team rule: exactly 6 seats)
```bash
.venv/Scripts/python scripts/arena_replay.py --rounds 1000 \
    --player orcho-a2c=agents/orcho/market-share/checkpoint.zip \
    --player krischo=agents/Krischo/rppo-lstm/checkpoint.zip \
    --player yavkata=agents/Yavkata/recurrent-ppo/checkpoint.zip \
    --player misho-jack=agents/misho/jack/checkpoint.zip \
    --npc cherry-picker+ --npc fast-follower+ --out runs/replay.html
```
`A2C.load("agents/orcho/market-share/checkpoint.zip")` also works. Deterministic, 318-D observation
in, 44-D action out, no VecNormalize, nothing read at decision time.

## Results (practice only, not the final scoring)
6 matches per lineup, every home rotation, seed 0:

| Lineup | orcho-a2c v3 | Wins | Krischo (2nd) |
|---|---|---|---|
| teammates + cherry-picker+ + fast-follower+ | **1,486,343** | **6 / 6** | 960,977 |
| teammates + 2 trained A2C rivals | **1,210,107** | **6 / 6** | 733,706 |

Earlier versions in the second lineup: v1 (lowest-index regions, 10 stores) 452,422, rank 2;
v2 (fixed pay 0.45) 1,213,277, 6/6. v3 matches v2 there and is +36% where v2's staff got unhappy.

## Known limits
- Apart from staff pay it follows one fixed plan and does not react to rivals. Against 2-3 copies of
  its own recipe every strategy we tried becomes a lottery: all copies rush the same METRO regions
  early, go into debt and most get restructured (see `scripts/strategy_search.py`).
- More than ~14 stores makes staff unhappy and collapses profit; it never goes past 12.
- Final scoring has no NPCs and uses undisclosed seeds and home rotations.
