# Which model is Orcho's (for whoever tests it in the playground)

**Add this file:** `agents/orcho/market-share/checkpoint.zip`
**Agent name to show:** `orcho-a2c` (version 2)
**Algorithm / policy:** A2C, `MlpPolicy` (declare `a2c-mlp`). Stable-Baselines3 2.9.2.
**Entry file:** `agents/orcho/market-share/entry.json` (team and contact still say TODO: fill them in before submitting).

## What it is
A normal SB3 A2C checkpoint trained by **imitation**: the scripted recipe `orcho-recipe2` in
`scripts/aggressive_npcs.py` played matches and the A2C policy was fitted to copy it
(`scripts/clone_recipe.py --expert orcho-recipe2`, 26,000 samples incl. 3 DAgger rounds). It reads
only its own store counts. The recipe:
- opens the **biggest-demand regions first** (METRO, then AFFLUENT, LOYAL, FRONTIER), up to **12 stores**
- price 2.0; marketing, assortment, promotion and service 1.0
- staff pay (wage spend) 0.45, which keeps staff happiness at 100%; research 0.2

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
6 matches vs teammates + 2 trained rivals, every home rotation (`scripts/arena_rotate.py`, seed 0):

| Seat | Mean score | Wins | Mean rank |
|---|---|---|---|
| **orcho-a2c v2** | **1,213,277** | **6 / 6** | **1.00** |
| krischo | 733,868 | 0 | 2.00 |
| rival-a2c-s2 | 9,008 | 0 | 3.33 |
| rival-a2c-s1 | -2,698 | 0 | 4.00 |
| misho-jack | -12,382 | 0 | 4.67 |
| yavkata | -76,544 | 0 | 6.00 |

Version 1 (lowest-index regions, 10 stores) averaged 452,422 and ranked 2nd in the same test.

## Known limits
- It copies one fixed recipe. It does not adapt to rivals. If a rival uses the same recipe, the two
  split the same big regions and the result depends on where each starts.
- More than ~14 stores makes staff unhappy and collapses profit; it never goes past 12.
- Final scoring has no NPCs and uses undisclosed seeds and home rotations.
