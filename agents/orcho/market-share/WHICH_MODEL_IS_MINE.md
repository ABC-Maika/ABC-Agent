# Which model is Orcho's (for whoever tests it in the playground)

**Add this file:** `agents/orcho/market-share/checkpoint.zip`
**Agent name to show:** `orcho-a2c`
**Algorithm / policy:** A2C, `MlpPolicy` (declare `a2c-mlp`). Stable-Baselines3 2.9.2.
**Entry file:** `agents/orcho/market-share/entry.json` (team and contact still say TODO: fill them in before submitting).

## What it is
A normal SB3 A2C checkpoint. It was trained by **imitation**: the scripted recipe in
`scripts/aggressive_npcs.py` (`orcho-recipe`) played matches, and the A2C policy was fitted to copy
what it saw and did (`scripts/clone_recipe.py`, with DAgger rounds). It reads only its own store
counts. The recipe: price 2.0, marketing 1.0, assortment, promotion and service 1.0, research 0,
wage spend 0.5 (keeps staff happy), and stop expanding at 10 stores.

## Load and play it
```bash
python scripts/arena_competition.py --balanced --rounds 1000 \
    --player orcho-a2c=agents/orcho/market-share/checkpoint.zip \
    --npc cherry-picker --npc fortress --npc fast-follower --npc sprawler \
    --out runs/check.json --csv runs/check.csv
```
`A2C.load("agents/orcho/market-share/checkpoint.zip")` also works. Deterministic, 318-D observation
in, 44-D action out, no VecNormalize, nothing read at decision time.

## Results (practice only, not the final scoring)
| Test | Result |
|---|---|
| Organizer self-check vs the 4 NPCs, seed 0 | 1st, 1,911,573 (cherry-picker 958,273) |
| 12 matches vs 5 trained rivals (2 seeds x 6 home rotations) | won 12 of 12, mean 1,193,470 |

## Known limits
- It copies one fixed recipe. It does not adapt to rivals. If a rival uses the same recipe, the two
  split the same regions and the result depends on where each starts.
- The strongest rivals we trained are 100,000-step A2C agents, not what other teams will bring.
- Final scoring has no NPCs and uses undisclosed seeds and home rotations.
