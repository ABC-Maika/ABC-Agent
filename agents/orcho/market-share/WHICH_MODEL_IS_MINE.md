# Which model is Orcho's (for whoever tests it in the playground)

**Add this file:** `agents/orcho/market-share/checkpoint.zip`
**Agent name to show:** `orcho-a2c` (version 4)
**Algorithm / policy:** A2C, `MlpPolicy` (declare `a2c-mlp`). Stable-Baselines3 2.9.2.
**Entry file:** `agents/orcho/market-share/entry.json` (team and contact still say TODO: fill them in before submitting).

## What it is
A normal SB3 A2C checkpoint trained by **imitation**: the scripted recipe `orcho-recipe3` in
`scripts/aggressive_npcs.py` played matches and the A2C policy was fitted to copy it
(`scripts/clone_recipe.py --expert orcho-recipe3 --see employee_happiness`, 26,000 samples incl. 3
DAgger rounds). It reads its own store counts and its staff happiness. No custom code in the zip.
The recipe is deliberately aggressive:
- **Ring expansion:** opens the six METRO regions as a ring, always continuing from the end of the
  chain it already holds (so copies of it walk to different neighbours instead of all piling into
  regions 1, 5 and 9), then the AFFLUENT regions, up to **12 stores**
- price 2.0; marketing, assortment, promotion and service all at the maximum (1.0); research 0.2
- **smart pay:** staff pay 0.45 while staff are happy, raised up to 1.0 as happiness drops
  (`0.45 + 4 * (1 - happiness)`). The wage staff demand rises with your market share, so a fixed
  pay let happiness collapse; smart pay keeps it near 100%.

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

## Results (practice only, not the final scoring; 6 matches per cell, every home rotation)
| Lineup | orcho-a2c v4 | Wins | Next best |
|---|---|---|---|
| teammates + 2 trained A2C rivals (seed 0) | **1,055,478** | **6 / 6** | krischo 869,400 |
| teammates + cherry-picker+ + fast-follower+ (seed 0) | **1,376,438** | **6 / 6** | krischo 925,660 |
| teammates + 2 trained rivals, fresh seed 1 | **1,036,948** | **6 / 6** | krischo 827,471 |
| v4 vs 5 copies of v3 (seed 0 / seed 1) | **491,506 / 435,385** | 5 / 6, 4 / 6 | best v3 copy 84,770 / 49,335 |
| v4 + 2 v3 vs 3 Krischo | **366,399** | 4 / 6 | krischo2 205,030 |
| organizer self-check (4 NPCs, `--balanced`) | **3,233,077**, 1st | | cherry-picker -38,681 |

Version history: v1 (lowest-index regions, 10 stores) 452,422; v2 (biggest regions first, fixed pay
0.45) 1,213,277 but staff collapsed to ~2% happy in the NPC lineup; v3 (smart pay) 1,486,343.
v4 equals v3 against the teammates and is far better when several copies share a match.

## Why version 4
Teammates found that with several copies of our agent in one match the winner depended on the home
region, not on the agent. Tracing it: v3 opened regions "lowest number first", so every copy rushed
regions 1 and 5 at the start, the agents starting at 1, 5 and 9 had their home crowded, ran out of
cash and got restructured to 3 stores (about -280k). v4's ring order spreads the copies out (see
`scripts/strategy_search.py` and the organizer's `guides/TRAINING_TOURNAMENT_HINTS.md`, section 4).

## Known limits
- Apart from staff pay it follows one fixed plan and does not react to rivals. With 6 identical
  copies in one match it is still a lottery which two survive (the others go into debt early).
- More than ~14 stores makes staff unhappy and collapses profit; it never goes past 12. Buying
  extra low-value regions when cash is plentiful is untested.
- Final scoring has no NPCs and uses undisclosed seeds and home rotations.
