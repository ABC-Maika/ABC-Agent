# Which model is Orcho's (for whoever tests it in the playground)

**Add this file:** `agents/orcho/market-share/checkpoint.zip`
**Agent name to show:** `orcho-a2c` (version 5)
**Algorithm / policy:** A2C, `MlpPolicy` (declare `a2c-mlp`). Stable-Baselines3 2.9.2.
**Entry file:** `agents/orcho/market-share/entry.json` (team and contact still say TODO: fill them in before submitting).

## What it is
A normal SB3 A2C checkpoint trained by **imitation**: the scripted recipe `orcho-recipe4` in
`scripts/aggressive_npcs.py` played matches and the A2C policy was fitted to copy it
(`scripts/clone_recipe.py --expert orcho-recipe4 --see employee_happiness`, 26,000 samples incl. 3
DAgger rounds). It reads its own store counts and its staff happiness. No custom code in the zip.
The recipe is deliberately aggressive:
- **Highest-number-first expansion (v5):** opens the METRO regions, then the AFFLUENT regions, highest
  region number first (most agents open lowest number first, so this dodges their crowd), up to **12 stores**
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

## v5 vs v4 (6 home rotations each)
| Lineup | v4 | v5 |
|---|---|---|
| newest agents, seeds 0 / 1 / 2 | 575,755 / 565,325 / 562,749 | **689,158 / 664,373 / 690,702** (18/18 wins) |
| hunter bots + krischo-lstm | 667,465 (worst -315,140) | 673,150 (worst -263,994) |
| 5 Krischo mlp-v6 | 314,612 (worst -275,156) | **434,365** (worst +186,072) |
| old team lineup | 1,055,478 (6/6) | 1,004,086 (5/6) |
| organizer self-check | 3,233,077 | 3,140,507 |

## Results of v4 (practice only, not the final scoring; 6 matches per cell, every home rotation)
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

v4 vs 5 copies of v3 on more seeds: seed 2 525,009 (5 / 6 wins), seed 3 611,331 (5 / 6); best v3 copy
53,757 / 172,820. Reproduce with `scripts/arena_rotate_parallel.py` (`SEED=<n>`) and
`scripts/arena_copies_repro.py`.

## Tests that did NOT help (so nobody repeats them)
All against teammates + rivals / teammates + NPCs, 3 home rotations, vs the 12-store recipe:
| Idea | Result |
|---|---|
| Fewer stores (6, 8, 9, 10, 11) | 6 METRO only -30%; each extra AFFLUENT store adds about 50k; 12 is the peak |
| More stores (14, 16, 24) once cash is plentiful | 14: -11%, 16: -20%, 24: -90% (running cost rises faster than store count) |
| Third tier (LOYAL) instead of AFFLUENT | -28% to -45% |
| Price 1.97 (Krischo's) / 1.94 / 1.90 | 1.97 = tie; 1.90 +0.8% / +5.9% but uneven, not adopted |
| Krischo's warehouse and rival-price follow rules | equal or slightly worse |
| Cutting marketing/assortment from round 900 ("harvest") | about -10% |
| Opening only quiet regions / shared-city scoring | much worse vs teammates |

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

## Found after publishing v5 (2026-10-29)
- **v5 vs Krischo mlp-v8, 12 seeds x 6 home rotations (72 games):** v5 ~546k, mlp-v8 ~279k, v5 won 54 of 72 games, never lost money.
- **Organizer battle (6 rotations):** Krischo 928k, v5 ("ABC") 799k. v5 beat Krischo in 4 of 6 games; two games lost it: home 1 (-212k, 10 stores earning nothing, cash < 0, happiness 1.00 to 0.04, restructured to 3 stores) and home 17 (stuck at 6 stores for 170 rounds, 54k). Both: v5 opens the big METRO regions, which are the rivals' homes, so the demand is split three ways while Krischo opens the empty AFFLUENT regions first.
- **Hunter bots (`scripts/hunters.json`, round 3):** a bot with v5's opening and price 1.7 beats it (416k vs -111k) when 4 hunters crowd the same regions.
- **Scripted upgrade (not in this checkpoint):** `orcho-recipe5` = recipe4 + crowd switch (score = demand / (1 + 0.2 x rivals present)) + staff pay 0.40. About +6% and 11/12 wins vs v5's 9/12 in the scripted comparison, but the first network clone of it was 28% worse than v5 (opening copied badly), so it is NOT shipped.
- **Tests that did not help:** random multi-lever changes (-77% to -124%), marketing 0.95 (-26%), promotion 0.9 (-54%), 11 or 13 stores, aggressive region scoring (k = -0.2: -22%), frugal first 2 rounds (mixed), research 0 (no change).
- **RL fine-tune (10,000 steps, A2C and PPO):** 2.5% to 6.6% below v5; the first attempt collapsed because the script gave ignored inputs random weights (fixed: zero-start).
