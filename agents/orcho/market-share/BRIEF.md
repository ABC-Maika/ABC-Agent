# Market-share agent: what matters for this part

Pulled from the README, the competition site (/compete, /leaderboard) and the reward code. Source files named so you can check.

## 1. How you are judged (this decides everything)
- **Final ranking = aggregate `total_profit`** over several matches, no NPCs, seats and home regions rotated, seed undisclosed. Market share is **not** scored. (README "Rules")
- A seat's score = cumulative post-penalty profit (after wage and warehouse penalties). The site says this equals the reward an entrant trains on, but only in the default profit mode.
- So a market-share agent is a **means to profit**, not a goal. Share only helps if it converts to profit. Our own demo showed it: fast-follower held 46% share but made a fraction of cherry-picker's profit, and on the site's leaderboard the "Share" RL player ranked first with 26% share while fast-follower had 17% and a negative score.

## 2. What "market share" is in the code (`src/retail_simulator/core/reward.py`)
- `market_share = your units / total units` across all regions, per tick (0 if no demand). Demand-weighted, so big regions (METRO) count more.
- Per region the share is capped: in the balanced world `max_share_per_region = 0.6`. Nobody can hold more than 60% of a region; the excess goes to the other present retailers. A region only you are in tops out at 60% (the leaderboard shows exactly 60.0% for lone seats).
- Loyalty is a separate component (demand-weighted loyalty across regions).

## 3. The reward you can train with
- `--reward profit` (default): reward = profit. 
- `--reward weighted`: z-scored blend, weights fixed in `RewardConfig`: profit 0.5, revenue 0.2, **market_share 0.2**, loyalty 0.1. Components are normalised (running mean/std) before weighting, so 0.2 is not "20% of profit".
- The trainer has **no flag for the weights**. Share-first needs a small edit to `scripts/arena_train_player.py` (override `weight_market_share`; the four weights must sum to 1.0 or it raises).
- Recipe seeds already in the repo for a share-ish agent: Share = `--reward weighted --seeds 1,1,43`, entropy 0.0.

## 4. Levers you control (44-D action, 9 continuous + 35 argmax-decoded logits)
price index (x1.05 to x2.00 seen in matches; true range unchecked), marketing, assortment, promotion, service level, loyalty spend, research, wage spend, warehouse invest, expansion (target region, 0 = hold), automation (tier 0..2).
- Share-driving levers: price (lower = more units), marketing, assortment, promotion, service, loyalty spend, and **expansion** (more regions = more demand pool).
- Costs that punish chasing share: promotion/marketing spend, **wages and warehouse penalties**, and growth overhead that rises faster than linearly with store count (`overhead_exponent 1.5`, reference 4.5 stores).
- Expansion: one-time **5,000** per new region, only where you have no store, region 0 is never enterable, and you must have the cash (starting cash 10,000, so about two opens at the start).
- Promotion is contested in the balanced world (`promo_contest=1`).

## 5. World facts you will exploit
- 25 regions, 24 enterable. Class cycle from region 1: METRO (biggest), AFFLUENT, LOYAL, FRONTIER. Homes for seats are METRO regions 1, 5, 9, 13, 17, 21; up to 6 seats.
- Match = 1,000 consecutive rounds, never ends early. Cash can go only about 5,000 below zero (credit limit); overdraft rate 3%. After 50 ticks of debt a **restructure** can force you down to your best 3 regions (`restructure=50/3/5000`). Over-expanding on credit can get you cut back.
- Observation is 318-D; action space does not grow with regions. Expansion is masked: you cannot pick a region you already hold or cannot afford.

## 6. Hard rules for the entry
- Train in `--world balanced`. Train **robustly**: different seeds, home regions and opponents. A policy tuned to one seed or one seating will not hold in the tournament.
- One checkpoint per team. SB3/sb3-contrib `.zip`, deterministic at scoring, `VecNormalize` stats are not applied, no network/file reads in predict.
- `entry.json`: team, agent_name, policy_kind, contact, training_recipe, sb3_version.
- Round 0 checkpoint is untrained; use the last round (`round_5.zip` at the default 6 rounds).

## 7. Open questions for you
- Share-first reward or profit with a share bonus? (Final score is profit; I would keep profit's weight highest and treat share as shaping.)
- Do you want a weights flag added to the trainer?
- Deadline: still "TBA".
