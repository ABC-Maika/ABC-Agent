# Counter-strategy for the SAC agent vs Orcho and Krischo

Everything below is in the balanced world, 1000 rounds, run with `scripts/arena_competition.py --balanced` or `ArenaMatch` (nothing in `src/` edited).
"Measured" = read from a run I made (scratch outputs in `runs/cs/`). "Inferred" = my reading, not tested.
Helper scripts (this folder): `cs_analyze.py` (per-seat summary of an arena CSV), `cs_probe.py` (scripted probe seat vs Orcho + Krischo).

## 1. What each opponent does (measured)

### Orcho (`agents/orcho/market-share/checkpoint.zip`, A2C, imitation of a script)
- Posture is constant for all 1000 rounds: price 2.00, marketing 1.0, assortment 1.0, **promotion 1.0**, service 1.0, loyalty 0, research 0.20, warehouse 0, wage 0.50 (rises when staff are unhappy; happiness 0.99).
- Expansion: grabs regions biggest-demand first, one every 1 to 5 ticks. Stores: 4 to 5 at tick 10, **12 by tick ~35 to 50**, then never changes. Order seen: homes/METRO (1, 5, 9, 13, 17, 21), then AFFLUENT (2, 6, 10, 14, 18, 22). 6 METRO + 6 AFFLUENT, never LOYAL or FRONTIER. It also enters other seats' home regions at tick 1 to 3.
- Profit per round is flat (about 2,100 to 2,250 vs NPCs, about 1,400 in a 2-seat duel). No late-game change.
- Results: vs 4 NPCs 2.2M to 3.2M (seed 1 with Krischo also seated 2,205,103). Duel vs Krischo 1.36M to 1.39M every time.

### Krischo (`agents/Krischo/rppo-lstm/checkpoint.zip`, RecurrentPPO, patched)
- Constant too: price 1.98, marketing 1.0, assortment 1.0, promotion 1.0, service 0.95, loyalty 0.02, research 0.05, warehouse 0.03, wage 0.51 (happiness 1.0).
- Expansion: METRO first (5 stores at tick 10, 6 METRO by tick ~19), then at most 1 to 4 AFFLUENT, **holds at 7 to 10 stores forever**. Cash piles up unused (1.5M at the end).
- Profit per round about 1,500 vs NPCs (7 stores), 950 to 970 in the duel.
- Results: duel vs Orcho 947k to 967k (8 to 9 stores), always second. With 4 NPCs seated too: 1.53M vs Orcho 2.21M (seed 1), 1.63M vs 1.73M (seed 8, 10 stores).
- Krischo's own NOTES claim it beats Orcho 6 of 6. I could not reproduce that: Orcho won 8 of 8 head-to-head matches here (duel homes 1,5 / 1,21 / 17,9 / 13,5 in both seat orders, seeds 1, 2, 7), by about 420k each.

### Both together
- 2 seats (Orcho vs Krischo): Orcho 1.38M, Krischo 0.96M, Orcho 12 stores vs Krischo 8, share 0.58 / 0.42. Same result under every home rotation tried (the match is insensitive to homes when only two seats exist).
- With the 4 NPCs: Orcho 2.2M / Krischo 1.5M (seed 1), 1.73M / 1.63M (seed 8, homes 17,21,1,5,9,13). The NPCs lose money in every seed except the cherry-picker (+127k to +161k when only one RL agent is seated).

## 2. Weaknesses (measured unless marked inferred)

1. **The first 50 rounds are a cash race, and losers never recover (the biggest finding).** Starting cash is 10,000, an expansion costs 5,000 and cash can fall only about 5,000 below zero. Any seat that cannot afford the next store stays at 3 stores and bleeds. Evidence:
   - Six seats, no NPCs, 3 Orcho + 3 Krischo copies, homes 1,5,9,13,17,21, seed 3: only **two** seats survived (Orcho copy 867k with 12 stores, Krischo copy 709k with 10). The other four sat at 3 stores with cash -5k to -21k and profit -240k to -284k each. The same 2-winner pattern in a second rotation (seed 4: 935k, 730k, four seats near -260k).
   - Scripted probe with marketing 0.5 or 0.3, or assortment 0.5, or promotion 0, or price 1.0 to 1.5: always stuck at 3 stores, -150k to -400k. Marketing 1.0 + assortment 1.0 + promotion 1.0 + price about 2.0 gets the opening revenue to afford stores 4, 5, 6.
   - Both opponents therefore die when 2+ other seats fight for the same METRO regions in the first ticks (inferred: they have no fallback). The NPC field in training never produces this pressure.
2. **Promotion is effectively mandatory.** Both opponents run promotion 1.0, and `promo_cost_per_unit` is 0 in the balanced world. A probe with promotion 0 earned revenue 725 vs 844/856 per round at tick 2 and lost money at 3 stores (-394/round at tick 5 vs +574 for Orcho). Do not let SAC learn to save on promotion.
3. **Krischo stops at 7 to 10 stores with idle cash** (1.5M unused). It never touches LOYAL or FRONTIER, and it reacts to nothing (levers constant). It collapses from 7 to **5 stores and ~238k** when a third seat contests METRO (every probe run with 6+ stores: Krischo 237k to 240k, vs 950k to 1.08M when the third seat was starved).
4. **Orcho is static at 12 stores** and its profit per round is flat; it has no late-game or reactive behavior. But it is the strong one, and sharing its 12 regions hurts it too: Orcho falls 1.37M to 0.57M to 0.83M when a probe with the same recipe sits beside it (see 3.1).
5. Both keep price at about 2.0 (the ceiling). I found no profit gain from undercutting (probe price 1.8 vs 2.0 within 5k; price 1.5 and below stuck in the cash race).
6. Loyalty spend changed nothing in the probe (0.3 vs 0: identical profit). Treat it as dead in this config (measured on one seed).

## 3. Probe results: what beats them (3 seats: Orcho, Krischo, scripted probe; seed 1, homes 1,5,9; probe = price 2.0, marketing 1.0, assortment 1.0, promotion 1.0, service 1.0, wage 0.5, expands biggest-demand-first up to N stores)

| Probe stores | Orcho | Krischo | Probe | Notes |
|---|---|---|---|---|
| 4 | 1,025,743 | 355,771 | 152,240 | too few |
| 6 (all METRO) | 826,865 | 237,918 | 438,240 | |
| **8 (6 METRO + 2 AFFL)** | 736,794 | 238,079 | **482,200** | best store cap tested |
| 10 | 648,678 | 238,423 | 362,148 | overlap with Orcho hurts both |
| 12 | 569,213 | 238,742 | 307,129 | mirror of Orcho: loses to Orcho |

Seed 2 / homes 9,13,1 gave the same shape (6 stores 429k, 8 stores 470k, 10 stores 345k). Variants at 8 stores: service 0.5 gave 515,436 (+33k), wage 0.7 gave 490,288, wage 0.3 gave 398,631, price 1.8 gave 477,777, marketing 0.7 or assortment 0.5 fell into the debt trap (-180k, -188k).

Reading: a copy of Orcho does not beat Orcho; capping at about 8 stores earns more than 12. The probe never beat Orcho (737k vs 482k) in these runs. Taking Krischo down and avoiding the debt trap is what is available. I did not find a lever that beats Orcho head-to-head (not measured: lower service, wage tuning jointly, entering LOYAL/FRONTIER, late harvesting).

## 4. Counter-strategy for SAC

### 4.1 Opponent field (highest value)
The current `NpcFieldEnv` has 4 NPCs that earn -350k to +160k. A SAC trained there never faces the cash race or a rival earning 1.4M. Change the field:
- Add scripted stand-ins built from the measurements above, as closures like `cs_probe.probe` (it already builds actions with `_structured_action`/`encode_action`):
  - **orcho-like**: price 2.0, marketing/assortment/promotion/service 1.0, research 0.2, wage `min(1, 0.45 + 4*(1-happiness))` (the recipe in `WHICH_MODEL_IS_MINE.md`), expand biggest-demand-first up to 12 stores. Randomize the cap in {8, 10, 12}.
  - **krischo-like**: price 1.98, marketing/assortment/promotion 1.0, service 0.95, research 0.05, warehouse 0.03, wage 0.5, METRO first, then AFFLUENT until a cap in {6, 8, 10}, then hold.
  - the 4 existing NPCs.
- Per episode sample 5 opponents: always include at least one orcho-like, with probability about 0.5 a krischo-like, fill the rest from the NPCs and the cap variants (so the policy does not overfit to one rival). Optionally with small probability seat the real frozen zips (Orcho is an MLP zip and cheap to run, Krischo is an LSTM and needs state handling, so scripted approximations are simpler).
- Code: new file `agents/Yavkata/sac/train_sac_field.py` that subclasses `NpcFieldEnv` (from `scripts/train_vs_npcs.py`) and overrides `reset` to build the opponent list; copy the per-episode roster logic from `reset`/`step` (there `self._npcs` is a list of predict closures called with `(obs, episode_start=...)`, so scripted closures slot in with no other change). Note that opponents in `NpcFieldEnv` are unmasked, so the stand-ins must not expand unaffordably (the Orcho/Krischo zips are masked): add a cash check (`obs["cash"] >= 5000`) before opening a store. Then pass the env class through `make_env_fns` (read-only script, so wrap it). `train_sac.py` can keep calling the existing `main` for the rest.

### 4.2 Reward
Keep profit only (default `reward="profit"`). Do not use the `balanced` or `marketing` bonuses (happiness 300 / coverage 30 / awareness 100 are far larger than per-round profit of a few hundred and can push the policy away from profit). The debt trap is already punished by profit. If SAC still falls into the trap after training on the new field, the only shaping I would add is a small penalty (about 1 to 5 reward points per tick) while cash < 0 in the first 100 ticks, tied to weakness 2.1.

### 4.3 Play rules the policy should discover (these are targets to check, not hard-coded)
1. Opening: marketing 1.0, assortment 1.0, promotion 1.0, price near 2.0 in ticks 1 to 50 while cash is thin; open stores only when cash >= 5,000 (masking already enforces affordability). Never trade opening revenue for savings.
2. Expansion: take the six METRO regions first, then 2 AFFLUENT (about 8 stores), and stop. Do not mirror Orcho's 12 stores. Open METRO regions that Orcho/Krischo have not yet taken first when possible (they take regions 1 to 5 within 3 ticks, so the contested ones go early).
3. Posture: price 2.0 (undercutting did not pay), wage near 0.5 to 0.7 with happiness near 1.0, service 0.5 to 1.0 (0.5 was +33k at 8 stores), research small, loyalty irrelevant.
4. Expected vs Krischo: it stops at 7 to 10 stores with idle cash; do not expect it to react, so no special posture is needed. Expected vs Orcho: it will not react either, and nothing measured here beats it head-to-head, so aim for second and large absolute profit.

## 5. Evaluation plan
Run after training, same commands for any checkpoint (replace `SAC` path). All with `--balanced --rounds 1000`:
1. 3 seats, no NPC: `--player SAC=agents/Yavkata/sac/checkpoint.zip --player Orcho=agents/orcho/market-share/checkpoint.zip --player Krischo=agents/Krischo/rppo-lstm/checkpoint.zip`, homes rotated over (1,5,9), (9,13,1), (17,21,5) in all seat orders, seeds 1 to 3 (about 20 s each). **Success: SAC profit >= 480k in every run (the scripted 8-store probe), no run in the 3-store debt trap, and SAC ahead of Krischo.** A stretch goal is >= 740k (Orcho level in that match).
2. 6 seats, no NPC (tournament-like): SAC + Orcho + Krischo + three more entries from `agents/*/*/checkpoint.zip` (e.g. `agents/misho/*`), homes 1,5,9,13,17,21 rotated, 4 seeds. The reference numbers I measured with Orcho and Krischo in such a six-seat match were: only 2 of 6 seats escaped the cash race (Orcho 867k to 935k, Krischo 709k to 730k). **Success: SAC finishes first or second in total profit over the rotation and never has a negative-profit match.**
3. Self-check vs 4 NPCs as a sanity check only (the final scoring has no NPCs; Orcho 3.2M and Krischo 3.26M in that field).
Sanity check on the opponent stand-ins: the scripted orcho-like and krischo-like closures should reproduce within about 10% the profits of the real zips in 2 and 3 seat runs above (Orcho 1.38M duel, Krischo 0.96M duel); verify this before training on them.

## 6. What I did not measure
- SAC itself (its run and `checkpoint.zip` were not touched).
- Region-level price/share interplay beyond store counts; whether any lever beats Orcho head-to-head; lower service, LOYAL/FRONTIER entry, a late harvest.
- The six-seat match with SAC and three other entries; my six-seat runs used 3 Orcho + 3 Krischo copies, and the two-winner outcome is from two seeds only.
- The scripted orcho-like/krischo-like fidelity (not yet built as training closures; `cs_probe.py` is the template).

## 7. Battle analysis (why SAC lost the six-seat match)

Data: `runs/arena6_field.csv/.json` (SAC `checkpoint_field.zip`, Orcho, Krischo, cherry-picker, fortress, fast-follower; seed 0; homes 1,5,9,13,17,21) and `runs/arena6.csv` (older `checkpoint.zip`). Per-tick dump: `runs/cs_battle_dump.txt`. Probe script: `runs/cs/battle_probe.py`. All numbers below are measured unless tagged (inferred).

### 7.1 Where it diverges: tick 2, and the cash race is lost by tick 3
| tick | Orcho (stores / cash / profit) | Krischo | SAC field |
|---|---|---|---|
| 1 | 2 / 9,958 / -42 (rev 605) | 2 / 9,942 / -58 | 2 / 9,615 / -385 (rev 618) |
| 2 | 3 / 5,221 / +263 (rev 1,109) | 3 / 5,180 / +237 | 3 / 9,299 / -317 (rev 422; its store from tick 1 is counted, none opened on tick 2) |
| 3 | 4 / 580 / +359 | 4 / 549 / +370 | 3 / 4,002 / -297 |
| 4-10 | 4 stores, +640 to +744 per tick, cash rises to 5,447 | same (+639 to +740) | 3 stores, -265 to -305 per tick, cash falls 3,697 to 2,040 |
| 11 | 5th store (cash 1,047) | 5th store | cash 1,777, never reaches 5,000 again |
| 18 | 6 stores, +1,191 per tick | 6 stores | cash below zero (-205) |
| 31 / 41 / 51 | 9 / 12 / 12 stores | 7 stores | 3 stores, cash -4,063 / -5,098 / -5,313 (overdraft cap about -5,000) |

Opening is a cash race: a store costs 5,000 and is gated on start-of-tick cash >= 5,000 (`core/affordability.py`). Orcho and Krischo open stores 3 and 4 on ticks 2 and 3, then earn +640 to +740 per tick, so store 5 comes at tick 11 and the snowball starts (12 stores at tick ~41, 2,000+ profit per tick). SAC opens stores only on ticks 1 and 3 (expansion picks 22 and 23), its cash is then 4,002, and it never again has 5,000.

### 7.2 Which lever differs (first 20 ticks, SAC field vs Orcho; measured means)
- Price 1.16 vs 2.0 (tick 1: 0.52; ticks 7-10 about 1.05). Orcho sits at the 2.0 ceiling.
- Promotion **0.014** vs 1.0. Service **0.05** vs 1.0 (0.0001 to 0.003 on ticks 5-30). This is the "promotion is mandatory" finding of 2.2 (revenue 422 vs 1,109 at tick 2).
- **Loyalty 0.91 vs 0**, **research 0.88 vs 0.20**, **wage 0.97 vs 0.45**: SAC pays near-maximum on three cost levers that give nothing here (2.6 measured loyalty as dead). Marketing (0.95) and assortment (0.79) are close to Orcho's, so they are not the cause.
- Net: -287 profit per tick at 3 stores vs +700 at 4 stores. SAC burns 270 to 300 cash per tick, so 4,002 at tick 3 is below zero by tick 18.
- Earlier `checkpoint.zip` (arena6.csv): same failure with other levers (price 1.6, promotion 0.57, service 0.44, wage 0.93, warehouse 0.80, loyalty 0.03). Cash 4,775 at tick 2 and 4,604 at tick 3, i.e. 225 to 400 short of the gate, then -255 to -316 per tick: stuck at 3 stores, -10k total. From tick 100 both checkpoints hover near zero profit per tick (field: -37 per tick, cash pinned near -5,000, restructuring keeps 3 stores).

### 7.3 Does it try to expand and get masked?
The recorded expansion action is 0 on every tick after tick 3 (field checkpoint), and with cash < 5,000 the gate makes any open impossible, so whether the policy still wants to expand is not visible in the CSV (inferred) and it does not matter: the failure is upstream. It is the cash race, caused by loss-making levers on ticks 2-20, not a masking problem. From about tick 30 SAC's levers drift toward the Orcho posture (promotion 0.89, assortment 0.99, service 0.5) but by then cash is about -5,000 and store 4 needs +5,000 pre-capex cash, so the trap is absorbing.

### 7.4 Control: scripted orcho-like policy in SAC's seat (measured; same field, homes, seed 0)
`runs/cs/battle_probe.py` replaces seat 0 with the scripted seat of `train_sac_field.py` (price 2.0, marketing/assortment/promotion/service 1.0, research 0.2, wage 0.5, loyalty 0, biggest-demand-first, cash gate 5,000). Real Orcho, Krischo and the three NPCs unchanged:
| seat 0 script | seat 0 profit | Orcho | Krischo | NPCs |
|---|---|---|---|---|
| cap 12, seed 0 | **+873,609** | 901,304 | 394,117 | -374k, -375k, -339k |
| cap 8, seed 0 | +814,489 | 1,073,192 | 393,846 | same |
| cap 12, seed 1 | +874,996 | 901,445 | 397,002 | same |
SAC field scored -45,567, so the same seat, home and seed is worth about +874k with the plain recipe (a 920k swing). Orcho falls from 2.2M to 0.90M when that seat plays properly. Stores at tick 51: script 6, Orcho 7, Krischo 5. Cap 12 beat cap 8 for the seat here (one scenario, seeds 0 and 1 for cap 12 only).

### 7.5 Minimal training plan (caller implements)
Problem size: the field run was 50,000 SAC steps (about 50 episodes; `ep_rew_mean` -65 at 4k steps; about 42 fps per the log). A randomly initialised max-entropy actor has to find a 5-lever combination that is profitable only if price, promotion, service, wage and loyalty are all right in ticks 1-2, with 50 episodes to do it (inferred). Mid-range lever noise lands in the debt trap (2.1: marketing 0.5, promotion 0 or price <= 1.5 each stuck at 3 stores).
1. **Behavior-cloning warm start (highest value).** Roll out the scripted orcho-like seat (cap 12; wage `min(1, 0.45 + 4*(1-happiness))`, otherwise as 7.4) as the learner in `FieldEnv` (same mask/gate), about 30 episodes x 1,000 ticks = 30k transitions. Pre-train the SAC actor on (obs, action) with MSE on the bounded output (expansion block = the chosen region), 5 to 10 epochs, load into `SAC`. Pre-fill the replay buffer with the same 30k transitions (rewards from the env) so the critic sees +700 per tick. Target after BC alone: close to the script's +874k (clone is somewhat worse, inferred).
2. **Opening action prior (safety net).** For ticks 1-60 of training episodes clamp the unproductive levers to the measured opponent values: loyalty <= 0.05, research <= 0.25, wage 0.40-0.60, promotion >= 0.9, service >= 0.9, price >= 1.9 (or blend 50/50 with the scripted opening). Release after tick 60; evaluate unclamped.
3. **Small opening reward penalty.** -0.005 policy units per tick (= -5 profit) while cash < 5,000 and stores < 5 in ticks 1-60. Basis: SAC loses 270-300 per tick there, but the payoff (store 4 at tick 3) is delayed and dwarfed by the +2,200 per tick later. Optional +0.5 per store opened before tick 60 (inferred). Keep total shaping under about 2 policy units per episode; profit stays the objective.
4. **Steps and entropy.** After BC, 300k steps (about 2 h at 42 fps single env; use more envs if available), fixed `ent_coef` 0.01 to 0.05 instead of learned from 0.3, `learning_starts` >= the 30k pre-fill so early noisy updates do not wipe the cloned actor. Evaluate every 50k steps with 7.6. (Step counts and entropy values are inferred, not tested.)
5. **Field.** Keep the current `FieldEnv`, but make the real Orcho appear in at least 60% of episodes: Orcho already holds 7 stores at tick 51 vs the learner's 6.

### 7.6 Check that it worked
`scripts/arena_competition.py --balanced --rounds 1000 --player SAC=agents/Yavkata/sac/checkpoint_field.zip --player Orcho=... --player Krischo=... --npc cherry-picker --npc fortress --npc fast-follower --homes 1,5,9,13,17,21 --seed 0`, then seeds 1-3 and rotated homes. Pass: SAC >= 4 stores by tick 5 and >= 6 by tick 51, cash never < 0 in the first 100 ticks, total profit >= 700k (script: 874k) and ahead of Krischo (394k). Fail signature: still 3 stores at tick 30.

### 7.7 Measured vs inferred
Measured: tables 7.1, 7.2, 7.4, the 5,000 gate, SAC's expansion picks and lever means, step count and fps. Inferred: that masking is irrelevant to SAC's intent (7.3), that 50 episodes is too few, the BC outcome, shaping size and entropy/step choices (nothing in 7.5 has been trained). Not measured: LSTM-Krischo effects on BC, home rotations for the scripted seat beyond region 1, any SAC retraining.
