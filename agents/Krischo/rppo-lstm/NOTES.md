# rppo-lstm

- **Owner:** Krischo
- **Algorithm / policy:** recurrent-ppo-lstm (sb3-contrib `RecurrentPPO`, `MlpLstmPolicy`, default 256 hidden, separate actor/critic LSTM)
- **Hypothesis:** profit in the arena is decided by a few levers (price, marketing, assortment, expansion) and by when to expand and harvest. A rule-based expert found by search and league play is a strong teacher; an LSTM can imitate it, remember the match phase (the observation has no round counter) and then improve past it with RL against many opponents.
- **Training command:** pipeline below. The entry is `checkpoint.zip` = `checkpoint_v8.zip` plus a wage patch (step 5) and an expansion patch (step 6); `checkpoint_v8.zip` is the unpatched original.
- **sb3_version:** 2.9.2 (sb3-contrib 2.9.2, torch 2.14.1+cpu, Python 3.12)
- **Date trained:** 2026-10-28

## How the agent was built (all scripts in `agents/Krischo/scripts/`, run from the repo root)
1. **Expert search** (`krischo_optimize.py search`): a parametric rule-based policy (levers + where/when to expand) searched against the NPC field, validated on held-out seeds. Parameters: `expert_params_v1..v4.json`.
2. **League** (`krischo_league.py league`): co-evolution in six-seat matches without NPCs. Our expert is optimised against five opponents from a growing pool, then a best-response opponent is searched against it and added to the pool.
3. **Distillation** (`krischo_optimize.py clone`, `krischo_league.py dagger`): behaviour cloning of the expert into the LSTM (expansion up-weighted, observation scale folded into the LSTM input weights so the zip takes RAW observations), then DAgger (the LSTM plays, the expert labels its states).
4. **RL fine-tuning** (`krischo_rl.py`): RecurrentPPO on profit for 280k steps from the distilled agent, 11 parallel envs, seat 0 vs five expert opponents from the pool (the held-out test opponents excluded), gamma 0.999, lr 3e-5. The result is `checkpoint_v8.zip`.
5. **Wage patch** (`krischo_patch.py --wage 0.5`): v8 paid almost no wages (wage_spend 0.04), so staff happiness fell to 0 and wage write-offs ate profit. A constant +0.46 on the `wage_spend` output bias (no retraining, everything else unchanged) brings the mean to ~0.5 and staff to 100%.
6. **Expansion patch** (`krischo_patch.py --boost 1.2 --boost-classes 1`): v8 expanded only to the 6 METRO regions and then held forever (hold logit -1.05 vs -1.40 for the best free region, with up to 3M idle cash). +1.2 on the six AFFLUENT expansion logits makes it buy AFFLUENT regions after the METRO ones; it stops by itself around 9 stores. Result: `checkpoint.zip` (wage patch + this patch).

## Results (balanced world, 1000 rounds, six seats unless noted)
| Test | Result |
|---|---|
| Self-check vs the 4 NPCs (5 seats, seed 0, homes 1,5,9,13,17) | rank 1, score 2,592,400 |
| 12 matches vs 8 held-out smart expert opponents | mean 349,613, worst match +144,373, mean rank 2.17, 0 wins out of 12 |
| Final-style tournament (12 entrants, 24 matches, ranked by total profit) | **#1 with 5,135,424**; the rule expert is #2 with 3,449,343 (more wins, worst match -252k) |
| 16 matches vs a held-out opponent zoo (NPCs, reactive, switching, passive, trained zips, experts) | mean 639,143, mean rank 1.88, 5 wins; beats every non-expert type in 100% of pairings, the strong expert family in 44% |

**With the wage patch (`checkpoint.zip`) vs plain v8:** opponent zoo (16 matches) mean 764,787 vs 639,143, mean rank 1.38 vs 1.88, 10 wins vs 5, worst match 308,526 vs 262,431; seed-7 six-seat match 1,488,768 vs 1,327,552 (+12%). Against the 8 specialist experts built to beat our family it is weaker (mean 262,209 vs 349,613, worst match -287,694, 2 wins vs 0), the expected price of paying wages when margins are thin. The numbers above this paragraph are plain v8.

**With the expansion patch (`checkpoint.zip`, now) vs the wage-only version:** six seats vs Orcho (6 matches, homes rotated) 1,801,641 vs 554,213 (before 1,486,167 vs 750,703), ahead 6 of 6; AFFLUENT regions held 0.64 per region (Orcho 0.42), METRO 1.00. Opponent zoo (16 matches) mean 845,122 vs 764,787, mean rank 1.19 vs 1.38, 13 wins vs 10, worst match 349,223 vs 308,526; 12 matches vs the 8 held-out experts 317,634 vs 262,209, rank 2.67 vs 2.92, 5 wins vs 2.

**Tried and not adopted:** DAgger from the v5 expert with Orcho as opponent (worse, 0 of 6 wins), RL with a per-store bonus (100k, 300k steps, also with expansion-only exploration noise: the agent still never left METRO), larger AFFLUENT boosts (+2.0 buys AFFLUENT before METRO), forcing LOYAL (all six bought, profit 914k vs 1.80M: LOYAL stores do not pay), raising `research` (no change, the policy ignores the competitor signals).

Full history of every run (scores, diagnosis, what was optimised): `.claude/RUNLOG.md` on the author's machine (not shared via git).

## Known limits
- It still loses to the strongest specialist experts (high price cut, marketing 1.0, 12 stores): second place in most six-seat matches against them.
- It uses almost no `research` (output ~0.05, so the rival signals stay noisy); raising it by +0.4 and RL on an opponent zoo (v9) did not help.
- Trained and tested only with six seats and the balanced world; opponents were our own experts, reactive scripts, NPCs and earlier zips, not other teams' RL agents.

## What I learned / what to try next
- Training from scratch with RL (10k steps) lost money; starting from a searched expert was what made the agent strong.
- Cloning alone caps the agent at the expert; DAgger made it steadier and RL pushed its mean above the expert's.
- Next: counter the specialist family directly, self-play against its own checkpoints, check 3 to 5 seats.
