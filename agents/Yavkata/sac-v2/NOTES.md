# sac-v2 (YavkataSACv2)

- **Algorithm:** stable-baselines3 `SAC`, `MlpPolicy`, `policy_kind: "sac"`, stock checkpoint (no by-value functions), raw observations (scale folded in), bounded logits.
- **What checkpoint.zip is:** `checkpoint_b_warm.zip` from `train_v2.py` (BC + DAgger actor, warmed critics). **No SAC gradient phase**: 3,000 SAC steps (actor lr 3e-5, ent 0.003) on top of the warm start took the mean from +367k to -199k (measured), so the shipped zip is the warm start, as in v1.
- Date: 2026-10-29. Not committed.

## Top 2 (measured: 12 full6 matches, seeds 1,2 x 6 home rotations, baseline sac v1 in seat 0; plus all earlier runs/*.json)
Orcho > Krischo in every recorded run (runs/arena*, runs/cs/*: Orcho 1st, Krischo 2nd, 40+ matches; Orcho was 1st in all duo/six-seat files). SAC v1 3rd (mean 290k, rank 2.75, 0 wins).
| agent | mean profit full6 | stores @50/100/200/1000 | notes |
|---|---|---|---|
| Orcho | 747k (462k-991k) | 8.4 / 11.8 / 11.8 / 11.8 | 12 stores by round ~100, then fixed forever. price 2.0, marketing/assortment/promotion/service 1.0, research 0.2, wage 0.45+4*(1-happiness). Weakest at home 13 (472k vs 680-900k elsewhere) |
| Krischo | 369k (205k-522k) | 5.2 / 6.5 / 7.1 / 7.2 | METRO then a few AFFLUENT, stops at 7-10 stores, price ~1.98, service 0.95, wage 0.5, research 0.05. Weak at homes 9/13/17/21 (~300k vs ~500k at 1/5) |
Measured in earlier traces (runs/arena6_league.csv): Orcho cash 90 after store 4 on round 3, then stuck at 4 stores for ~50 rounds, ~70 profit/store/round at 12 stores.
Minor seats: Jack (3 stores, -31k), JackyT (-153k), YavRPPO (-193k): all cash-spiral losers, not threats.

## Holes found (each measured)
1. **Service 1.0 is over-spend.** Scripted Orcho-clone in the full6 field (seed 1, 6 homes): service 1.0 -> 471k; 0.7 -> 514k; 0.5 -> 531k; 0.3 (+price 1.9, research 0) -> 559k; **0.15 -> 598k, rank 1.00 (beat Orcho 545k in 6/6)**; 0.0 -> 582k. Orcho and Krischo both run service ~1.0/0.95, so they pay for service that does not earn it back. Also research 0 (+4k) and price 1.9 (+12k) helped slightly.
2. **Store cap 12 is a cliff, not a floor.** Same script: cap 10 402k, 12 471k, 14 268k, 16 81k (overhead exponent 1.5 plus cash crunch). Do not out-expand Orcho; its 12 is near the optimum. Late-store cash reserve did not help (451k).
3. **Orcho's opening is cash-gated (4 stores until ~round 50) and its policy never adapts** (fixed levers, fixed 12 stores, no reaction to rivals), so a leaner-cost copy with the same expansion beats it without any fancy tactic.
4. **Krischo stalls at 7 stores** and is home-dependent. Not attackable directly; just out-grow it.
5. **v1 weakness (the thing v2 fixes): clone drift.** The scripted demonstrator earns 471k in full6, its BC clone (sac v1) only 290k: v1 had 5.3 stores at round 100 and 6.0 at 200 (Orcho 11.8). DAgger relabelling of the clone's own states removes this.
Not measured: reactions to a rival entering a region (Orcho/Krischo do not appear to react; I did not probe), research effects, any late-game decay (Orcho profit/round flat after round 200 in the traces).

## How v2 exploits it
Demonstrator = Orcho's expansion (demand order, cap 12, open when cash >= 5000) with service 0.15, price 1.9, research 0, wage 0.45+4*(1-happiness) (`script_search.py`, found by `script_search{,2,3,4}.py` over the full6 field on seed 1). Cloned into the SAC actor (BC 3000 steps on 16k demo transitions) + 3 DAgger rounds (8k states each, beta 0.3), critics warmed 3000 TD steps, reward = profit + opening-race and debt penalties from `agents/Yavkata/sac/train_sac_bc.py`. Opponent field in training: real Orcho (p 0.85, else scripted stand-in), real Krischo (p 0.9), NPCs for the other seats.

## Training command
`python agents/Yavkata/sac-v2/train_v2.py --demo-json '{"service":0.15,"price":1.9,"research":0.0}' --demo-steps 16000 --clone-steps 3000 --dagger-rounds 3 --dagger-steps 8000 --critic-steps 3000 --sac-steps 0 --seed 1 --out agents/Yavkata/sac-v2/checkpoint_b.zip` (writes `checkpoint_b_warm.zip`; ~8 min).

## Results (eval_field.py, no NPCs, cand in seat 0 + Orcho, Krischo, Jack, JackyT, YavkataRPPO; homes rotated over all 6 METRO homes)
| candidate | seeds | n | mean | min | mean rank | wins | Orcho | Krischo |
|---|---|---|---|---|---|---|---|---|
| sac v1 (agents/Yavkata/sac/checkpoint.zip) | 1,2 | 12 | 290k | 169k | 2.75 | 0 | 747k | 369k |
| sac checkpoint_v2 | 1,2 | 12 | 256k | 121k | 2.92 | 0 | 893k | 487k |
| v2 warm, service 0.3 demonstrator, 2 DAgger | 1,2 | 12 | 367k | 190k | 1.92 | 2 | 462k | 244k |
| v2 + 3k SAC steps | 1,2 | 12 | -199k | -227k | 5.92 | 0 | | |
| **v2 shipped (service 0.15, 3 DAgger)** | 1,2 | 12 | **507k** | 291k | 1.67 | 5 | 540k | 288k |
| sac v1, held-out seed 3 | 3 | 6 | 306k | 165k | 2.83 | 0 | 677k | 355k |
| **v2 shipped, held-out seed 3** | 3 | 6 | **567k** | 316k | 1.50 | 3 | 560k | 250k |
Caveat: the demonstrator parameters were tuned on seed 1 (scripted runs), so seed 1 is in-sample; seeds 2 and 3 were not used for tuning. v2 beats Orcho on mean in the full6 field (seed 3: 567k vs 560k; seeds 1-2: 507k vs 540k, i.e. roughly level), and v1 by ~+200-260k. 6-18 matches: not a final rank. The Orcho figure drops when v2 is seated because v2 takes its regions. Not run: top2-only (3-seat) field, other minor-seat mixes, seeds beyond 1-3.

## Files
`checkpoint.zip`, `entry.json`, `train_v2.py`, `script_search.py` (demonstrator + search harness), `script_search2/3/4.py` (batches), `eval_field.py`, `trace_csv.py`, `checkpoint_b_warm.zip` (= checkpoint.zip), `checkpoint_new_warm.zip` (service 0.3 version). Scratch logs in `runs/v2/`.

## Open
Further SAC steps wreck the policy even at lr 3e-5 (critic/actor mismatch after a short warm-up); a real SAC phase would need a BC-loss term in the actor update and a much longer critic warm-up. Lower service than 0.15 and joint lever search not explored beyond the batches above.
