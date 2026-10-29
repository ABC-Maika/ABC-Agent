# orcho-v6 (A2C, RL-trained 10k steps)

A2C `MlpPolicy` (`a2c-mlp`), SB3 2.9.2. RL-trained on profit for 10,000 steps in the league
(`scripts/league_finetune.py`), warm-started from a network copy of the scripted v6 recipe.

## What is new vs v5
- **Crowd pricing:** price 2.0 while our regions are quiet; it drops by 0.1 per extra rival in our average
  region (floor 1.7). Margins are thin (about 100 profit a round on 2,700 revenue), so a small undercut
  where it is crowded pushes rivals into debt; after 50 rounds of debt the game cuts them to 3 stores.
- **Service 0.2** (teammate's find): service brings no customers here; it only sets how much of the
  region's demand you can supply (50% + 50% x service) and adds up to 10% to the cost of goods. 0.2 supplies
  exactly 60% (the share cap), so no lost sales and goods are ~8% cheaper.
- Staff pay 0.40 base (+4 x (1 - happiness)), research 0.1, regions scored demand / (1 + 0.2 x rivals).
- Inputs: its own store counts, staff happiness, rivals present per region.

## How it was made
1. `scripts/contender_search.py --v6 1/2/3`: 15 scripted ideas, 5 lineups x 2 homes. Winner `crowd+svc`
   (= `CONTENDERS["v6"]` in `scripts/hunter_bots.py`): 502k per game vs 253k for the v5 recipe, never worse
   in any lineup. Did NOT help: cutting marketing in debt (-48% worst lineup), no borrowing for stores,
   price war when rich, store cap 10 (-50% vs 5 Krischo copies).
2. `scripts/clone_recipe.py --expert contender-v6 --see employee_happiness --see "others_present_region_*"
   --episodes 10 --dagger-rounds 3 --dagger-episodes 6 --opening-repeat 3 --rivals <5 lineups of real
   checkpoints and hunters>` (28,000 states; seat 0 plays real checkpoints, not the NPC field).
3. `scripts/league_finetune.py --start runs/v6/orcho-v6-clone.zip --algo a2c --steps 10000 --warmup 3000
   --lr 3e-5 --std 0.03 --gamma 0.999 --n-steps 64` (league zoo now also holds v5 and the v6 recipe).

## Exam (6 home rotations each, seeds never trained on)
| Test | v6 | v5 |
|---|---|---|
| Same match, seed 31 (+ v9, mlp-v8, SAC, Jerky) | **422,398**, 5/6 wins | 302,480, 0/6 |
| Same match, seed 32 | **270,272**, 4/6 wins | 140,756, 0/6 |
| Swap test, newest entries (seed 33) | 367,732, 6/6, worst +186k | 367,767, 6/6, worst +173k |
| Swap test, hunter bots (seed 33) | **47,822**, rank 3.2 | -265,915, rank 5.2 |

## Known limits
- `hunter-1` (price 1.7, blocker opening) still beats it in the hunter lineup (307k vs 48k).
- One home per seed still collapses (about -205k), as with v5.
- Only 10,000 RL steps (team test budget); a 300k run is the next step.
