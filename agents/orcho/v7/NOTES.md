# orcho-v7 (main Orcho agent)

v5's plan made cheaper and more aggressive, then RL fine-tuned 10k steps (A2C).

- Plan (`hunter_bots.CONTENDERS["v7"]`): v5's expansion order (mirror, 12 stores), service 0.2 (fill rate 0.6 = the
  share cap, so no lost sales and cheaper goods), research 0.1, and a price cut where our regions are crowded
  (price 2.0 - 0.1 x (rivals per own region - 1), floor 1.7).
- Found by `contender_search.py --v6 8`: 463k per game vs v5's 298k over three lineups (site lineup, and two full of
  v5-style rivals).
- Cloned (`clone_recipe.py`, DAgger, 100% expansion agreement), then `league_finetune.py` 10k steps against the
  teammates' agents (Yavkata sac-v2/sac, Misho jerky/jack, Krischo v10-rl/v9, orcho v5).

## Organizer site (https://retail-sim-web-lpc53f73za-ew.a.run.app/battle, 1000 rounds, all 6 home rotations)

| Test | Lineup (besides our seat) | v5 in our seat | **v7 in our seat** |
|---|---|---|---|
| Site lineup, seed 31 | Yavkata sac-v2, jerky, Krischo v10-rl, jack, v6.2 | 1st, 4.00M | 1st, 3.73M |
| v5's weak spot, seed 32 | v5, Yavkata sac-v2, Krischo v10-rl, v6.2, jerky | 3rd, 0.16M | **1st, 2.62M** |
| **Both** | | 4.16M | **6.35M** |

v7 beats Yavkata and jerky in every match of both tests. Slightly below v5 when it is the only v5-style agent
(-0.27M), far stronger when other v5-style agents crowd the map (+2.46M).

## Limits
Two site battles (12 matches); Krischo v11 not tested (not pushed yet). Losing matches still end near -120k to -200k.
