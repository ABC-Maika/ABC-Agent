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

## Later local tests (the site's battle feature was removed; local simulator, same engine)

12 games vs the newest agents (Krischo v11-rl and v13, Jerky lab-opt, Yavkata sac-v2, v6.2), 18 games in the light lineup:

| RL steps | Light lineup (avg / worst) | Newest lineup (avg / place) |
|---|---|---|
| 5k | +25k / -232k | -120k / 3.92 |
| **10k (this checkpoint)** | **+459k / +185k** | **-100k / 3.50** |
| 50k | +136k / -237k | -139k / 3.42 |
| 100k | +28k / -279k | -150k / 4.25 |

More or fewer RL steps both hurt; do not retrain this checkpoint. A copy of its opening (Krischo v13, Jerky) shares its first regions, so
in the newest lineup it loses (Yavkata sac-v2 wins 7 of 12 there).

## Limits
Two site battles (12 matches) plus the local tests above. Losing matches still end near -120k to -250k.
