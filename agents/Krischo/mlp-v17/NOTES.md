# Krischo mlp-v17 (experimental, for teammates to test)

Orcho's v7 checkpoint (agents/orcho/v7) trained 200k steps with plain A2C RL: lr 1e-4, n_steps 64, no anchor
(weights not pulled back), learner seat vs random lineups from site7x/old7x (Orcho seat = v6.2).
Script: runs/strategy/expert_v12/clone/ft_anchor_v12b.py `--algo a2c --lr 1e-4 --alpha 1.0 --steps 200000`.

## Our gate (seeds 4 and 5, 6 homes = 12 matches per cell, mean profit per game)

| Agent | site7x | old7x |
|---|---|---|
| v7 original | 231k | 979k |
| mlp-v13 (v7 + 10k anchored A2C) | 225k | 992k |
| this run at 50k steps | -59k | 430k |
| this run at 100k steps | 126k | 438k |
| this run at 150k steps | -59k | 367k |
| **this checkpoint (200k)** | **not gated when committed** | |

So far the curve does not climb: it is below v7 and v13 on both lineups. Committed so teammates can test it on
their own lineups; do NOT treat it as better than mlp-v13 until a gate says so.

Earlier runs with an anchor and a smaller lr (1e-5 / 5e-6, 50k to 200k steps) stayed near v7 (about 195k-265k site7x,
905k-1037k old7x), with no clear gain either.
