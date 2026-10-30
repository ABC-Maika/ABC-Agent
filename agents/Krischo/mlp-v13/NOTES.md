# Krischo mlp-v13

Orcho's v7 checkpoint (agents/orcho/v7) fine-tuned 10k steps with A2C (lr 1e-5, n_steps 64), weights pulled
halfway back to v7 afterwards. The learner seat faced random lineups drawn from the site7 and old7 setups
(Orcho seat = v6.2). Script: runs/strategy/expert_v12/clone/ft_anchor_v12b.py (`--algo a2c`).

## Gate (seeds 4 and 5, 6 homes each = 12 matches, Orcho seat v6.2, mean profit per game)

| Focal | site7x | old7x | old7x worst home |
|---|---|---|---|
| mlp-v11-rl | -28k | 366k | -286k |
| v7 original | 231k | 979k | 85k |
| **v13 (this)** | 225k | 992k | 368k |

Equal to v7 within noise; far above mlp-v11-rl on both lineups. Weak homes on site7x are the same as v7's (5 and 9).

## Limits
Two seeds, near-deterministic opponents. This is v7's strength, not a new strategy: our own scripted expert
(smooth KX12) scored 378k on site7x / 885k on old7x but could not be cloned into a checkpoint that held up.
