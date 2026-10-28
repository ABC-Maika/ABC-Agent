---
description: Run the 4-agent comparison (rotated homes, several seeds) and summarise who wins where
argument-hint: [rounds, default 1000]
---
Compare every agent in `agents/*/*/checkpoint.zip`. Rounds: $ARGUMENTS (default 1000).

1. List the agents found. Stop and say so if fewer than 2 have a `checkpoint.zip`.
2. For each seed in 0, 1, 2 and each cyclic rotation of the homes (`1,5,9,13`, `5,9,13,1`, `9,13,1,5`, `13,1,5,9`) run
   `python scripts/arena_competition.py --balanced --rounds <rounds> --seed <seed> --player <name>=<zip> ... --homes <rotation> --out runs/compare/<seed>_<rot>.json --csv runs/compare/<seed>_<rot>.csv`
   Seat all agents; do not add NPCs (the final tournament has none). Run matches in the background, they are slow.
3. Aggregate total score per agent (mean, min, max over matches) and per home region.
4. Report where each agent is better or worse: score, final share, stores, and whether the lead holds across rotations.
5. Write the table to `agents/COMPARISON.md`. Do not commit.
