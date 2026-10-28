# Orlin's plan

_Last updated: 2026-10-28. Updated as we go in the Claude session._

## Goal
Build my agent for the team's 4-agent comparison, help compare them, and make sure the team submits one strong checkpoint before the deadline (TBA, announced by the organizer).

## Done
- [x] Venv (Python 3.12) with `.[rl,train,recurrent]` installed.
- [x] Ran an NPC-only match (200 rounds, cherry-picker won).
- [x] Replay viewer `scripts/arena_replay.py` -> `runs/replay.html` (Bulgaria map, audit log, region inspector, key moments).
- [x] Repo pushed to https://github.com/ABC-Maika/Simulation (`origin`); organizer repo kept as `upstream`.
- [x] `.claude/` folder, `agents/` layout and team how-to (`agents/README.md`).

## Next
- [ ] Pick my agent (A: profit PPO / B: weighted PPO / C: explorer / D: RecurrentPPO or SAC) and agree the split with the team.
- [ ] `/new-agent orlin <agent-name> <recipe>`, then train (run in background, 6 rounds x 300k steps is slow).
- [ ] Self-check against NPCs with 2+ different `--homes`.
- [ ] Replay viewer with a trained player (`--player`): untested.
- [ ] When 2+ checkpoints exist: `/compare-agents`, read `agents/COMPARISON.md`.
- [ ] Decide the team's single entry, write `entry.json`, file the GitHub entry issue.

## Open questions
- Who takes which of A to D? (team names are not yet in the repo)
- Deadline: not announced yet.
- Is the viewer's price-index bar range (0.5 to 2.0) right? It is a guess.

## Notes / decisions
- Train and judge only in the balanced world.
- Final tournament has no NPCs and rotates seats: do not trust a result from one seating.
- `*.zip` is gitignored except under `agents/`, so checkpoints travel through git there.
