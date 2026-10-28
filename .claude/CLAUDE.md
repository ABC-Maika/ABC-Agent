# Retail Simulator Arena — team conventions

Hackathon repo: train RL agents for the 25-region arena, compare them, submit **one** checkpoint per team.
Rules and API: see `README.md`. Everything is trained and judged in the **balanced** world (`--world balanced`, `--balanced`).

## Environment
- Python 3.11 or 3.12 only (3.13+ is unsupported). `python -m venv .venv`, then `pip install -e ".[rl,train,recurrent]"`.
- `runs/`, `.venv/` are gitignored. Checkpoints only travel through `agents/` (see below).

## Our plan: 4 agents, compared later
Each teammate builds ONE agent in `agents/<your-name>/<agent-name>/`. Full how-to: `agents/README.md`.
Slash commands: `/new-agent` (scaffold + train recipe), `/compare-agents` (run the comparison).

## Rules for changes
- Do not edit the upstream arena code in `src/` to make an agent score better; the organizer scores on the original.
- Never commit secrets. Only the final-round checkpoint goes in `agents/` (not every round).
- Personal plans live in `.claude/plans/<name>/PLAN.md`; update yours when a step finishes.
- Commit and push to `origin` (ABC-Maika/Simulation). `upstream` is the organizer's repo, pull-only.
