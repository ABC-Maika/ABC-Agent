---
description: Scaffold a new team agent folder and print the training command
argument-hint: <your-name> <agent-name> [ppo-profit|ppo-weighted|ppo-explorer|recurrent|sac]
---
Create a new agent for the team comparison. Arguments: $ARGUMENTS

1. Read `agents/README.md` and `agents/_template/NOTES.md`.
2. Create `agents/<your-name>/<agent-name>/` by copying `agents/_template/NOTES.md` into it. Do not overwrite an existing folder.
3. Fill the NOTES.md header (owner, algorithm, hypothesis) from the arguments; ask the user only if the algorithm is unclear.
4. Print the exact training command for the chosen recipe from `agents/README.md`, with the label `<your-name>_<agent-name>`.
5. Do NOT start training unless the user says so (it takes a long time).
