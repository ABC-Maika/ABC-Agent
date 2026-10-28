# Prompt: Train a RecurrentPPO agent for the Retail Arena

## Objective
Act as a Senior Reinforcement Learning Engineer. Build and train a **RecurrentPPO** (sb3-contrib, `MlpLstmPolicy`) agent for the 25-region Retail Simulator Arena. The LSTM's job is to track how rivals behave over time: the competitor block of the observation is noisy, so memory across ticks is what the agent adds over a plain PPO player.

Read first: `README.md` (rules, API), `agents/README.md` (where agents live), and `scripts/train_vs_npcs.py` (the existing RecurrentPPO/SAC/PPO example trainer — extend it, do not rewrite it from scratch).

## Fixed facts (do not change)
- World: balanced, `arena_balanced_config()`. Never train against overrides.
- Observation: 318-D float vector. Action: 44-D continuous vector. Take both spaces from the env, never hard-code widths.
- Competitor block is PERCEIVED (noised); it is sharpened by the `research` lever, which the agent learns to use. Do NOT add a wrapper that reshapes, reorders or reweights the observation or action — the scorer feeds the raw 318-D vector into the saved zip.
- Reward = per-tick post-penalty profit (`--reward profit`). This is exactly the score. Do not reward competitor-tracking directly.
- Do not edit `src/` (upstream arena code).

## 1. Environment
Build the env the way `scripts/arena_competition.py` does:
```python
RetailParallelEnv(arena_balanced_config(), n_learning_agents=n_seats,
                  home_regions=arena_home_regions(n_seats), seed=...,
                  mask_expansion_seats=tuple(range(n_seats)))
env.render_mode = None   # before the supersuit wrap
```
- Train with expansion masking ON for the learning seat (it is evaluated masked).
- Every episode: reshuffle the six metro homes (1, 5, 9, 13, 17, 21) across seats and draw a fresh world seed, so the policy cannot memorise one seating. Final scoring uses hidden seeds and rotated homes.
- Opponents: the four NPC archetypes (cherry-picker, fortress, fast-follower, sprawler), as `train_vs_npcs.py` already seats them. Optionally mix in league snapshots of the agent's own earlier checkpoints for robustness.
- Truncate at 1,000 ticks (match length); SB3 bootstraps through truncation.
- No `VecNormalize` (statistics are not applied at scoring). Train on raw observations, as the PPO path of `train_vs_npcs.py` does.
- Do NOT bound the logit dims to [-1, 1] (that is SAC-only; it breaks expansion decoding for the PPO family).

## 2. Agent
- `sb3_contrib.RecurrentPPO("MlpLstmPolicy", ...)` — stock policy class only. A custom policy or feature-extractor class will not load in the organizer's ladder.
- Expose through `policy_kwargs`: `lstm_hidden_size`, `n_lstm_layers`, `shared_lstm`, `enable_critic_lstm`, `net_arch`. These are stored in the zip, so they are safe.
- Hyperparameter flags: `n_steps` (default 128), `batch_size`, `n_epochs`, `gamma`, `gae_lambda` (0.98 in the reference recipes), `ent_coef`, `learning_rate` (constant or linear schedule), `n_envs`, `total_timesteps`, `seed`.
- Cap nothing special for memory (on-policy, no replay buffer), but keep `n_envs` modest on a laptop.

## 3. Trainer
Extend `scripts/train_vs_npcs.py` (or add `agents/<your-name>/<agent-name>/train.py` that imports from it):
- `DummyVecEnv`/`SubprocVecEnv` over the single-seat Gymnasium wrapper.
- TensorBoard logging to `runs/`.
- `CheckpointCallback` every N steps, plus an `EvalCallback` that plays a full 1,000-tick match against the NPC field with deterministic prediction and the LSTM state carried across the whole match (`lstm_states`, `episode_start`).
- Save the final model with `model.save(...)` as a `.zip`.
- `runs/` is gitignored; only the final checkpoint goes under `agents/<your-name>/<agent-name>/`.

## 4. Evaluation (no new evaluate.py)
Use the provided self-check, and rotate homes before trusting any rank below first:
```bash
python scripts/arena_competition.py --balanced --rounds 1000 \
    --player MyAgent=<path-to-final>.zip \
    --npc cherry-picker --npc fortress --npc fast-follower --npc sprawler \
    --out standings.json --csv decisions.csv
# repeat with --homes 5,9,13,17,21 and other values
```
Also check the log for the checkpoint-detection warning: if the ladder can't read the zip it silently loads it as `ppo-mlp` (stateless), which would hide a broken recurrent entry.

## 5. Deliverables
- Trainer code with complete, commented, PEP 8 code — no placeholders or TODOs.
- The exact CLI commands to train and to self-check.
- `entry.json` with `policy_kind: "recurrent-ppo-lstm"`, `team`, `agent_name`, `contact`, `training_recipe`, `sb3_version` (read from `unzip -p model.zip system_info.txt`).

## 6. Rules to respect
- Python 3.11 or 3.12; install `pip install -e ".[rl,train,recurrent]"`.
- Predict reads only the observation (no network calls, no file reads); deterministic for the same observation and seed.
- One checkpoint per team; do not submit `round_0`/untrained snapshots; byte-identical checkpoints are disqualified.
- Commit and push to `origin` (ABC-Maika/Simulation); `upstream` is pull-only. Never commit secrets.
