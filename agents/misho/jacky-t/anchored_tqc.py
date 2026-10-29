"""TQC whose actor update is anchored to demonstrations (TD3+BC style), plus the demo loader.

Plain RL fine-tuning of a behaviour-cloned actor drifted away from the clone (the critics start out
wrong, so the actor chases a bad Q): Jacky-T v1.0 and v1.2 both got worse. The anchored update keeps
the reinforcement-learning objective but adds a pull towards the teacher's actions on a fixed set of
demonstration states:

    actor_loss = alpha / mean|Q| * (entropy_term - Q) + MSE(tanh(actor_mean(demo_obs)), demo_action)

Only ``train`` differs from ``sb3_contrib.TQC``. The demonstrations and ``alpha`` are excluded from
``save``, and the algorithm class is not stored in a zip (only the standard ``TQCPolicy`` is), so
a saved model loads as a plain TQC with nothing custom inside.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch as th
from sb3_contrib import TQC
from sb3_contrib.common.utils import quantile_huber_loss
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F

from retail_simulator.core.encoding import flatten_action_space_layout


def load_demos(
    paths: Sequence[Path], max_rank: int = 0, wage_floor: float = 0.0, expand_boost: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Teacher recordings from ``mimic.py collect``, with the teachers' flawed episodes removed.

    ``max_rank``: keep only episodes where the teacher finished that high or better, with a profit.
    ``wage_floor``: staff pay below ~0.5 leaves happiness near 0 and triggers the write-off penalty
    (KrischoV8: wage 0.05, happiness 0.01; Orcho and Krischo at ~0.5 hold happiness at 1.0), so the
    wage target is lifted to this floor.

    ``expand_boost``: rows where the teacher expands (not "hold") are repeated this many times. They are
    only ~4% of ticks, so a plain fit averages the expansion choice away to "hold" and the clone stays
    at 3 stores while the teachers grow to 12.
    """
    loaded = [np.load(p) for p in paths]
    obs = np.concatenate([d["obs"] for d in loaded])
    act = np.concatenate([d["act"] for d in loaded])
    if max_rank:
        keep = np.concatenate([(d["ep_rank"] <= max_rank) & (d["ep_profit"] > 0) for d in loaded])
        print(f"keeping {int(keep.sum())}/{len(keep)} samples from episodes ranked <= {max_rank}")
        obs, act = obs[keep], act[keep]
    if wage_floor:
        for slot in flatten_action_space_layout().levers:
            if slot.spec.name == "wage_spend":
                # Not in rounds 1-5: the teachers pay nothing there so cash stays above the 5,000 store
                # gate (the demos are whole 1,000-round episodes, so the row index gives the round).
                late = (np.arange(len(act)) % 1000) >= 5
                act[late, slot.offset] = np.maximum(act[late, slot.offset], wage_floor)
    if expand_boost > 1:
        for slot in flatten_action_space_layout().levers:
            if slot.spec.name == "expansion":
                block = act[:, slot.offset : slot.offset + slot.width]
                grow = np.flatnonzero(block.argmax(1) != 0)
                reps = np.concatenate([np.arange(len(act))] + [grow] * (expand_boost - 1))
                print(f"expansion rows {len(grow)} x{expand_boost}")
                obs, act = obs[reps], act[reps]
    return obs, act


class AnchoredTQC(TQC):
    demo_obs: th.Tensor | None = None  # scaled observations (the obs scale is folded in later)
    demo_act: th.Tensor | None = None  # teacher actions scaled to [-1, 1]
    anchor_alpha: float = 2.5

    def _excluded_save_params(self) -> list[str]:
        return [*super()._excluded_save_params(), "demo_obs", "demo_act", "anchor_alpha"]

    def set_demos(self, obs: np.ndarray, act: np.ndarray, alpha: float) -> None:
        scaled = np.clip(self.policy.scale_action(act), -1.0, 1.0)
        self.demo_obs = th.as_tensor(obs, dtype=th.float32, device=self.device)
        self.demo_act = th.as_tensor(scaled, dtype=th.float32, device=self.device)
        self.anchor_alpha = alpha

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Identical to ``TQC.train`` except for the actor loss below.
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()  # type: ignore[operator]
            else:
                ent_coef = self.ent_coef_tensor
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_quantiles = self.critic_target(replay_data.next_observations, next_actions)
                n_target = self.critic.quantiles_total - self.top_quantiles_to_drop_per_net * self.critic.n_critics
                next_quantiles, _ = th.sort(next_quantiles.reshape(batch_size, -1))
                next_quantiles = next_quantiles[:, :n_target]
                target_quantiles = next_quantiles - ent_coef * next_log_prob.reshape(-1, 1)
                target_quantiles = replay_data.rewards + (1 - replay_data.dones) * discounts * target_quantiles
                target_quantiles.unsqueeze_(dim=1)

            current_quantiles = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = quantile_huber_loss(current_quantiles, target_quantiles, sum_over_quantiles=False)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            qf_pi = self.critic(replay_data.observations, actions_pi).mean(dim=2).mean(dim=1, keepdim=True)
            rl_loss = (ent_coef * log_prob - qf_pi).mean()
            if self.demo_obs is not None:
                pick = th.randint(0, len(self.demo_obs), (batch_size,), device=self.device)
                mean_actions, _, _ = self.actor.get_action_dist_params(self.demo_obs[pick])
                anchor = F.mse_loss(th.tanh(mean_actions), self.demo_act[pick])
                scale = self.anchor_alpha / qf_pi.abs().mean().detach().clamp_min(1e-6)
                actor_loss = scale * rl_loss + anchor
            else:
                actor_loss = rl_loss
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
