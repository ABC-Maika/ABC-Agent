"""RecurrentPPO trained the sac-v2 way, but settling in 2-3 metro / affluent zones.

sac-v2 recipe (agents/Yavkata/sac-v2): scripted demonstrator -> BC + DAgger clone -> warmed critic -> short anchored
RL in the Orcho/Krischo field. Here the learner stays a stock sb3-contrib RecurrentPPO (MlpLstmPolicy):
  1. demonstrator = metro-then-affluent expansion capped at ``--cap`` stores (home + 1-2), service 0.15, price 1.9,
     research 0, wage 0.45 + 4 * (1 - happiness)  (the sac-v2 lever recipe, small footprint);
  2. BC on whole episodes through the actor LSTM, then DAgger rounds (the clone acts, the demonstrator labels);
  3. value head fitted on the demonstrator's discounted returns (so PPO's first updates do not wreck the clone);
  4. PPO in the field (real Orcho p 0.85, real Krischo p 0.9, NPCs) with small lr / tight clip; best held-out snapshot kept.
Zip takes raw observations (obs scale folded into both LSTM input weights).

    python agents/Yavkata/recurrent-ppo/train_rppo_zones.py --steps 50000 --cap 3
"""
from __future__ import annotations

import argparse, copy, json, sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "sac")); sys.path.insert(0, str(HERE.parent / "sac-v2"))
import train_recurrent_ppo as base  # noqa: E402
import train_sac_bc as bc  # noqa: E402
import train_sac_field as field  # noqa: E402
from script_search import scripted  # noqa: E402

REWARD_SCALE = 1e-3
GAMMA = base.REWARD_GAMMA


def make_env(seed, ticks, obs_scale):
    return Monitor(bc.OpeningEnv(seed, episode_ticks=ticks, bound_logits=False, reward_scale=REWARD_SCALE, obs_scale=obs_scale))


def zone_params(env, cap, extra):
    """Demonstrator params; order = metro (by demand) then affluent (by demand)."""
    p = {"order": "m_a_demand", "cap": cap, "service": 0.15, "price": 1.9, "research": 0.0}
    p.update(extra)
    return p


def patch_order():
    import script_search as ss
    orig = ss.make_order

    def make_order(cfg, kind):
        if kind == "m_a_demand":
            regs = cfg.demand.regions
            by = sorted(range(1, len(regs)), key=lambda r: (-regs[r].base_regional_demand, r))
            return [r for r in by if (r - 1) % 4 == 0] + [r for r in by if (r - 1) % 4 == 1]
        return orig(cfg, kind)
    ss.make_order = make_order


def rollout(env, obs_scale, params, model=None, beta=1.0, rng=None):
    """One episode. Acts with the demonstrator (prob beta per tick, or always if no model) else the clone.
    Returns scaled obs, demonstrator action labels, rewards."""
    obs, _ = env.reset()
    demo = scripted(env._config, params)
    O, A, R = [], [], []
    state, start = None, np.array([True])
    done = False
    while not done:
        raw = env._obs[env._agents[0]]
        lab = np.asarray(demo(raw, episode_start=env._tick == 0), dtype=np.float32).ravel()
        O.append(obs.copy()); A.append(lab)
        if model is not None:  # advance the clone's LSTM state on every visited state
            a_m, state = model.predict(obs[None], state=state, episode_start=start, deterministic=True)
            a_m = a_m[0]
        start = np.array([False])
        act = lab if (model is None or rng.random() < beta) else a_m
        obs, r, _, trunc, _ = env.step(act)
        R.append(r); done = trunc
    return np.array(O, np.float32), np.array(A, np.float32), np.array(R, np.float32)


def seq_forward(policy, obs, critic=False):
    lstm = policy.lstm_critic if critic else policy.lstm_actor
    out, _ = lstm(obs.unsqueeze(1))  # (T,1,F) -> (T,1,H)
    out = out.squeeze(1)
    return policy.mlp_extractor.forward_critic(out) if critic else policy.mlp_extractor.forward_actor(out)


def bc_fit(model, episodes, epochs, lr=5e-4):
    pol = model.policy
    params = [p for n, p in pol.named_parameters() if n.startswith(("lstm_actor", "mlp_extractor.policy_net", "action_net"))]
    opt = torch.optim.Adam(params, lr=lr)
    low = torch.as_tensor(model.action_space.low); high = torch.as_tensor(model.action_space.high)
    for ep in range(epochs):
        tot = 0.0
        for i in np.random.permutation(len(episodes)):
            O, A, _ = episodes[i]
            mean = pol.action_net(seq_forward(pol, torch.as_tensor(O)))
            # continuous levers are clipped to bounds at play time, so compare clipped means
            loss = torch.nn.functional.mse_loss(torch.clamp(mean, low, high), torch.as_tensor(A))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
            tot += float(loss)
        print(f"  bc epoch {ep}: mse {tot / len(episodes):.5f}", flush=True)


def value_fit(model, episodes, epochs, lr=1e-3):
    pol = model.policy
    params = [p for n, p in pol.named_parameters() if n.startswith(("lstm_critic", "mlp_extractor.value_net", "value_net"))]
    opt = torch.optim.Adam(params, lr=lr)
    data = []
    for O, _, R in episodes:
        G = np.zeros_like(R); run = 0.0
        for t in range(len(R) - 1, -1, -1):
            run = R[t] + GAMMA * run; G[t] = run
        data.append((torch.as_tensor(O), torch.as_tensor(G)))
    for ep in range(epochs):
        tot = 0.0
        for i in np.random.permutation(len(data)):
            O, G = data[i]
            v = pol.value_net(seq_forward(pol, O, critic=True)).squeeze(-1)
            loss = torch.nn.functional.mse_loss(v, G)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
            tot += float(loss)
        print(f"  value epoch {ep}: mse {tot / len(data):.4f}", flush=True)


class Snapshot(base.BestSnapshot):
    """Held-out score in the Orcho/Krischo field (profit = sum of unscaled rewards incl. shaping penalties)."""

    def score(self):
        totals = []
        for s in self._seeds:
            env = bc.OpeningEnv(s, episode_ticks=self._ticks, bound_logits=False, obs_scale=self._obs_scale)
            obs, _ = env.reset(); state, start, tot, done = None, np.array([True]), 0.0, False
            while not done:
                a, state = self.model.predict(obs[None], state=state, episode_start=start, deterministic=True)
                obs, r, _, done, _ = env.step(a[0]); start = np.array([False]); tot += r
            totals.append(tot / 1.0)
        mean = float(np.mean(totals)); better = mean > self.best_score
        print(f"[snapshot] step {self.num_timesteps}: held-out {mean:,.0f}{'  <- best' if better else ''}", flush=True)
        if better:
            self.best_score = mean; self.best_state = copy.deepcopy(self.model.policy.state_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=3, help="max stores (home + 1-2 zones)")
    ap.add_argument("--demo-json", default="{}")
    ap.add_argument("--demo-episodes", type=int, default=6)
    ap.add_argument("--dagger-rounds", type=int, default=2)
    ap.add_argument("--dagger-episodes", type=int, default=4)
    ap.add_argument("--bc-epochs", type=int, default=15)
    ap.add_argument("--value-epochs", type=int, default=15)
    ap.add_argument("--steps", type=int, default=50_000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=512)
    ap.add_argument("--snapshot-every", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-subproc", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "checkpoint_zones.zip")
    a = ap.parse_args()

    patch_order()
    field.P_REAL = 0.85; field.P_KRISCHO = 0.9
    bc.OpeningEnv.debt_scale = 1.0
    set_random_seed(a.seed)
    layout = base.ObsLayout.build()
    obs_scale = base.estimate_obs_scale(layout, a.seed, 4.0)
    params = zone_params(None, a.cap, json.loads(a.demo_json))
    print("demonstrator", params, flush=True)

    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(a.seed).spawn(a.n_envs)]
    fns = [partial(make_env, s, 1000, obs_scale) for s in seeds]
    venv = SubprocVecEnv(fns) if not a.no_subproc else DummyVecEnv(fns)
    venv.action_space.seed(a.seed)
    model = RecurrentPPO(
        "MlpLstmPolicy", venv, learning_rate=a.lr, n_steps=a.n_steps, batch_size=256, n_epochs=4, gamma=GAMMA,
        gae_lambda=0.95, clip_range=0.1, ent_coef=a.ent_coef, max_grad_norm=0.5, verbose=1,
        policy_kwargs={"lstm_hidden_size": 128, "n_lstm_layers": 1, "enable_critic_lstm": True, "log_std_init": -2.5},
    )
    base.init_action_head(model)

    # 1-2. demonstrations + BC + DAgger
    denv = bc.OpeningEnv(a.seed + 7000, bound_logits=False, reward_scale=REWARD_SCALE, obs_scale=obs_scale)
    eps = [rollout(denv, obs_scale, params) for _ in range(a.demo_episodes)]
    print(f"demo episodes: {len(eps)}, mean return(scaled) {np.mean([e[2].sum() for e in eps]):.1f}", flush=True)
    bc_fit(model, eps, a.bc_epochs)
    rng = np.random.default_rng(a.seed)
    for k in range(a.dagger_rounds):
        new = [rollout(denv, obs_scale, params, model=model, beta=0.3, rng=rng) for _ in range(a.dagger_episodes)]
        eps += new
        print(f"dagger round {k}: clone-mix return(scaled) {np.mean([e[2].sum() for e in new]):.1f}", flush=True)
        bc_fit(model, eps, a.bc_epochs)
    # 3. critic warm start on the demonstrator's returns
    value_fit(model, eps, a.value_epochs)

    # 4. PPO, keeping the best held-out snapshot (the warm start itself is the first candidate)
    snap = Snapshot(obs_scale, seeds=[5000 + i for i in range(3)], ticks=1000, every=a.snapshot_every)
    snap.init_callback(model); snap.score()
    model.learn(total_timesteps=a.steps, callback=snap)
    venv.close()
    model.policy.load_state_dict(snap.best_state)
    print(f"kept best snapshot (held-out {snap.best_score:,.0f})")

    probe = base._probe_observations(obs_scale, a.seed)
    before = base._deterministic_actions(model, probe / obs_scale)
    base.fold_obs_scale(model, obs_scale)
    drift = float(np.max(np.abs(before - base._deterministic_actions(model, probe)) / np.maximum(1.0, np.abs(before))))
    if drift > 1e-4:
        raise RuntimeError(f"fold drift {drift:.2e}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(a.out); print("saved", a.out, f"(fold drift {drift:.1e})")


if __name__ == "__main__":
    main()
