"""sac-v2 trainer: scripted demonstrator (best of script_search*.py) -> BC + DAgger into a SAC actor -> warmed critics
-> short, anchored SAC phase in the Orcho/Krischo field. Reuses agents/Yavkata/sac/train_sac_bc.py machinery.

Why DAgger: in the full6 field the scripted orcho-like demonstrator earns ~471k, but the BC-cloned SAC earned ~290k
(clone drift: it visits states the demonstrator never showed). Relabelling the clone's own states with the demonstrator fixes that.
Anchoring: actor lr 3e-5, small fixed entropy, long critic warm-up; the debt/opening penalties of train_sac_bc stay on.

    python agents/Yavkata/sac-v2/train_v2.py --demo-json '{"cap":12}' --sac-steps 20000 --out agents/Yavkata/sac-v2/checkpoint_new.zip
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "sac"))
import train_sac_bc as bc  # noqa: E402
import train_sac_field as field  # noqa: E402
import train_vs_npcs as base  # noqa: E402
from script_search import scripted  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.utils import set_random_seed  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402


def make_envs(n, seed, obs_scale):
    return [bc.OpeningEnv(seed + 5000 + i, bound_logits=True, reward_scale=base.SAC_REWARD_SCALE, obs_scale=obs_scale) for i in range(n)]


def dagger(model, obs_scale, params, steps, seed, n_envs=4, beta=0.3):
    """Roll the clone (with prob beta the demonstrator acts), label every visited state with the demonstrator."""
    policy, space = model.policy, model.action_space
    rng = np.random.default_rng(seed)
    envs = make_envs(n_envs, seed, obs_scale)
    demos = [None] * n_envs; obs = [None] * n_envs
    def start(i):
        o, _ = envs[i].reset(); obs[i] = o; demos[i] = scripted(envs[i]._config, params)
    for i in range(n_envs): start(i)
    X, Y = [], []
    for t in range(steps // n_envs):
        for i, env in enumerate(envs):
            raw = env._obs[env._agents[0]]
            lab = np.clip(demos[i](raw, episode_start=env._tick == 0), space.low, space.high)
            lab_s = np.clip(policy.scale_action(lab), -1, 1)
            X.append(obs[i].copy()); Y.append(lab_s)
            if rng.random() < beta: a_s = lab_s
            else:
                with torch.no_grad():
                    a_s = policy.actor(torch.as_tensor(obs[i][None]), deterministic=True)[0].numpy()
            o, r, _, trunc, _ = env.step(policy.unscale_action(a_s).astype(np.float32))
            obs[i] = o
            if trunc: start(i)
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def bc_fit(model, X, Y, steps, lr=5e-4, bs=256):
    actor = model.policy.actor
    params = [p for n, p in actor.named_parameters() if not n.startswith("log_std")]
    opt = torch.optim.Adam(params, lr=lr)
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    for s in range(steps):
        idx = torch.randint(0, len(Xt), (bs,))
        loss = torch.nn.functional.mse_loss(actor(Xt[idx], deterministic=True), Yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 1000 == 0: print(f"  bc step {s}: mse {float(loss):.5f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-json", default="{}")
    ap.add_argument("--demo-steps", type=int, default=40_000)
    ap.add_argument("--clone-steps", type=int, default=4000)
    ap.add_argument("--dagger-rounds", type=int, default=3)
    ap.add_argument("--dagger-steps", type=int, default=16_000)
    ap.add_argument("--critic-steps", type=int, default=8000)
    ap.add_argument("--sac-steps", type=int, default=20_000)
    ap.add_argument("--actor-lr", type=float, default=3e-5)
    ap.add_argument("--ent-coef", type=float, default=0.003)
    ap.add_argument("--p-real", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "checkpoint_new.zip")
    a = ap.parse_args()
    params = json.loads(a.demo_json)
    field.P_REAL = a.p_real; field.P_KRISCHO = 0.9
    bc.OpeningEnv.debt_scale = 1.0
    base.NpcFieldEnv = bc.OpeningEnv
    bc.orcho_demo = lambda config, cap, order: scripted(config, params)  # fill_buffer_with_demos picks this up
    set_random_seed(a.seed)
    obs_scale, samples = base.estimate_obs_scale(a.seed)
    targs = argparse.Namespace(seed=a.seed, n_envs=4, episode_ticks=1000, bound_logits=True, reward="profit")
    venv = DummyVecEnv(base.make_env_fns(targs, reward_scale=base.SAC_REWARD_SCALE, obs_scale=obs_scale))
    venv.action_space.seed(a.seed)
    model = SAC("MlpPolicy", venv, buffer_size=a.demo_steps + a.sac_steps + 1000, gradient_steps=-1,
                ent_coef=a.ent_coef, learning_starts=0, verbose=1, learning_rate=a.actor_lr)
    bc.fill_buffer_with_demos(model, obs_scale, a.demo_steps, a.seed)
    bc.clone_actor(model, a.clone_steps)
    Xs, Ys = [], []
    for k in range(a.dagger_rounds):
        X, Y = dagger(model, obs_scale, params, a.dagger_steps, a.seed + 10 * k)
        Xs.append(X); Ys.append(Y)
        # also include demo-buffer data for the fit
        bc_fit(model, np.concatenate(Xs), np.concatenate(Ys), a.clone_steps)
        print(f"dagger round {k}: {len(np.concatenate(Xs))} labelled states", flush=True)
    with torch.no_grad():
        model.policy.actor.log_std.bias.fill_(-2.5)
    bc.warm_critics(model, a.critic_steps, a.ent_coef)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bc.save_folded_copy(model, a.out.with_name(a.out.stem + "_warm.zip"), obs_scale, samples)
    if a.sac_steps > 0:
        model.learn(total_timesteps=a.sac_steps, reset_num_timesteps=True)
        bc.fold_and_check(model, obs_scale, samples)
        model.save(a.out); print("saved", a.out)


if __name__ == "__main__":
    main()
