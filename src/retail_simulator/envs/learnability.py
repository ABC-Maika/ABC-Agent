"""SB3 PPO vs a random baseline — the single-agent learnability gate.

PASS requires BOTH: ``ppo_mean >= random_mean + 2 * random_std`` AND a non-degenerate learning curve (final window mean > initial window mean). A JSON report is written for provenance; a learning-curve PNG is saved only if ``matplotlib`` is importable (best-effort, never a hard dependency).

Layer note: ``harness -> envs -> core``; importing ``gymnasium`` / ``stable-baselines3`` here is allowed (forbidden only under ``core/``). Heavy RL imports are lazy, so ``import retail_simulator.envs`` stays cheap.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from retail_simulator.logging_setup import human_steps

if TYPE_CHECKING:  # type-only imports; never pulled in at runtime import time
    import gymnasium as gym

# The store count of an agent present in region 0 only (the reset state). An
# episode "opens region 1" the tick its open-store count first exceeds this — the seam
# no-ops an unaffordable/illegal open, so a rising ``stores`` means an open was ACCEPTED.
_REGION_0_ONLY_STORES: int = 1

# Default scoring/eval horizon (~2 simulated years) — mirrors the core hint.
DEFAULT_MAX_EPISODE_STEPS: int = 104
# Default number of evaluation episodes for both PPO and the random baseline.
DEFAULT_EVAL_EPISODES: int = 20
# "Quick" smoke budget for CI (~200k steps); the full research gate is ~2M.
QUICK_TOTAL_STEPS: int = 200_000
FULL_TOTAL_STEPS: int = 2_000_000
# The "beats random" margin: PPO mean must clear random mean by this many sigma.
GATE_SIGMA_MULTIPLE: float = 2.0

ENV_ID: str = "RetailSim-v0"

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpansionUptake:
    """How often a trained agent opens region 1, and when (median open tick)."""

    fraction_opening: float
    median_open_tick: float | None
    episodes: int


@dataclass(frozen=True)
class GateReport:
    """The gate's measured quantities + PASS/FAIL components."""

    ppo_mean_reward: float
    random_mean_reward: float
    random_std_reward: float
    margin: float
    threshold: float
    beats_random: bool
    curve_trends_up: bool
    total_steps: int
    eval_episodes: int
    learning_curve: tuple[tuple[int, float], ...]
    expansion_uptake: ExpansionUptake | None = None

    @property
    def passed(self) -> bool:
        """PASS rule: beats random AND the curve trends up."""
        return self.beats_random and self.curve_trends_up

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable view (tuples become lists; uptake None-safe)."""
        data = asdict(self)
        data["learning_curve"] = [list(point) for point in self.learning_curve]
        data["passed"] = self.passed
        # asdict already turned the nested ExpansionUptake into a dict (or left it None).
        return data


def evaluate_random_baseline(
    env: gym.Env[Any, Any],
    *,
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> list[float]:
    """Roll out the random policy (``action_space.sample()``) for N episodes.

    Returns per-episode total (undiscounted) returns; deterministic given ``seed`` (per-episode reset seed + seeded action-space RNG).
    """
    returns: list[float] = []
    env.action_space.seed(seed)
    for episode in range(episodes):
        _obs, _info = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        while not done:
            action = env.action_space.sample()
            _obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            done = bool(terminated) or bool(truncated)
        returns.append(total)
    return returns


def evaluate_policy_returns(
    model: Any,
    env: gym.Env[Any, Any],
    *,
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> list[float]:
    """Roll out a trained SB3 model deterministically for N episodes.

    Per-episode total returns via ``model.predict(.., deterministic=True)`` (no exploration noise); same per-episode seeding as the baseline.
    """
    returns: list[float] = []
    for episode in range(episodes):
        obs, _info = env.reset(seed=seed + episode)
        done = False
        total = 0.0
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            done = bool(terminated) or bool(truncated)
        returns.append(total)
    return returns


def measure_expansion_uptake(
    model: Any,
    env: gym.Env[Any, Any],
    *,
    episodes: int = DEFAULT_EVAL_EPISODES,
    seed: int = 0,
) -> ExpansionUptake:
    """Fraction of episodes in which the agent opened region 1, and the median open tick."""
    open_ticks: list[int] = []
    for episode in range(episodes):
        obs, _info = env.reset(seed=seed + episode)
        done = False
        opened_at: int | None = None
        step_index = 0
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)
            step_index += 1
            if opened_at is None and int(info.get("stores", 0)) > _REGION_0_ONLY_STORES:
                # First tick the open-store count rose => region 1 was opened; prefer the
                # env's own episode_step, else fall back to the loop counter.
                opened_at = int(info.get("episode_step", step_index))
            done = bool(terminated) or bool(truncated)
        if opened_at is not None:
            open_ticks.append(opened_at)

    fraction_opening = len(open_ticks) / episodes if episodes else 0.0
    median_open_tick = float(np.median(open_ticks)) if open_ticks else None
    return ExpansionUptake(
        fraction_opening=fraction_opening,
        median_open_tick=median_open_tick,
        episodes=episodes,
    )


def _curve_trends_up(curve: list[tuple[int, float]]) -> bool:
    """Is the learning curve non-degenerate (final window > initial window)?

    With < 4 samples the trend cannot be judged, so it is treated as non-degenerate (the >= 2 sigma test still governs). Otherwise compares the mean of the last quarter of logged points against the first quarter.
    """
    if len(curve) < 4:
        return True
    rewards = [reward for _step, reward in curve]
    window = max(1, len(rewards) // 4)
    initial = float(np.mean(rewards[:window]))
    final = float(np.mean(rewards[-window:]))
    return final > initial


def run_gate(
    *,
    total_steps: int = QUICK_TOTAL_STEPS,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    seed: int = 42,
    n_envs: int = 1,
    report_path: Path | None = None,
    save_plot: bool = True,
    progress: bool = False,
) -> GateReport:
    """Train PPO, compare to random, and return the PASS/FAIL :class:`GateReport`.

    Trains ``PPO("MlpPolicy", env)`` for ``total_steps`` (no custom wrappers), then evaluates PPO and the random baseline over ``eval_episodes`` at a fixed ``seed``. ``n_envs > 1`` collects rollouts in parallel (``SubprocVecEnv``, fork start method, sub-envs seeded ``seed + i``); evaluation always runs on a single env. With ``report_path`` given, the report JSON (+ best-effort ``learning_curve.png``) is written. Heavy RL imports happen here so the module stays import-light.
    """
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    import retail_simulator  # noqa: F401  (registers RetailSim-v0)

    train_env: Any
    if n_envs > 1:
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import SubprocVecEnv

        # make_vec_env wraps each sub-env in Monitor, so the curve callback's
        # ep_info_buffer is populated exactly as in the single-env path.
        train_env = make_vec_env(
            ENV_ID,
            n_envs=n_envs,
            seed=seed,
            env_kwargs={"max_episode_steps": max_episode_steps},
            vec_env_cls=SubprocVecEnv,
            # fork avoids re-importing __main__ (forkserver/spawn break for a
            # non-importable caller); safe here — the env has no background threads.
            vec_env_kwargs={"start_method": "fork"},
        )
    else:
        train_env = Monitor(gym.make(ENV_ID, max_episode_steps=max_episode_steps))
    curve: list[tuple[int, float]] = []
    callback = _make_curve_callback(
        curve, log_every=max(total_steps // 20, 1), label="learnability gate", logger=_log
    )

    model = PPO("MlpPolicy", train_env, seed=seed, verbose=1 if progress else 0)
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=False)

    eval_env = gym.make(ENV_ID, max_episode_steps=max_episode_steps)
    ppo_returns = evaluate_policy_returns(model, eval_env, episodes=eval_episodes, seed=seed)
    expansion_uptake = measure_expansion_uptake(model, eval_env, episodes=eval_episodes, seed=seed)
    random_returns = evaluate_random_baseline(eval_env, episodes=eval_episodes, seed=seed)
    eval_env.close()
    train_env.close()

    report = build_report(
        ppo_returns=ppo_returns,
        random_returns=random_returns,
        learning_curve=list(curve),
        total_steps=total_steps,
        eval_episodes=eval_episodes,
        expansion_uptake=expansion_uptake,
    )

    if report_path is not None:
        _write_report(report, report_path, save_plot=save_plot)

    return report


def build_report(
    *,
    ppo_returns: list[float],
    random_returns: list[float],
    learning_curve: list[tuple[int, float]],
    total_steps: int,
    eval_episodes: int,
    expansion_uptake: ExpansionUptake | None = None,
) -> GateReport:
    """Assemble the :class:`GateReport` from per-episode returns + the learning curve."""
    ppo_mean = float(np.mean(ppo_returns))
    random_mean = float(np.mean(random_returns))
    # Population std of the baseline returns: the ">= 2 sigma" rule wants the baseline's own spread.
    random_std = float(np.std(random_returns))
    margin = ppo_mean - random_mean
    threshold = GATE_SIGMA_MULTIPLE * random_std
    beats_random = margin >= threshold
    trends_up = _curve_trends_up(learning_curve)
    return GateReport(
        ppo_mean_reward=ppo_mean,
        random_mean_reward=random_mean,
        random_std_reward=random_std,
        margin=margin,
        threshold=threshold,
        beats_random=beats_random,
        curve_trends_up=trends_up,
        total_steps=total_steps,
        eval_episodes=eval_episodes,
        learning_curve=tuple((int(s), float(r)) for s, r in learning_curve),
        expansion_uptake=expansion_uptake,
    )


def _make_curve_callback(
    sink: list[tuple[int, float]],
    *,
    log_every: int | None = None,
    label: str | None = None,
    logger: logging.Logger | None = None,
) -> Any:
    """Build an SB3 callback that records rolling mean episode reward.

    ``BaseCallback`` is imported lazily (only available with the ``[train]`` extra). At each rollout end the callback appends ``(timestep, mean_reward)`` to ``sink`` from the ``Monitor``-tracked episode buffer. With ``log_every`` set, it also emits a progress line every ``log_every`` timesteps so a long ``model.learn`` never runs silently in the terminal.
    """
    from stable_baselines3.common import base_class
    from stable_baselines3.common.callbacks import BaseCallback

    class _CurveCallback(BaseCallback):
        def __init__(
            self,
            curve_sink: list[tuple[int, float]],
            *,
            log_every: int | None,
            label: str | None,
            logger: logging.Logger | None = None,
        ) -> None:
            super().__init__()
            self._sink = curve_sink
            self._log_every = log_every
            self._label = label
            self._logger = logger if logger is not None else _log
            self._next_log_at = log_every
            # The model is bound later by SB3 in init_callback(); capture the start
            # there so the first progress line has a real rate.
            self._start_time = time.monotonic()
            self._start_steps = 0
            self._last_log_time: float | None = None
            self._last_log_steps: int = 0

        def init_callback(self, model: base_class.BaseAlgorithm) -> None:
            super().init_callback(model)
            self._start_time = time.monotonic()
            self._start_steps = int(model.num_timesteps)

        def _on_rollout_end(self) -> None:
            buffer = getattr(self.model, "ep_info_buffer", None)
            rewards = [float(ep["r"]) for ep in buffer or [] if "r" in ep]
            if rewards:
                self._sink.append((int(self.num_timesteps), float(np.mean(rewards))))
            if not (self._log_every and self._next_log_at is not None):
                return
            if int(self.num_timesteps) < self._next_log_at:
                return
            total = int(self.locals.get("total_timesteps", 0) or self.model.num_timesteps or 0)
            # A rollout runs to the episode's end, so the counter can overshoot the
            # target; clamp the display so progress never reads past 100%.
            done = int(self.num_timesteps) if total <= 0 else min(int(self.num_timesteps), total)
            prefix = f"{self._label}: " if self._label else ""
            # Rate since the previous log line, or since training start for the first one.
            now = time.monotonic()
            base_time = self._last_log_time if self._last_log_time is not None else self._start_time
            base_steps = (
                self._last_log_steps if self._last_log_time is not None else self._start_steps
            )
            elapsed = now - base_time
            sps = (int(self.num_timesteps) - base_steps) / elapsed if elapsed > 0 else 0.0
            self._last_log_time = now
            self._last_log_steps = int(self.num_timesteps)
            last = self._sink[-1][1] if self._sink else None
            if last is None or math.isnan(last):
                self._logger.info(
                    "%straining %s/%s steps · %.1f sps",
                    prefix,
                    human_steps(done),
                    human_steps(total),
                    sps,
                )
            else:
                self._logger.info(
                    "%straining %s/%s steps · ep_reward %+.1f · %.1f sps",
                    prefix,
                    human_steps(done),
                    human_steps(total),
                    last,
                    sps,
                )
            self._next_log_at += self._log_every

        def _on_step(self) -> bool:
            return True

    return _CurveCallback(sink, log_every=log_every, label=label, logger=logger)


def _write_report(report: GateReport, report_path: Path, *, save_plot: bool) -> None:
    """Write the gate report JSON and (best-effort) a learning-curve PNG."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    if save_plot and report.learning_curve:
        _maybe_save_curve_plot(report, report_path.parent / "learning_curve.png")


def _maybe_save_curve_plot(report: GateReport, plot_path: Path) -> bool:
    """Save a learning-curve PNG if matplotlib is importable; else skip silently.

    Returns ``True`` if a plot was written (matplotlib is an optional ``[viz]`` extra, so this degrades to a no-op when absent).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend; no display required
        import matplotlib.pyplot as plt
    except ImportError:
        _log.debug("matplotlib not installed; skipping learning-curve plot")
        return False

    steps = [s for s, _ in report.learning_curve]
    rewards = [r for _, r in report.learning_curve]
    fig, ax = plt.subplots()
    ax.plot(steps, rewards, label="PPO mean episode reward")
    ax.axhline(
        report.random_mean_reward,
        color="grey",
        linestyle="--",
        label="random baseline mean",
    )
    ax.set_xlabel("timesteps")
    ax.set_ylabel("mean episode reward")
    ax.set_title("Learnability: PPO vs random")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return True
