"""Hot-seat game driver: turn loop, end-game prompts, reveal, and debrief writer."""

from __future__ import annotations

import json as _json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from retail_simulator.core.encoding import decode_action
from retail_simulator.envs.hotseat.board import _BOARD_WIDTH, prompt_human_action
from retail_simulator.envs.hotseat.model import (
    _COUNTER_RATIONALE,
    BOARD_RIVAL_FIELDS,
    COUNTERED_BY,
    PLAYER_LEVERS,
    PLAYSTYLES,
    SELF_LAST_PROFIT_FIELD,
    SELF_MARKET_SHARE_FIELD,
    GameResult,
    RunningImpression,
    TurnRecord,
    _ArchetypeNPCPolicy,
    obs_field,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as _npt


# --- Turn-loop driver ---


def _pick_hidden_archetype(seed: int) -> str:
    """Pick the HIDDEN opponent playstyle from a SEEDED RNG (reproducible per seed)."""
    rng = np.random.default_rng(seed)
    return str(rng.choice(PLAYSTYLES))


def print_rps_primer(output_fn: Callable[[str], None]) -> None:
    """Teach the three playstyles + the abstract counter-cycle (NO opponent reference).

    The first-time primer the CLI shows before the game starts: it teaches what-beats-what
    ABSTRACTLY (no reference to the current opponent), so the human learns the cycle
    without being told which archetype they face.
    """
    output_fn(
        "\n".join(
            [
                "=" * _BOARD_WIDTH,
                "HOT-SEAT — know your opponents (rock / paper / scissors)".center(_BOARD_WIDTH),
                "=" * _BOARD_WIDTH,
                "Three playstyles, each beaten by exactly one other:",
                "",
                "  LEAN      defend margin: high price, minimal spend, no growth.",
                "  AGGRESSOR max growth: cheap price, heavy marketing/promotion, expand fast.",
                "  PATIENT   measured re-investor: moderate everything, build steadily.",
                "",
                "  LEAN beats AGGRESSOR   (fat margin out-survives the burn)",
                "  AGGRESSOR beats PATIENT (early aggressive expansion takes share)",
                "  PATIENT beats LEAN      (compounds past the under-investor)",
                "",
                "Watch the RIVAL panel, infer the playstyle, and play its counter.",
                "=" * _BOARD_WIDTH,
            ]
        )
    )


def play_hotseat_game(
    *,
    config: Any,
    seed: int,
    n_ticks: int,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    archetype: str | None = None,
    action_fn: Callable[[Mapping[str, Any]], _npt.NDArray[np.float32]] | None = None,
) -> GameResult:
    """Run one hot-seat game: human at seat 0, the archetype NPC at seat 1, for n_ticks.

    ``archetype`` forces the playstyle (default: a seeded pick); ``action_fn`` scripts
    seat-0's action from its info instead of prompting.
    """
    from retail_simulator.core.reward import weighted_objective
    from retail_simulator.envs.parallel_env import RetailParallelEnv
    from retail_simulator.envs.parallel_gate import _components_from_info
    from retail_simulator.envs.spike_archetypes import calibrated_archetypes

    chosen_archetype = archetype if archetype is not None else _pick_hidden_archetype(seed)
    if chosen_archetype not in PLAYSTYLES:
        raise ValueError(f"unknown archetype {chosen_archetype!r}; known: {PLAYSTYLES}")
    predict = calibrated_archetypes()[chosen_archetype]

    # Seat 0 is the human (learning); seat 1 is the NPC rival. The placeholder archetype
    # only shapes the seat plan; the policy is replaced with the chosen archetype.
    env = RetailParallelEnv(config, n_learning_agents=1, npc_archetypes=("discounter",))
    env._world._npc_policies[1] = _ArchetypeNPCPolicy(predict)

    observations, infos = env.reset(seed=seed)
    # The seam's expansion mask is fixed-capacity (N_REGIONS_MAX wide); read this world's
    # real region count once and trim the mask to it everywhere.
    reset_state = env._state
    assert reset_state is not None
    n_regions = len(reset_state.regions)
    impression = RunningImpression()
    turns: list[TurnRecord] = []
    prior_profit: float | None = None
    reward_cfg = config.reward
    own_score = 0.0
    rival_score = 0.0

    for tick in range(1, n_ticks + 1):
        obs = observations["retailer_0"]
        info = infos["retailer_0"]
        if action_fn is None:
            # Interactive path: render the board and prompt the human's levers.
            action = prompt_human_action(
                obs,
                impression,
                info,
                tick=tick,
                n_ticks=n_ticks,
                n_regions=n_regions,
                input_fn=input_fn,
                output_fn=output_fn,
                prior_profit=prior_profit,
            )
        else:
            # Scripted path: action_fn supplies seat-0's action from its info; no board.
            action = action_fn(info)
        prior_profit = obs_field(obs, SELF_LAST_PROFIT_FIELD)

        observations, _rewards, _term, _trunc, infos = env.step({"retailer_0": action})

        # Score both seats on this tick's stationary weighted_objective: seat 0 from its
        # public info, seat 1 from the seam's raw stash.
        own_score += weighted_objective(_components_from_info(infos["retailer_0"]), reward_cfg)
        raw_infos = env._last_raw_infos
        assert raw_infos is not None  # set by env.step we just called
        rival_score += weighted_objective(_components_from_info(raw_infos[1]), reward_cfg)

        # Fold the post-step obs only (never the reset obs — it is seeded with true values).
        post_obs = observations["retailer_0"]
        impression.update(post_obs)
        turns.append(
            _capture_turn(
                tick, action, post_obs, impression, info["action_mask"]["expansion"][:n_regions]
            )
        )

    guess, counter_guess = _prompt_end_game_guess(input_fn=input_fn, output_fn=output_fn)
    # Final seated cash is a secondary record only; the verdict is the cumulative score
    # accumulated above.
    final_state = env._state
    assert final_state is not None  # n_ticks >= 1 ⇒ at least one step ran after reset
    own_cash = float(final_state.retailers[0].cash)
    rival_cash = float(final_state.retailers[1].cash)
    won = own_score > rival_score
    correct = guess == chosen_archetype
    _print_reveal(
        chosen_archetype,
        guess=guess,
        counter_guess=counter_guess,
        correct=correct,
        won=won,
        own_score=own_score,
        rival_score=rival_score,
        own_cash=own_cash,
        rival_cash=rival_cash,
        output_fn=output_fn,
    )
    return GameResult(
        seed=seed,
        n_ticks=n_ticks,
        archetype_actual=chosen_archetype,
        turns=turns,
        guess=guess,
        counter_guess=counter_guess,
        correct=correct,
        won=won,
        own_final_score=own_score,
        rival_final_score=rival_score,
        own_final_cash=own_cash,
        rival_final_cash=rival_cash,
    )


def _capture_turn(
    tick: int,
    action: _npt.NDArray[np.float32],
    post_obs: _npt.NDArray[np.float32],
    impression: RunningImpression,
    expansion_mask: Any,
) -> TurnRecord:
    """Build the per-tick debrief record from the action + post-step obs + impression."""
    levers = decode_action(action).levers
    human_levers = {lever: float(levers[lever]) for lever in PLAYER_LEVERS}
    raw = {
        label: obs_field(post_obs, field_name) for label, field_name in BOARD_RIVAL_FIELDS.items()
    }
    return TurnRecord(
        tick=tick,
        human_levers=human_levers,
        perceived_rival_raw=raw,
        perceived_rival_mean=impression.means(),
        own_market_share=obs_field(post_obs, SELF_MARKET_SHARE_FIELD),
        own_last_profit=obs_field(post_obs, SELF_LAST_PROFIT_FIELD),
        expansion_mask=[bool(b) for b in expansion_mask],
    )


def _prompt_playstyle(
    prompt: str, *, input_fn: Callable[[str], str], output_fn: Callable[[str], None]
) -> str:
    """Prompt for one playstyle label until a valid one (or empty → first) is given."""
    while True:
        raw = input_fn(prompt).strip().lower()
        if raw == "":
            return PLAYSTYLES[0]
        if raw in PLAYSTYLES:
            return raw
        output_fn(f"  ! pick one of {', '.join(PLAYSTYLES)} — try again.")


def _prompt_end_game_guess(
    *, input_fn: Callable[[str], str], output_fn: Callable[[str], None]
) -> tuple[str, str]:
    """Prompt the human for (opponent-playstyle guess, the playstyle they think counters it)."""
    output_fn("\nGame over. Time to call it:")
    guess = _prompt_playstyle(
        f"  Which playstyle did you face? ({'/'.join(PLAYSTYLES)}): ",
        input_fn=input_fn,
        output_fn=output_fn,
    )
    counter_guess = _prompt_playstyle(
        f"  Which playstyle counters it? ({'/'.join(PLAYSTYLES)}): ",
        input_fn=input_fn,
        output_fn=output_fn,
    )
    return guess, counter_guess


def _print_reveal(
    archetype_actual: str,
    *,
    guess: str,
    counter_guess: str,
    correct: bool,
    won: bool,
    own_score: float,
    rival_score: float,
    own_cash: float,
    rival_cash: float,
    output_fn: Callable[[str], None],
) -> None:
    """Reveal the archetype, win/loss, and counter rationale.

    The score line reports the cumulative weighted_objective (the verdict metric); final
    cash is a secondary line.
    """
    true_counter = COUNTERED_BY[archetype_actual]
    output_fn(
        "\n".join(
            [
                "=" * _BOARD_WIDTH,
                "REVEAL".center(_BOARD_WIDTH),
                "=" * _BOARD_WIDTH,
                f"  You faced       : {archetype_actual.upper()}",
                f"  Your guess      : {guess.upper()}  ({'correct' if correct else 'wrong'})",
                (f"  It is countered : {true_counter.upper()}  "
                 f"(you said {counter_guess.upper()}: "
                 f"{'right' if counter_guess == true_counter else 'not quite'})"),
                f"  Score           : you {own_score:,.1f}  vs  rival {rival_score:,.1f}",
                f"  (final cash      : you {own_cash:,.0f}  vs  rival {rival_cash:,.0f})",
                f"  Result          : {'YOU WIN' if won else 'YOU LOSE'}",
                "",
                f"  {_COUNTER_RATIONALE[archetype_actual]}",
                "=" * _BOARD_WIDTH,
            ]
        )
    )


# --- Debrief writer ---


def append_debrief(result: GameResult, path: Path) -> None:
    """Append the game record as one .jsonl line (deterministic — no wall-clock).

    Re-running the same seed with the same inputs writes a byte-identical line
    (sort_keys keeps key order stable).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _json.dumps(result.to_dict(), sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
