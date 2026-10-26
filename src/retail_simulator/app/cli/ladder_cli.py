"""CLI commands: league ladder (head-to-head ladder over participants)."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import typer

# Module-load import (def-time constraint): feeds `league ladder`'s
# `--participants` help text from the real allowlist.
from retail_simulator.envs.ladder import LADDER_ARCHETYPE_LABELS

if TYPE_CHECKING:
    # Type-only: annotations are not evaluated (PEP 563); the real imports live
    # inside the command bodies.
    from collections.abc import Callable

    from retail_simulator.core.config import CoreConfig
    from retail_simulator.envs.ladder import LadderResult

from . import _atomic_write_text, _echo, _log, _state, _status, league_app


def _split_participant_label(spec: str) -> tuple[str, str | None]:
    """Split an optional trailing ``:label`` off a participant ``spec``.

    * ``archetype:``/``random:`` specs already carry a colon, so an explicit
      label is a third segment (``archetype:lean:mylabel``); any other part
      count or an empty segment (e.g. the typo ``archetype:lean:``) is a
      ``ValueError``.
    * Anything else splits on the LAST ``:`` (checkpoint paths never contain
      one on POSIX); an empty label after a trailing ``:`` is a ``ValueError``.

    Returns ``(raw, explicit_label)``; ``explicit_label`` is ``None`` when the
    caller supplied none (the spec's own ``default_label`` applies).
    """
    from retail_simulator.envs.ladder import (
        ARCHETYPE_SPEC_PREFIX,
        RANDOM_SPEC_PREFIX,
    )

    if spec.startswith((ARCHETYPE_SPEC_PREFIX, RANDOM_SPEC_PREFIX)):
        parts = spec.split(":")
        if len(parts) == 2:
            return spec, None
        if len(parts) == 3 and all(parts):
            return f"{parts[0]}:{parts[1]}", parts[2]
        raise ValueError(
            f"participant {spec!r} must be '<prefix>:<value>' or "
            "'<prefix>:<value>:<label>' (no empty segments)"
        )
    if ":" in spec:
        raw, _, label = spec.rpartition(":")
        if not label:
            raise ValueError(f"participant {spec!r} has an empty label after ':'")
        return raw, label
    return spec, None


def _format_ladder_table(result: LadderResult) -> str:
    """Render a :class:`LadderResult` as a plain-ASCII ranking table (no color).

    Blocks: the ranking rows (field margin + CI, and the revenue/share/profit
    component margins — ``"n/a"`` when absent), the battle-pair rows, and a
    footer with the split-half Kendall-τ and the tournament summary (Condorcet
    winner, cyclic/decisive triples — a trailing ``*`` marks a triple containing
    a checkpoint). A ``homes=`` line appears only when ``home_regions`` is set;
    ``loyalty`` is never rendered.
    """
    lines = [
        f"ladder_id={result.ladder_id}",
        f"env={result.config_label}",
    ]
    if result.home_regions is not None:
        lines.append(f"homes={list(result.home_regions)}")
    lines.append(
        f"seeds: n={result.n_seeds} set={list(result.seed_set)} n_episodes={result.n_episodes}"
    )
    lines.append("")
    lines.append(
        f"{'rank':>4} {'label':<24} {'kind':<10} {'field_margin':>14} "
        f"{'ci_low':>14} {'ci_high':>14} {'rev_fm':>12} {'share_fm':>10} {'profit_fm':>12}"
    )
    for entry in result.entries:
        fm = entry.component_field_margins
        rev_fm = "n/a" if fm is None else f"{fm['revenue']:.1f}"
        share_fm = "n/a" if fm is None else f"{fm['market_share']:.4f}"
        profit_fm = "n/a" if fm is None else f"{fm['profit']:.1f}"
        lines.append(
            f"{entry.rank:>4} {entry.label:<24} {entry.kind:<10} {entry.field_margin:>14.6f} "
            f"{entry.ci_low:>14.6f} {entry.ci_high:>14.6f} {rev_fm:>12} {share_fm:>10} "
            f"{profit_fm:>12}"
        )

    lines.append("")
    lines.append("battle pairs (A over B):")
    lines.append(
        f"{'label_a':<24} {'label_b':<24} {'score_mean':>14} {'score_ci':>34} "
        f"{'revenue':>12} {'share':>10} {'profit':>12}"
    )
    for pair in result.pairwise:
        score_ci = (
            "n/a"
            if pair.ci_low is None or pair.ci_high is None
            else f"[{pair.ci_low:.6f}, {pair.ci_high:.6f}]"
        )
        cm = pair.component_mean_margins
        revenue = "n/a" if cm is None else f"{cm['revenue']:.1f}"
        share = "n/a" if cm is None else f"{cm['market_share']:.4f}"
        profit = "n/a" if cm is None else f"{cm['profit']:.1f}"
        lines.append(
            f"{pair.label_a:<24} {pair.label_b:<24} {pair.mean_margin:>14.6f} {score_ci:>34} "
            f"{revenue:>12} {share:>10} {profit:>12}"
        )

    lines.append("")
    tau = result.kendall_tau_split_half
    tau_str = "n/a (n_seeds < 4)" if tau is None else f"{tau:.6f}"
    lines.append(f"kendall_tau_split_half (REPORTED): {tau_str}")

    tournament = result.tournament
    if tournament is None:
        lines.append("tournament: n/a")
    else:
        condorcet_winner = (
            "none" if tournament.condorcet_winner is None else tournament.condorcet_winner
        )
        lines.append(
            f"tournament: condorcet_winner={condorcet_winner} "
            f"cyclic_triples={tournament.n_cyclic_triples}/{tournament.total_triples} "
            f"decisive_cycle={tournament.decisive_cycle} "
            f"decisive_triples={len(tournament.decisive_triples)} "
            f"(checkpoint-inclusive: {tournament.checkpoint_inclusive_decisive_triples})"
        )
        if tournament.decisive_triples:
            kind_by_label = {entry.label: entry.kind for entry in result.entries}
            for x, y, z in tournament.decisive_triples:
                is_checkpoint_inclusive = any(
                    kind_by_label[label] == "checkpoint" for label in (x, y, z)
                )
                marker = " *" if is_checkpoint_inclusive else ""
                lines.append(f"  {x} > {y} > {z} > {x}{marker}")
    return "\n".join(lines)


def _ladder_env_default() -> tuple[CoreConfig | None, str]:
    """The ``default`` ``--env`` value: no config override — ``run_ladder`` falls
    back to its own ``competition_config()`` default."""
    from retail_simulator.envs.ladder import DEFAULT_ENV_LABEL

    return None, DEFAULT_ENV_LABEL


def _ladder_env_calibrated() -> tuple[CoreConfig | None, str]:
    """The ``calibrated`` ``--env`` value: ``calibrated_spike_config()``."""
    from retail_simulator.envs.ladder import CALIBRATED_ENV_LABEL
    from retail_simulator.envs.spike import calibrated_spike_config

    return calibrated_spike_config(), CALIBRATED_ENV_LABEL


def _ladder_env_arena() -> tuple[CoreConfig | None, str]:
    """The ``arena`` ``--env`` value: the 25-region candidate-lock arena.

    Returns only the ``(config, config_label)`` pair; the two-seat arena seating
    is resolved separately in :func:`league_ladder`.
    """
    from retail_simulator.envs.arena import ARENA_ENV_LABEL, arena_lock_config

    return arena_lock_config(), ARENA_ENV_LABEL


def _ladder_env_arena_balanced() -> tuple[CoreConfig | None, str]:
    """The ``arena-balanced`` ``--env`` value: the same 25-region arena with a
    working-capital credit line and a smaller warehouse footprint.

    Seating is resolved separately in :func:`league_ladder`, like ``arena``.
    """
    from retail_simulator.envs.arena import (
        ARENA_BALANCED_ENV_LABEL,
        arena_balanced_config,
    )

    return arena_balanced_config(), ARENA_BALANCED_ENV_LABEL


# `--env` name -> zero-arg ``(config, config_label)`` builder. Single source of
# truth for the `--env` help text and the unknown-env [FAIL] allowlist (no
# hand-maintained name list); each builder's heavy imports run only when invoked.
_LADDER_ENV_BUILDERS: dict[str, Callable[[], tuple[CoreConfig | None, str]]] = {
    "default": _ladder_env_default,
    "calibrated": _ladder_env_calibrated,
    "arena": _ladder_env_arena,
    "arena-balanced": _ladder_env_arena_balanced,
}


def _resolve_ladder_env(env: str) -> tuple[CoreConfig | None, str]:
    """Resolve an ``--env`` name to a ``(config, config_label)`` pair.

    Dispatches through :data:`_LADDER_ENV_BUILDERS`; an unknown name raises
    ``ValueError`` listing the same allowlist.
    """
    try:
        builder = _LADDER_ENV_BUILDERS[env]
    except KeyError:
        allowed = ", ".join(_LADDER_ENV_BUILDERS)
        raise ValueError(f"unknown --env {env!r}; must be one of: {allowed}") from None
    return builder()


@league_app.command("ladder")
def league_ladder(
    participants: list[str] = typer.Option(
        ...,
        "--participants",
        help="Repeatable participant spec: a checkpoint zip `path` (or `path:label`), "
        "`archetype:<name>` for a calibrated or geo archetype "
        f"({'/'.join(LADDER_ARCHETYPE_LABELS)}), or "
        "`random:<seed>` for a random baseline — the latter two optionally suffixed "
        "`:label` (e.g. `random:3:baseline`; a checkpoint path keeps its own `path:label` "
        "form). When no label is given one is derived (the zip stem, the archetype name, "
        "or `random_<seed>`). A geo archetype's expansion targeting only makes sense on a "
        "multi-region world (`--env arena`) — on the smaller default/calibrated env it "
        "still runs, just as degenerately as a calibrated archetype does on the arena. "
        "Require >=2; a checkpoint path must exist and end in `.zip`; labels must be "
        "unique.",
    ),
    n_seeds: int = typer.Option(
        30,
        "--n-seeds",
        min=1,
        help="Fixed shared seed-set size; every unordered pair is played position-controlled over seeds 0..n_seeds-1 (the fairness contract).",
    ),
    n_episodes: int = typer.Option(
        1,
        "--n-episodes",
        min=1,
        help="Position-controlled head-to-head episodes per (pair, seed).",
    ),
    bootstrap_seed: int = typer.Option(
        0,
        "--bootstrap-seed",
        help="Seeds the bootstrap-CI seed-resample RNG (kept SEPARATE from the rollout seeds; "
        "fixing it makes the CIs — and the whole ladder.json — bit-reproducible).",
    ),
    out: Path = typer.Option(
        None,
        "--out",
        help="Path for the ladder.json artifact (default runs/<ladder_id>/ladder.json).",
    ),
    table: bool = typer.Option(
        False,
        "--table",
        help="Print an ASCII ranking table instead of the JSON payload (the artifact is "
        "still written either way).",
    ),
    env: str = typer.Option(
        "default",
        "--env",
        help="Which CoreConfig the ladder rolls out on — one of: "
        + ", ".join(_LADDER_ENV_BUILDERS)
        + " ('default' = competition_config; 'calibrated' = "
        "calibrated_spike_config; 'arena' = the 25-region candidate-lock "
        "arena; 'arena-balanced' = the same 25-region arena with a "
        "working-capital credit line + a smaller warehouse footprint "
        "— both arena envs seat both participants at "
        "arena_home_regions(2)). Recorded in the artifact as config_label; folds "
        "into the default ladder_id only when non-default, so every default-env "
        "ladder_id stays byte-identical.",
    ),
) -> None:
    """Run a head-to-head ladder over the given participants; write ladder.json and print the report."""
    from retail_simulator.envs.arena import arena_home_regions
    from retail_simulator.envs.ladder import parse_participant_spec, run_ladder

    # -- Boundary validation: --env ------------------------------------------------------
    try:
        config, config_label = _resolve_ladder_env(env)
    except ValueError as exc:
        _echo(_status("[FAIL]", str(exc), "red"))
        raise typer.Exit(code=1) from exc

    # Arena seating: two seats (learner + opponent) on the arena's home regions;
    # "arena-balanced" shares the same 25-region table, so it seats identically.
    home_regions = arena_home_regions(2) if env in ("arena", "arena-balanced") else None

    # -- Boundary validation: --participants ---------------------------------------------
    # Surface operator typos (too few, malformed spec, missing/non-zip checkpoint,
    # duplicate label) as clean [FAIL] + exit 1 before any rollout.
    if len(participants) < 2:
        _echo(
            _status(
                "[FAIL]",
                f"--participants requires at least 2 entries; got {len(participants)}",
                "red",
            )
        )
        raise typer.Exit(code=1)

    parsed: list[tuple[str, str]] = []
    seen_labels: set[str] = set()
    for spec in participants:
        try:
            raw, explicit_label = _split_participant_label(spec)
            participant_spec = parse_participant_spec(raw)
        except ValueError as exc:
            _echo(_status("[FAIL]", str(exc), "red"))
            raise typer.Exit(code=1) from exc
        label = explicit_label if explicit_label is not None else participant_spec.default_label
        # `.zip` + existence checks are checkpoint-only — archetype/random specs
        # have no filesystem footprint and are passed through verbatim.
        source = participant_spec.source
        if participant_spec.kind == "checkpoint":
            path = Path(source)
            if path.suffix != ".zip":
                _echo(_status("[FAIL]", f"participant path {path} must end in '.zip'", "red"))
                raise typer.Exit(code=1)
            if not path.is_file():
                _echo(_status("[FAIL]", f"participant checkpoint not found: {path}", "red"))
                raise typer.Exit(code=1)
            source = str(path)
        if label in seen_labels:
            _echo(_status("[FAIL]", f"duplicate participant label {label!r}", "red"))
            raise typer.Exit(code=1)
        seen_labels.add(label)
        parsed.append((source, label))

    _log.info(
        "running ladder: participants=%d, n_seeds=%d, n_episodes=%d, bootstrap_seed=%d, env=%s",
        len(parsed),
        n_seeds,
        n_episodes,
        bootstrap_seed,
        config_label,
    )
    result = run_ladder(
        parsed,
        n_seeds=n_seeds,
        n_episodes=n_episodes,
        bootstrap_seed=bootstrap_seed,
        config=config,
        config_label=config_label,
        home_regions=home_regions,
    )

    # Default artifact path: <run-dir>/<ladder_id>/ladder.json.
    out_path = (
        out if out is not None else Path(_state["run_dir"]) / result.ladder_id / "ladder.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        out_path,
        json.dumps(asdict(result), indent=2, sort_keys=True),
    )
    _echo(f"wrote ladder artifact: {out_path}")

    if table:
        _echo(_format_ladder_table(result))
    else:
        _echo(json.dumps(asdict(result), indent=2, sort_keys=True))


