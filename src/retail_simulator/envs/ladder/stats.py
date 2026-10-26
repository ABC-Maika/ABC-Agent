"""Statistical helpers for the ladder: ranking, margin CIs, and content hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, NamedTuple

import numpy as np

from .model import DEFAULT_ENV_LABEL


def _kendall_tau(rank_a: Sequence[int], rank_b: Sequence[int]) -> float:
    """Kendall tau between two total orders (ties count as neither)."""
    if len(rank_a) != len(rank_b):
        raise ValueError(
            f"_kendall_tau requires equal-length rankings; "
            f"got len(rank_a)={len(rank_a)}, len(rank_b)={len(rank_b)}"
        )
    n = len(rank_a)
    if n < 2:
        # No pairs to compare; 1.0 keeps the diagnostic defined at the edge.
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a_order = rank_a[i] - rank_a[j]
            b_order = rank_b[i] - rank_b[j]
            product = a_order * b_order
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            # product == 0: a tie, counted as neither (τ-a convention).
    total_pairs = n * (n - 1) // 2
    return float(concordant - discordant) / float(total_pairs)


def _field_margin_from_pair_means(
    label: str,
    others: Sequence[str],
    pair_mean: dict[tuple[str, str], float],
) -> float:
    """Mean of ``label``'s margin over every other participant in ``others``.

    ``pair_mean`` stores A-over-B per sorted pair; the margin is negated when
    ``label`` is the pair's second member.
    """
    margins: list[float] = []
    for other in others:
        if label < other:
            margins.append(pair_mean[(label, other)])
        else:
            margins.append(-pair_mean[(other, label)])
    return float(np.mean(margins))


def _bootstrap_margin_ci(
    label: str,
    others: Sequence[str],
    pair_per_seed: dict[tuple[str, str], np.ndarray],
    *,
    n_seeds: int,
    resampled_seed_indices: np.ndarray,
) -> tuple[float, float]:
    """Bootstrap 95% CI on ``label``'s field margin, resampling the seed axis.

    Each row of ``resampled_seed_indices`` (n_resamples, n_seeds) is applied to
    every pair consistently, so each iteration is one coherent surrogate seed
    set; the CI is the 2.5th/97.5th percentile of the resampled means.
    """
    # Orient each pairing label-over-other into a (n_others, n_seeds) matrix.
    oriented_rows: list[np.ndarray] = []
    for other in others:
        if label < other:
            oriented_rows.append(pair_per_seed[(label, other)])
        else:
            oriented_rows.append(-pair_per_seed[(other, label)])
    oriented = np.asarray(oriented_rows, dtype=np.float64)
    # Mean over the others axis is the field margin per seed.
    field_per_seed = oriented.mean(axis=0)  # shape (n_seeds,)
    # Average field-per-seed over each resampled seed row.
    boot_field = field_per_seed[resampled_seed_indices].mean(axis=1)  # (n_resamples,)
    ci_low = float(np.percentile(boot_field, 2.5))
    ci_high = float(np.percentile(boot_field, 97.5))
    return ci_low, ci_high


def _content_hash_ladder_id(
    participants: Sequence[tuple[str, str]],
    *,
    n_seeds: int,
    n_episodes: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    config_label: str = DEFAULT_ENV_LABEL,
    home_regions: tuple[int, ...] | None = None,
) -> str:
    """Derive a stable ladder_id from the field and run settings (config label / home regions only when non-default)."""
    fingerprint: dict[str, Any] = {
        "participants": [list(p) for p in participants],
        "n_seeds": n_seeds,
        "n_episodes": n_episodes,
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
    }
    if config_label != DEFAULT_ENV_LABEL:
        fingerprint["config_label"] = config_label
    if home_regions is not None:
        fingerprint["home_regions"] = list(home_regions)
    digest = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()
    return f"ladder-{digest[:12]}"


class _ScoreMarginRow(NamedTuple):
    """One pair's score margin + bootstrap CI as concrete floats.

    Kept separate from PairwiseMargin (whose CIs are ``float | None``) so
    _build_tournament_summary never has to narrow an Optional.
    """

    label_a: str
    label_b: str
    per_seed_margins: tuple[float, ...]
    mean_margin: float
    margin_se: float
    ci_low: float
    ci_high: float


def _rank_labels_on_seed_subset(
    labels: Sequence[str],
    pair_per_seed: dict[tuple[str, str], np.ndarray],
    seed_indices: np.ndarray,
) -> list[str]:
    """Rank labels by field-margin computed over only seed_indices.

    Used by the split-half diagnostic (evens / odds seed subsets).
    """
    field_by_label: dict[str, float] = {}
    for label in labels:
        others = [other for other in labels if other != label]
        oriented_rows: list[np.ndarray] = []
        for other in others:
            if label < other:
                oriented_rows.append(pair_per_seed[(label, other)][seed_indices])
            else:
                oriented_rows.append(-pair_per_seed[(other, label)][seed_indices])
        oriented = np.asarray(oriented_rows, dtype=np.float64)
        field_by_label[label] = float(oriented.mean())
    return sorted(labels, key=lambda lbl: (-field_by_label[lbl], lbl))


def _split_half_kendall_tau(
    labels: Sequence[str],
    pair_per_seed: dict[tuple[str, str], np.ndarray],
    *,
    n_seeds: int,
) -> float | None:
    """Kendall tau between the field rankings over even and odd seeds (None below 4 seeds)."""
    if n_seeds < 4:
        return None
    even_indices = np.arange(0, n_seeds, 2)
    odd_indices = np.arange(1, n_seeds, 2)
    even_order = _rank_labels_on_seed_subset(labels, pair_per_seed, even_indices)
    odd_order = _rank_labels_on_seed_subset(labels, pair_per_seed, odd_indices)
    # Map each label to its rank in each half, then τ over those ranks in a fixed label order.
    even_rank = {label: pos for pos, label in enumerate(even_order)}
    odd_rank = {label: pos for pos, label in enumerate(odd_order)}
    fixed = sorted(labels)
    rank_a = [even_rank[label] for label in fixed]
    rank_b = [odd_rank[label] for label in fixed]
    return _kendall_tau(rank_a, rank_b)
