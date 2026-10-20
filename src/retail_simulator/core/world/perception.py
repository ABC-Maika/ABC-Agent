"""Perception helpers: noise and seat each learning agent's perceived competitor view."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import CoreConfig
from retail_simulator.core.demand import (
    _presence_matrix,
    _reference_prices,
    _retailer_price_indices,
)
from retail_simulator.core.encoding import true_competitor_view
from retail_simulator.core.state import PerceivedCompetitor, WorldState

_PERCEIVED_FIELDS: tuple[tuple[str, str], ...] = (
    ("competitor_price", "price"),
    ("competitor_market_share", "unit"),
    ("competitor_awareness", "awareness"),
    ("competitor_assortment", "unit"),
    ("competitor_promotion", "unit"),
    ("competitor_loyalty_region_0", "unit"),
    ("competitor_awareness_region_1", "awareness"),
    ("competitor_loyalty_region_1", "unit"),
    ("competitor_market_share_region_1", "unit"),
    ("competitor_mean_price_region_0", "price"),
    ("competitor_mean_price_region_1", "price"),
)


def _bounds_for_kind(
    kind: str, price_bounds: tuple[float, float], awareness_cap: float
) -> tuple[float, float]:
    """(low, high) bounds for one perceived-field kind."""
    if kind == "price":
        return price_bounds
    if kind == "awareness":
        return (0.0, awareness_cap)
    return (0.0, 1.0)  # "unit": shares / loyalty / assortment / promotion


def _perceived_field_bounds(config: CoreConfig) -> list[tuple[float, float]]:
    """Bounds for the base per-region perceived fields."""
    price_bounds = (config.pricing.min_price_index, config.pricing.max_price_index)
    awareness_cap = 1.0 / config.marketing.decay
    return [
        _bounds_for_kind(kind, price_bounds, awareness_cap) for _name, kind in _PERCEIVED_FIELDS
    ]


_EXTRA_REGION_FIELD_KINDS: tuple[str, str, str, str] = ("awareness", "unit", "unit", "price")


def _extra_region_bounds(config: CoreConfig) -> list[tuple[float, float]]:
    """Per-column (low, high) bounds for one extra-region (r >= 2) perceived block."""
    price_bounds = (config.pricing.min_price_index, config.pricing.max_price_index)
    awareness_cap = 1.0 / config.marketing.decay
    return [
        _bounds_for_kind(kind, price_bounds, awareness_cap) for kind in _EXTRA_REGION_FIELD_KINDS
    ]


def _learning_seat_indices(state: WorldState) -> list[int]:
    """Indices of the learning (non-NPC) seats."""
    return [idx for idx, r in enumerate(state.retailers) if not r.is_npc]


def _true_reference_means(state: WorldState) -> npt.NDArray[np.float64]:
    """Per-region reference prices (mean over present others)."""
    n_regions = len(state.regions)
    prices = _retailer_price_indices(state)  # [K]
    presence = _presence_matrix(state, n_regions)  # [R, K]
    return _reference_prices(prices, presence)  # [R, K]


def _reference_wages(state: WorldState) -> npt.NDArray[np.float64]:
    """Per-region mean wage of the OTHER present retailers, per retailer.

    A sole-present retailer falls back to its own wage.
    """
    n_regions = len(state.regions)
    wages = np.array([r.wage_level for r in state.retailers], dtype=np.float64)  # [K]
    presence = _presence_matrix(state, n_regions)  # [R, K]
    return _reference_prices(wages, presence)  # [R, K]


def _observer_reference_overrides(
    ref: npt.NDArray[np.float64], obs_idx: int, n_regions: int
) -> dict[str, float]:
    """Mean-price field overrides for one observer's perceived view."""
    region_0 = float(ref[0, obs_idx])
    region_1 = float(ref[1, obs_idx]) if n_regions > 1 else region_0
    overrides = {
        "competitor_mean_price_region_0": region_0,
        "competitor_mean_price_region_1": region_1,
    }
    for r in range(2, n_regions):
        overrides[f"competitor_mean_price_region_{r}"] = float(ref[r, obs_idx])
    return overrides


def _seat_perceived_views(
    next_state: WorldState,
    research_by_seat: dict[int, float],
    config: CoreConfig,
    rng: np.random.Generator,
) -> None:
    """Noise each observer's perceived competitor view and write it into the state."""
    observers = _learning_seat_indices(next_state)
    n_obs = len(observers)
    n_regions = len(next_state.regions)
    n_extra_regions = max(0, n_regions - 2)
    n_fields = len(_PERCEIVED_FIELDS) + 4 * n_extra_regions
    noise = rng.standard_normal(size=(n_obs, n_fields))
    sigma_base = config.research.noise_sigma_base
    bounds = _perceived_field_bounds(config) + _extra_region_bounds(config) * n_extra_regions
    ref = _true_reference_means(next_state)  # [R, K]
    for row, obs_idx in enumerate(observers):
        fidelity = config.research.fidelity(research_by_seat.get(obs_idx, 0.0))
        scale = sigma_base * (1.0 - fidelity)
        true_view = true_competitor_view(next_state, obs_idx)
        # Guard: both paths must agree on the number of extra regions.
        assert len(true_view.extra_regions) == n_extra_regions, (
            f"true_competitor_view seated {len(true_view.extra_regions)} extra regions "
            f"but this seam expects {n_extra_regions} (n_regions={n_regions}) — the two "
            "derivations have drifted apart"
        )
        # The mean-price fields use the per-observer (self-excluded) reference values.
        ref_overrides = _observer_reference_overrides(ref, obs_idx, n_regions)
        noised: dict[str, float] = {}
        for col, (name, _kind) in enumerate(_PERCEIVED_FIELDS):
            true_value = ref_overrides.get(name, float(getattr(true_view, name)))
            low, high = bounds[col]
            value = true_value + float(noise[row, col]) * scale
            noised[name] = min(max(value, low), high)
        extra_regions: list[tuple[float, float, float, float]] = []
        base_col = len(_PERCEIVED_FIELDS)
        for i, (true_awareness, true_loyalty, true_share, _placeholder) in enumerate(
            true_view.extra_regions
        ):
            r = i + 2
            true_region_values = (
                true_awareness,
                true_loyalty,
                true_share,
                ref_overrides[f"competitor_mean_price_region_{r}"],
            )
            region_out = []
            for j in range(4):
                col = base_col + i * 4 + j
                low, high = bounds[col]
                value = true_region_values[j] + float(noise[row, col]) * scale
                region_out.append(min(max(value, low), high))
            extra_regions.append((region_out[0], region_out[1], region_out[2], region_out[3]))
        next_state.perceived_competitors[obs_idx] = PerceivedCompetitor(
            **noised, extra_regions=tuple(extra_regions)
        )
        next_state.research_fidelity[obs_idx] = fidelity


def _seat_true_perceived_views(state: WorldState, config: CoreConfig) -> None:
    """Seat each observer's perceived view at the true values (reset placeholder)."""
    del config
    n_regions = len(state.regions)
    ref = _true_reference_means(state)  # [R, K]
    n_extra_regions = max(0, n_regions - 2)
    for obs_idx in _learning_seat_indices(state):
        true_view = true_competitor_view(state, obs_idx)
        # Guard: both paths must agree on the number of extra regions.
        assert len(true_view.extra_regions) == n_extra_regions, (
            f"true_competitor_view seated {len(true_view.extra_regions)} extra regions "
            f"but this seam expects {n_extra_regions} (n_regions={n_regions}) — the two "
            "derivations have drifted apart"
        )
        ref_overrides = _observer_reference_overrides(ref, obs_idx, n_regions)
        extra_regions = tuple(
            (awareness, loyalty, share, ref_overrides[f"competitor_mean_price_region_{i + 2}"])
            for i, (awareness, loyalty, share, _placeholder) in enumerate(true_view.extra_regions)
        )
        state.perceived_competitors[obs_idx] = replace(
            true_view,
            competitor_mean_price_region_0=ref_overrides["competitor_mean_price_region_0"],
            competitor_mean_price_region_1=ref_overrides["competitor_mean_price_region_1"],
            extra_regions=extra_regions,
        )
        state.research_fidelity[obs_idx] = 0.0
