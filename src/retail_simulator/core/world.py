"""The world facade: the only place the world advances.

``World`` ties state, config, RNG, encoding, demand, accounting, reward,
and NPC together into one canonical transition that the Gymnasium and
PettingZoo wrappers adapt.

Determinism: one injected ``numpy.Generator`` is drawn in a fixed per-tick
order inside :meth:`World.step` (demand noise, NPC actions, accounting
noise, perception noise, wage write-off). ``(state, World.rng_state())``
fully serializes the simulation. ``World`` keeps no per-tick history and
never mutates the state passed to :meth:`step`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import numpy.typing as npt

from retail_simulator.core.accounting import apply_accounting
from retail_simulator.core.affordability import (
    compute_spend_clamp,
    gated_automation_capex,
    gated_expansion_capex,
    intended_discretionary_spend,
    scale_discretionary_levers,
)
from retail_simulator.core.config import CoreConfig, SeatSpec
from retail_simulator.core.demand import (
    _capacity_footprint,
    _decode_levers,
    _presence_matrix,
    _reference_prices,
    _retailer_price_indices,
    effective_warehouse_capacity,
    resolve_demand,
)
from retail_simulator.core.encoding import (
    decode_action,
    encode_observation,
    true_competitor_view,
)
from retail_simulator.core.npc import Balanced, Discounter, NPCPolicy, Premium, Wandering
from retail_simulator.core.reward import (
    RewardNormalizer,
    compute_reward_for_seat,
    reward_components,
)
from retail_simulator.core.rng import make_rng, snapshot
from retail_simulator.core.schema import (
    N_AUTOMATION_TIERS,
    N_REGIONS_MAX,
    ActionSchema,
    default_action_schema,
    validate_automation_width,
    validate_expansion_capacity,
)
from retail_simulator.core.state import (
    PerceivedCompetitor,
    RegionState,
    RetailerState,
    WorldState,
    default_segments,
)

# Seat 0 is the learning agent; seat 1 is the discounter NPC.
AGENT_INDEX: int = 0


def default_seat_plan() -> tuple[SeatSpec, ...]:
    """Default two-seat plan: learning agent[0] + discounter NPC[1]."""
    return (
        SeatSpec(is_npc=False, name="agent"),
        SeatSpec(is_npc=True, archetype="discounter", name="discounter"),
    )


def _make_npc_policy(archetype: str, schema: ActionSchema) -> NPCPolicy:
    """Build the NPC policy for an archetype name."""
    if archetype == "discounter":
        return Discounter(schema=schema)
    if archetype == "premium":
        return Premium(schema=schema)
    if archetype == "balanced":
        return Balanced(schema=schema)
    if archetype == "wandering":
        return Wandering(schema=schema)
    raise ValueError(f"unknown NPC archetype {archetype!r}")


def _default_seat_name(spec: SeatSpec, seat_index: int) -> str:
    """Display name for a seat without one: ``retailer_{i}`` or ``{archetype}_{i}``."""
    if spec.is_npc:
        return f"{spec.archetype}_{seat_index}"
    return f"retailer_{seat_index}"


STARTING_CASH: float = 15_000.0
INITIAL_LOYALTY: float = 0.0
INITIAL_AWARENESS: float = 0.0
INITIAL_STOCKPILE: float = 0.0
INITIAL_SERVICE_SCORE: float = 1.0
INITIAL_AUTOMATION_TIER: int = 0


def compute_expansion_mask(
    state: WorldState, agent_index: int, config: CoreConfig
) -> npt.NDArray[np.bool_]:
    """Regions where the agent can open a store (affordable, not already present)."""
    me = state.retailers[agent_index]
    capex = config.expansion.expansion_capex
    n_regions = len(state.regions)
    mask = np.zeros(N_REGIONS_MAX, dtype=bool)
    mask[0] = True
    for r in range(1, n_regions):
        already_present = r < len(me.stores_per_region) and me.stores_per_region[r] > 0
        mask[r] = (me.cash >= capex) and (not already_present)
    return mask


def compute_automation_mask(
    state: WorldState, agent_index: int, config: CoreConfig
) -> npt.NDArray[np.bool_]:
    """Automation tiers the agent can afford to step up to."""
    me = state.retailers[agent_index]
    capex_per_tier = config.automation.capex_per_tier
    prev = me.automation_tier
    mask = np.zeros(N_AUTOMATION_TIERS, dtype=bool)
    for t in range(N_AUTOMATION_TIERS):
        monotone = t >= prev
        differential = capex_per_tier[t] - capex_per_tier[prev]
        affordable = me.cash >= differential
        mask[t] = monotone and affordable
    return mask


def _restructuring_survivor_regions(
    stores_per_region: tuple[int, ...],
    last_market_share_per_region: tuple[float, ...],
    n_regions: int,
    keep_stores: int,
) -> set[int]:
    """Present regions to keep: the highest-market-share ones, down to ``keep_stores``."""
    present = [r for r in range(1, n_regions) if stores_per_region[r] > 0]
    ranked = sorted(present, key=lambda r: (-last_market_share_per_region[r], r))
    return set(ranked[:keep_stores])


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


@dataclass(frozen=True)
class StepResult:
    """Outcome of one :meth:`World.step`, keyed by seat index."""

    next_state: WorldState
    observations: dict[int, npt.NDArray[np.float32]]
    rewards: dict[int, float]
    infos: dict[int, dict[str, object]]
    terminated: dict[int, bool] = field(default_factory=dict)
    truncated: dict[int, bool] = field(default_factory=dict)


class World:
    """The world: build the initial state, run ticks, emit observations and rewards."""

    def __init__(self, config: CoreConfig | None = None, seed: int | None = None) -> None:
        self._config = config if config is not None else CoreConfig.default()
        self._schema: ActionSchema = default_action_schema()
        regions = self._config.demand.regions
        n_regions = len(regions) if regions else 1
        validate_expansion_capacity(n_regions)
        # The automation lever's width must match the number of configured tiers.
        validate_automation_width(len(self._config.automation.capex_per_tier))
        if STARTING_CASH < self._config.expansion.expansion_capex:
            raise ValueError(
                f"STARTING_CASH ({STARTING_CASH}) must be >= expansion_capex "
                f"({self._config.expansion.expansion_capex}) or the expansion lever is dead"
            )
        self._rng: np.random.Generator = make_rng(seed)
        # Resolve the seat plan and build one NPC policy per NPC seat.
        # Seat 0 must be a learning seat; NPC policies are called in seat order in step().
        self._seat_plan: tuple[SeatSpec, ...] = (
            self._config.seats if self._config.seats is not None else default_seat_plan()
        )
        if not self._seat_plan:
            raise ValueError("seat plan must have at least one seat")
        if self._seat_plan[0].is_npc:
            raise ValueError("seat 0 must be a learning seat (AGENT_INDEX = 0)")
        self._npc_policies: dict[int, NPCPolicy] = {
            idx: _make_npc_policy(spec.archetype, self._schema)
            for idx, spec in enumerate(self._seat_plan)
            if spec.is_npc and spec.archetype is not None
        }

    @property
    def config(self) -> CoreConfig:
        """The resolved configuration backing this world."""
        return self._config

    @property
    def rng(self) -> np.random.Generator:
        """The injected generator (exposed so wrappers can serialize it)."""
        return self._rng

    def rng_state(self) -> dict[str, object]:
        """Serializable RNG state; pair with a ``WorldState`` to checkpoint."""
        return snapshot(self._rng)

    def reset(self, seed: int | None = None) -> WorldState:
        """Build the initial ``WorldState``; re-seed the RNG when ``seed`` given."""
        if seed is not None:
            self._rng = make_rng(seed)

        # Build the regions from config (fallback: a single region). Realized demand and stockpile start empty.
        region_cfgs = self._config.demand.regions
        regions: list[RegionState]
        if region_cfgs:
            regions = [
                RegionState(
                    population=rc.base_regional_demand,
                    base_regional_demand=rc.base_regional_demand,
                    segment_mix=dict(rc.segment_mix),
                    realized_regional_demand=0.0,
                    stockpile=INITIAL_STOCKPILE,
                )
                for rc in region_cfgs
            ]
        else:
            regions = [
                RegionState(
                    population=self._config.demand.base_regional_demand,
                    base_regional_demand=self._config.demand.base_regional_demand,
                    segment_mix=dict(self._config.demand.segment_mix),
                    realized_regional_demand=0.0,
                    stockpile=INITIAL_STOCKPILE,
                )
            ]
        n_regions = len(regions)
        zeros = (0.0,) * n_regions

        retailers: list[RetailerState] = []
        initial_service_score = tuple(INITIAL_SERVICE_SCORE for _ in range(n_regions))
        for idx, spec in enumerate(self._seat_plan):
            if spec.presence is not None:
                stores = tuple(
                    spec.presence[r] if r < len(spec.presence) else 0 for r in range(n_regions)
                )
            elif spec.is_npc:
                stores = tuple(1 for _ in range(n_regions))
            else:
                stores = tuple(1 if r == 0 else 0 for r in range(n_regions))
            ramp_ticks = self._config.warehouse.ramp_ticks
            initial_store_age = tuple(ramp_ticks if stores[r] > 0 else 0 for r in range(n_regions))
            retailers.append(
                RetailerState(
                    name=spec.name if spec.name is not None else _default_seat_name(spec, idx),
                    is_npc=spec.is_npc,
                    cash=STARTING_CASH,
                    price_index=self._config.pricing.baseline_price_index,
                    stores_per_region=stores,
                    store_age_per_region=initial_store_age,
                    awareness_per_region=zeros,
                    loyalty_per_region=zeros,
                    last_market_share_per_region=zeros,
                    service_score_per_region=initial_service_score,
                    wage_level=self._config.wage.wage_base,
                    employee_happiness=1.0,
                    accumulated_overhead_basis=0.0,
                    last_turnover=0.0,
                    warehouse_capacity=self._config.warehouse.default_warehouse_capacity,
                    # No seat starts insolvent or restructured.
                    insolvent_ticks=0,
                    restructurings=0,
                    # No seat holds a pending action at reset: every NPC holds course on tick 0.
                    pending_action=None,
                )
            )

        reward_stats: dict[int, RewardNormalizer] = {}
        if self._config.reward.mode == "weighted":
            reward_stats = {
                idx: RewardNormalizer() for idx, r in enumerate(retailers) if not r.is_npc
            }

        segments = (
            dict(self._config.segments) if self._config.segments is not None else default_segments()
        )
        state = WorldState(
            tick=0,
            regions=regions,
            retailers=retailers,
            segments=segments,
            reward_stats=reward_stats,
        )
        # Seat perceived competitor views at the true values (no perception draw at reset).
        _seat_true_perceived_views(state, self._config)
        return state

    def agent_observation(
        self, state: WorldState, agent_index: int = AGENT_INDEX
    ) -> tuple[npt.NDArray[np.float32], dict[str, object]]:
        """Observation and info (action masks, spend clamp) for one seat."""
        obs = encode_observation(state, agent_index)
        n_regions = len(state.regions)
        warehouse_cfg = self._config.warehouse
        reset_footprint = _capacity_footprint(state, n_regions, warehouse_cfg)
        info: dict[str, object] = {
            "action_mask": {
                "expansion": compute_expansion_mask(state, agent_index, self._config),
                "automation": compute_automation_mask(state, agent_index, self._config),
            },
            # Same DX keys as ``step``; no tick has run, so no spend has been cut short.
            "spend_clamp": 1.0,
            # No tick has run, so no write-off was taken.
            "writeoff": 0.0,
            "mature_store_equivalents": float(reset_footprint[agent_index]),
            # No tick has run, so no seat has been restructured.
            "restructured": False,
            "insolvent_ticks": int(state.retailers[agent_index].insolvent_ticks),
        }
        return obs, info

    def step(
        self, state: WorldState, joint_action: dict[int, npt.NDArray[np.float32]]
    ) -> StepResult:
        """Run one tick; returns per-seat observations, rewards, and info."""
        # 1. Decode the joint action once, before anything else (pure, consumes no RNG).
        #    Absent retailers fall back to their seated price and the registry defaults.
        decoded_levers = _decode_levers(state, joint_action, self._schema)

        # 2. Capex gates: expansion (new stores) and automation (new tier).
        #    ``last_expansion`` records the choice, even a no-op, for reporting.
        n_regions = len(state.regions)
        capex, next_stores_per_region = gated_expansion_capex(
            state, decoded_levers.expansion, self._config
        )
        last_expansion: list[float] = [
            float(decoded_levers.expansion[idx]) for idx in range(len(state.retailers))
        ]
        automation_capex, new_automation_tier_arr, reported_target_arr = gated_automation_capex(
            state, decoded_levers.automation_tier, self._config
        )
        # Cast back to plain ints: RetailerState fields are declared ``int``.
        new_automation_tier: list[int] = [int(t) for t in new_automation_tier_arr]
        last_automation_tier_action: list[int] = [int(t) for t in reported_target_arr]

        # 3. Affordability clamp (deterministic, no RNG): scale spend to the cash budget.
        #    Gated on ``cash_budget_enabled``; when off, spend_clamp stays all-ones.
        wage_cfg = self._config.wage
        warehouse_cfg = self._config.warehouse
        n_retailers = len(state.retailers)

        # The intended wage is always the policy formula (wage_base + s_t * scale);
        # only the PAID wage (formula x clamp) is seated. Reading the seated wage back
        # as intent would make a clamped wage sticky.
        wage_spend_t = decoded_levers.wage_spend  # [K], s_t
        armed_t = wage_cfg.wage_spend_scale != 0.0
        intended_paid_wage = wage_cfg.wage_base + wage_spend_t * wage_cfg.wage_spend_scale
        # Is the wage line billed at all? None keeps accounting's float-op order untouched.
        wage_line_live = (
            wage_cfg.wage_base != 0.0
            or any(r.wage_level != 0.0 for r in state.retailers)
            or armed_t
        )

        cash_t = np.array([r.cash for r in state.retailers], dtype=np.float64)
        committed_capex = capex + automation_capex
        levers = decoded_levers
        spend_clamp = np.ones(n_retailers, dtype=np.float64)
        intended_spend = np.zeros(n_retailers, dtype=np.float64)
        if wage_cfg.cash_budget_enabled:
            intended_spend = intended_discretionary_spend(
                decoded_levers, intended_paid_wage, self._config
            )
            spend_clamp = compute_spend_clamp(
                cash_t, intended_spend, committed_capex, credit_limit=wage_cfg.credit_limit
            )
            levers = scale_discretionary_levers(decoded_levers, spend_clamp)

        # 4. Resolve demand (draw #1) against the levers decided above (clamped when the
        #    budget gate is on). DemandResult carries the paid lever values.
        demand_result = resolve_demand(
            state,
            joint_action,
            self._rng,
            self._config,
            self._schema,
            levers=levers,
            spend_clamp=spend_clamp,
        )

        # 5. NPC actions for the next tick, in seat order (draw #2).
        # An NPC's price seat is its NEXT price (what rivals perceive), not the one just charged.
        npc_prices: dict[int, float] = {}
        npc_pending: dict[int, tuple[float, ...]] = {}
        for idx, retailer in enumerate(state.retailers):
            if retailer.is_npc:
                policy = self._npc_policies[idx]
                flat = policy.act(state, self._rng, idx)
                # Keep the RAW flat action pending: next tick's decode is the single source of truth.
                npc_prices[idx] = decode_action(flat, self._schema).levers["price_index"]
                npc_pending[idx] = tuple(float(x) for x in flat)

        paid_wage_t: npt.NDArray[np.float64]
        if wage_cfg.cash_budget_enabled:
            paid_wage_t = intended_paid_wage * spend_clamp
        elif armed_t:
            paid_wage_t = intended_paid_wage
        else:
            paid_wage_t = np.array([r.wage_level for r in state.retailers], dtype=np.float64)

        wage_bill_arr: npt.NDArray[np.float64] | None = paid_wage_t if wage_line_live else None
        overhead_arr: npt.NDArray[np.float64] | None = None
        if wage_cfg.overhead_superlinear_coeff != 0.0:
            stores_total_arr = np.array(
                [sum(r.stores_per_region) for r in state.retailers], dtype=np.float64
            )
            excess = np.maximum(0.0, stores_total_arr - wage_cfg.overhead_threshold_stores)
            if wage_cfg.overhead_reference_stores == 1.0:
                overhead_arr = wage_cfg.overhead_superlinear_coeff * (
                    excess**wage_cfg.overhead_exponent
                )
            else:
                overhead_arr = wage_cfg.overhead_superlinear_coeff * (
                    (excess / wage_cfg.overhead_reference_stores) ** wage_cfg.overhead_exponent
                )

        invest_t = levers.warehouse_invest  # [K], v_t
        warehouse_spend_arr: npt.NDArray[np.float64] | None = None
        if warehouse_cfg.cost_per_unit_invest != 0.0 and bool(np.any(invest_t != 0.0)):
            warehouse_spend_arr = invest_t * warehouse_cfg.cost_per_unit_invest

        accounting = apply_accounting(
            state,
            demand_result,
            self._config,
            capex=capex,
            automation_capex=automation_capex,
            wage_bill=wage_bill_arr,
            overhead=overhead_arr,
            warehouse_spend=warehouse_spend_arr,
        )

        next_reward_stats: dict[int, RewardNormalizer] = {}

        # 6. Build the next WorldState: seat cash, per-region awareness and loyalty,
        #    stores, wages, warehouse capacity, and the reporting last_* fields.
        decay = self._config.marketing.decay
        awareness_cap = 1.0 / decay
        market_share = accounting.market_share  # [R, K]
        new_loyalty = accounting.new_loyalty  # [R, K]

        ref_wage = _reference_wages(state)  # [R, K], present-others' mean wage
        ref_wage_k = ref_wage.mean(axis=0)  # [K]
        own_share_k = market_share.mean(axis=0)  # [K]
        required_wage = (
            wage_cfg.wage_base
            + wage_cfg.wage_own_share_coeff * own_share_k
            + wage_cfg.wage_coupling_coeff * ref_wage_k
        )  # [K]
        happiness_decay = wage_cfg.happiness_decay

        seated_capacity = np.array(
            [r.warehouse_capacity for r in state.retailers], dtype=np.float64
        )  # [K], seated (pre-update) capacity — the value demand read this tick
        warehouse_capacity_next = seated_capacity.copy()
        if warehouse_cfg.capacity_per_unit_invest != 0.0:
            invest_mask = invest_t != 0.0
            warehouse_capacity_next[invest_mask] = (
                seated_capacity[invest_mask]
                + invest_t[invest_mask] * warehouse_cfg.capacity_per_unit_invest
            )

        # Utilization, throughput, and the shortfall below must all read the SAME
        # effective capacity. At the provided_capacity_per_store == 0.0 default the
        # result ALIASES seated_capacity — never mutate it in place.
        stores_total_t = _capacity_footprint(state, n_regions, warehouse_cfg)  # [K]
        effective_capacity_t = effective_warehouse_capacity(
            self._config, seated_capacity, stores_total_t
        )  # [K]
        served_t = demand_result.units.sum(axis=0)  # [K]
        finite_positive_capacity = np.isfinite(effective_capacity_t) & (effective_capacity_t > 0.0)
        utilization_t = np.zeros_like(effective_capacity_t)
        utilization_t[finite_positive_capacity] = np.minimum(
            served_t[finite_positive_capacity] / effective_capacity_t[finite_positive_capacity],
            1.0,
        )

        ramp_ticks = warehouse_cfg.ramp_ticks
        next_retailers: list[RetailerState] = []
        for idx, retailer in enumerate(state.retailers):
            # The NPC price seat feeds the perceived obs block (competitor_price) and
            # the tick-0 hold-course price; the pending action wins for charging.
            if retailer.is_npc:
                next_price = npc_prices[idx]
            elif idx in joint_action:
                next_price = float(demand_result.price_indices[idx])
            else:
                next_price = retailer.price_index

            # Report the clamped effective values actually played this tick.
            spend = float(demand_result.spend_fractions[idx])
            acted_assortment = float(demand_result.assortment[idx])
            acted_promotion = float(demand_result.promotion[idx])

            stores_next = next_stores_per_region[idx]

            if ramp_ticks > 0:
                prior_ages = retailer.store_age_per_region
                prior_stores = retailer.stores_per_region
                age_next: list[int] = []
                for r in range(n_regions):
                    prior_age = prior_ages[r] if r < len(prior_ages) else 0
                    was_present = r < len(prior_stores) and prior_stores[r] > 0
                    if stores_next[r] <= 0:
                        age_next.append(prior_age)  # never opened (or not yet)
                    elif was_present:
                        age_next.append(min(prior_age + 1, ramp_ticks))  # ages, capped
                    else:
                        age_next.append(0)  # opened THIS tick — the speed tax starts at 0
                store_age_next: tuple[int, ...] = tuple(age_next)
            else:
                store_age_next = retailer.store_age_per_region

            awareness_next: list[float] = []
            for r in range(n_regions):
                prior = (
                    retailer.awareness_per_region[r]
                    if r < len(retailer.awareness_per_region)
                    else 0.0
                )
                build = spend if stores_next[r] > 0 else 0.0
                awareness_next.append(min(max((1.0 - decay) * prior + build, 0.0), awareness_cap))

            # Per-region loyalty advances from this tick's share; demand reads it next tick.
            loyalty_next = tuple(float(new_loyalty[r, idx]) for r in range(n_regions))
            share_next = tuple(float(market_share[r, idx]) for r in range(n_regions))

            # service_score_next[r] = clip(
            #     (1 - decay) * prev[r]
            #       + decay * (1 - lost_sales_share_penalty * stockout_rate[r, k]),
            #     0.0, 1.0
            # )
            # The EMA falls with stockouts and recovers toward 1.0 when they stop.
            scm_cfg = self._config.scm
            service_decay = scm_cfg.service_score_decay
            lost_penalty = scm_cfg.lost_sales_share_penalty
            service_score_next: list[float] = []
            for r in range(n_regions):
                prior = (
                    retailer.service_score_per_region[r]
                    if r < len(retailer.service_score_per_region)
                    else INITIAL_SERVICE_SCORE
                )
                sr = float(demand_result.stockout_rate[r, idx])
                ema = (1.0 - service_decay) * prior + service_decay * (1.0 - lost_penalty * sr)
                service_score_next.append(min(max(ema, 0.0), 1.0))

            paid_wage = float(paid_wage_t[idx])
            shortfall = paid_wage - float(required_wage[idx])
            if wage_cfg.wage_shortfall_scale != 1.0:
                shortfall = shortfall / wage_cfg.wage_shortfall_scale
            happiness_target = min(max(1.0 + shortfall, 0.0), 1.0)
            happiness_next = min(
                max(
                    (1.0 - happiness_decay) * retailer.employee_happiness
                    + happiness_decay * happiness_target,
                    0.0,
                ),
                1.0,
            )

            next_retailers.append(
                RetailerState(
                    name=retailer.name,
                    is_npc=retailer.is_npc,
                    cash=float(accounting.new_cash[idx]),
                    price_index=next_price,
                    stores_per_region=stores_next,
                    store_age_per_region=store_age_next,
                    awareness_per_region=tuple(awareness_next),
                    loyalty_per_region=loyalty_next,
                    last_profit=float(accounting.profit[idx]),
                    last_revenue=float(accounting.revenue[idx]),
                    # Aggregate (region-0) reporting share for the legacy obs slot.
                    last_market_share=float(market_share[0, idx]),
                    last_market_share_per_region=share_next,
                    last_assortment=acted_assortment,
                    last_promotion=acted_promotion,
                    # The expansion choice acted this tick, surfaced in next-tick obs.
                    last_expansion=last_expansion[idx],
                    # Seated per-region service score (the EMA above).
                    service_score_per_region=tuple(service_score_next),
                    # Seated automation tier: demand reads it next tick for the COGS
                    # multiplier. last_automation_tier_action reports the decoded target.
                    automation_tier=new_automation_tier[idx],
                    last_automation_tier_action=last_automation_tier_action[idx],
                    wage_level=paid_wage,
                    employee_happiness=happiness_next,
                    accumulated_overhead_basis=float(accounting.overhead[idx]),
                    last_turnover=float(accounting.revenue[idx]),
                    warehouse_capacity=float(warehouse_capacity_next[idx]),
                    last_warehouse_utilization=float(utilization_t[idx]),
                    last_stockout_rate=float(demand_result.lost_sales_fraction[idx]),
                    # Carry the restructuring counters forward; only the gate below advances them.
                    insolvent_ticks=retailer.insolvent_ticks,
                    restructurings=retailer.restructurings,
                    pending_action=npc_pending.get(idx),
                )
            )

        decay_s = self._config.promotion.stockpile_decay
        build_s = self._config.promotion.stockpile_build
        cap_s = 1.0 / decay_s
        promotion = demand_result.promotion  # [K]
        shares = demand_result.shares  # [R, K]
        next_regions = []
        for r, region in enumerate(state.regions):
            promo_agg_r = float(np.dot(promotion, shares[r]))
            next_regions.append(
                RegionState(
                    population=region.population,
                    base_regional_demand=region.base_regional_demand,
                    segment_mix=dict(region.segment_mix),
                    realized_regional_demand=float(demand_result.total_demand[r]),
                    stockpile=min(
                        max((1.0 - decay_s) * region.stockpile + build_s * promo_agg_r, 0.0),
                        cap_s,
                    ),
                )
            )

        next_state = WorldState(
            tick=state.tick + 1,
            regions=next_regions,
            retailers=next_retailers,
            segments=dict(state.segments),
            # Seat the advanced reward normalizers (empty in profit mode).
            reward_stats=next_reward_stats,
        )

        # 7. Perception noise (draw #4, last): noised perceived competitor views and
        #    research fidelity. One constant-size draw, independent of fidelity.
        research_by_seat = {
            idx: float(demand_result.research[idx]) for idx in range(len(state.retailers))
        }
        _seat_perceived_views(next_state, research_by_seat, self._config, self._rng)

        writeoff_by_seat: dict[int, float] = {}
        if wage_cfg.writeoff_gate_active:
            n_ret = len(state.retailers)
            draw = self._rng.uniform(
                wage_cfg.writeoff_frac_min, wage_cfg.writeoff_frac_max, size=n_ret
            )
            for idx in range(n_ret):
                seated = next_state.retailers[idx]
                severity = wage_cfg.writeoff_severity_scale * (1.0 - seated.employee_happiness)
                prior_turnover = state.retailers[idx].last_turnover
                size = float(sum(seated.stores_per_region))
                leader_factor = size ** (wage_cfg.leader_weight_exponent - 1.0)
                writeoff = float(draw[idx]) * severity * prior_turnover * leader_factor
                if writeoff != 0.0:
                    seated.last_profit -= writeoff
                    seated.cash -= writeoff
                    writeoff_by_seat[idx] = writeoff

        penalty_by_seat: dict[int, float] = {}
        if warehouse_cfg.penalty_gate_active:
            n_ret = len(state.retailers)
            # One fixed-shape [K] draw for ALL seats: keeps the RNG cursor advance deterministic.
            draw = self._rng.uniform(
                warehouse_cfg.penalty_frac_min, warehouse_cfg.penalty_frac_max, size=n_ret
            )
            # The shortfall reads the SAME effective capacity the utilization readout uses.
            seated_capacity_next = np.array(
                [r.warehouse_capacity for r in next_state.retailers], dtype=np.float64
            )  # [K]
            stores_total_next = _capacity_footprint(next_state, n_regions, warehouse_cfg)  # [K]
            effective_next = effective_warehouse_capacity(
                self._config, seated_capacity_next, stores_total_next
            )  # [K]
            for idx in range(n_ret):
                seated = next_state.retailers[idx]
                # The required side uses the unbounded store sum (growth-overhead convention).
                stores_total = float(sum(seated.stores_per_region))
                required_capacity = warehouse_cfg.capacity_per_store * stores_total
                shortfall = max(0.0, required_capacity - float(effective_next[idx]))
                severity = warehouse_cfg.penalty_severity_scale * (
                    shortfall / (shortfall + warehouse_cfg.penalty_halfsat)
                )
                prior_turnover = state.retailers[idx].last_turnover
                penalty = float(draw[idx]) * severity * prior_turnover
                if penalty != 0.0:
                    seated.last_profit -= penalty
                    seated.cash -= penalty
                    penalty_by_seat[idx] = penalty

        restructured_by_seat: dict[int, bool] = {}
        if wage_cfg.restructuring_gate_active:
            debt_line = wage_cfg.credit_limit + wage_cfg.restructure_debt_threshold
            keep_stores = wage_cfg.restructure_keep_stores
            for idx in range(len(next_state.retailers)):
                seated = next_state.retailers[idx]
                if seated.cash < -debt_line:
                    seated.insolvent_ticks += 1
                else:
                    seated.insolvent_ticks = 0
                if seated.insolvent_ticks < wage_cfg.restructure_after_ticks:
                    continue
                # Region 0 is never closed; only r >= 1 counts against keep_stores.
                present_count = sum(
                    1 for r in range(1, n_regions) if seated.stores_per_region[r] > 0
                )
                if present_count > keep_stores:
                    kept = _restructuring_survivor_regions(
                        seated.stores_per_region,
                        seated.last_market_share_per_region,
                        n_regions,
                        keep_stores,
                    )
                    new_stores = list(seated.stores_per_region)
                    new_ages = list(seated.store_age_per_region)
                    for r in range(1, n_regions):
                        if new_stores[r] > 0 and r not in kept:
                            new_stores[r] = 0
                            new_ages[r] = 0  # no store history yet
                    seated.stores_per_region = tuple(new_stores)
                    seated.store_age_per_region = tuple(new_ages)
                seated.cash = 0.0
                seated.insolvent_ticks = 0
                seated.restructurings += 1
                restructured_by_seat[idx] = True

        reward_cfg = self._config.reward
        rewards: dict[int, float] = {}
        observations: dict[int, npt.NDArray[np.float32]] = {}
        infos: dict[int, dict[str, object]] = {}
        terminated: dict[int, bool] = {}
        truncated: dict[int, bool] = {}
        mature_store_equivalents = _capacity_footprint(next_state, n_regions, warehouse_cfg)
        for idx in range(len(next_state.retailers)):
            observations[idx] = encode_observation(next_state, idx)

            components = reward_components(
                accounting,
                demand_result,
                idx,
                writeoff=writeoff_by_seat.get(idx, 0.0),
                penalty=penalty_by_seat.get(idx, 0.0),
            )

            # Weighted reward only for learning seats (those with a normalizer);
            # all other seats take the post-penalty profit directly.
            has_normalizer = reward_cfg.mode == "weighted" and idx in state.reward_stats
            if has_normalizer:
                # Copy so the fold does not mutate the input state's stats.
                normalizer = RewardNormalizer.from_triples(state.reward_stats[idx].to_triples())
                rewards[idx] = compute_reward_for_seat(components, reward_cfg, normalizer, seat=idx)
                next_reward_stats[idx] = normalizer
            else:
                rewards[idx] = float(components.profit)

            infos[idx] = {
                "components": components.as_dict(),
                # The affordability clamp this seat's spend was scaled by (1.0 when the gate is off).
                "spend_clamp": float(demand_result.spend_clamp[idx]),
                # This seat's write-off this tick (0.0 when the gate is inert).
                "writeoff": float(writeoff_by_seat.get(idx, 0.0)),
                "mature_store_equivalents": float(mature_store_equivalents[idx]),
                # ``restructured`` is True only on the tick this seat was restructured.
                # The reward is untouched: the write-off is a balance-sheet event, not income.
                "restructured": bool(restructured_by_seat.get(idx, False)),
                "insolvent_ticks": int(next_state.retailers[idx].insolvent_ticks),
                "action_mask": {
                    "expansion": compute_expansion_mask(next_state, idx, self._config),
                    "automation": compute_automation_mask(next_state, idx, self._config),
                },
            }
            terminated[idx] = False
            truncated[idx] = False

        return StepResult(
            next_state=next_state,
            observations=observations,
            rewards=rewards,
            infos=infos,
            terminated=terminated,
            truncated=truncated,
        )
