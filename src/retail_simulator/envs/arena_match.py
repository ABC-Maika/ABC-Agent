"""One unbroken arena match, steppable round by round — the single seating/masking seam.

Used by ``scripts/arena_competition.py``, ``scripts/gen_leaderboard_artifact.py`` and the live demo. Seating, per-seat expansion masking, and the "log the EFFECTIVE decision" rule each have ONE owner here, rather than three copies that drift apart.

Seats play in the order given; exactly the ``kind="rl"`` seats are expansion-masked (archetypes stay unmasked — their own illegal picks are part of the posture). The env is reset ONCE in the constructor; ``episode_start`` is True only for round 1, so a recurrent policy's state carries across the whole match.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, TypeVar, cast

import numpy as np
import numpy.typing as npt

from retail_simulator.core.config import CoreConfig
from retail_simulator.core.encoding import decode_action
from retail_simulator.core.state import WorldState
from retail_simulator.core.world import compute_expansion_mask
from retail_simulator.envs.arena import (
    ARENA_BALANCED_ENV_LABEL,
    ARENA_ENV_LABEL,
    arena_balanced_config,
    arena_config,
    arena_home_regions,
    arena_lock_config,
)
from retail_simulator.envs.geo_archetypes import GEO_ARCHETYPE_LABELS, geo_archetypes
from retail_simulator.envs.gym_env import mask_expansion_logits
from retail_simulator.envs.ladder import _materialize_predicts
from retail_simulator.envs.parallel_env import RetailParallelEnv

# The levers every consumer records, in the order the study's CSV writes them;
# re-exported by scripts/arena_competition.py so its CLI contract is unchanged.
LOGGED_LEVERS: tuple[str, ...] = (
    "price_index",
    "marketing",
    "assortment",
    "promotion",
    "service_level",
    "loyalty_spend",
    "research",
    "wage_spend",
    "warehouse_invest",
    "expansion",
    "automation",
)

MIN_SEATS: int = 2


@dataclass(frozen=True)
class ArenaSeatSpec:
    """One seat: who sits there and how they act.

    ``kind="rl"`` seats carry a ``checkpoint`` and run with expansion masking on; ``kind="npc"`` seats carry an ``archetype`` label and run unmasked. ``name`` is unique within a match (standings are keyed by it); ``player`` is a display label only.
    """

    name: str
    kind: Literal["rl", "npc"]
    checkpoint: str | None = None
    archetype: str | None = None
    player: str | None = None


@dataclass(frozen=True)
class SeatRound:
    """What one seat did, and got, in one round.

    ``levers`` is the EFFECTIVE decision (post-mask for a masked seat), restricted to :data:`LOGGED_LEVERS`; ``opened_regions`` are the regions whose store count went 0 -> non-zero this round (a retailer can only open a region it has no store in, so ``len(opened_regions)`` is this round's store growth).
    """

    reward: float
    revenue: float
    profit: float
    market_share: float
    cash: float
    employee_happiness: float
    stores: int
    opened_regions: tuple[int, ...]
    levers: dict[str, float]


@dataclass(frozen=True)
class RoundResult:
    """The whole field after one round; ``round`` is 1-based (the round just completed).

    ``region_share[r][k]`` / ``region_stores[r][k]`` are region-major so a map renderer can read one region's split without transposing.
    """

    round: int
    seats: tuple[SeatRound, ...]
    region_share: tuple[tuple[float, ...], ...]
    region_stores: tuple[tuple[int, ...], ...]


_SectionT = TypeVar("_SectionT")


def _apply(section: _SectionT, overrides: Mapping[str, float]) -> _SectionT:
    """``dataclasses.replace`` over a runtime-keyed override map, returning the same type.

    Override field names come from CLI flags (not statically known); the cast keeps that to one line.
    """
    return cast("_SectionT", replace(cast("Any", section), **overrides))


def resolve_arena_world(
    *,
    balanced: bool = False,
    overhead: float | None = None,
    provided_capacity_per_store: float | None = None,
    capacity_per_store: float | None = None,
    penalty_severity_scale: float | None = None,
    penalty_frac_max: float | None = None,
    ramp_ticks: int | None = None,
    overdraft_rate: float | None = None,
    writeoff_frac_max: float | None = None,
    wage_own_share_coeff: float | None = None,
    overhead_reference_stores: float | None = None,
    restructure_after_ticks: int | None = None,
    restructure_keep_stores: int | None = None,
    max_share_per_region: float | None = None,
    marketing_cost_exponent: float | None = None,
    promotion_contest_weight: float | None = None,
    promo_cost_per_unit: float | None = None,
    stockpile_build: float | None = None,
    stockpile_decay: float | None = None,
) -> tuple[CoreConfig, str, dict[str, float]]:
    """Resolve an arena world from knob overrides; return (config, label, the effective overrides)."""
    if overhead is not None:
        config = arena_config(
            overhead_reference_stores=overhead,
            overhead_exponent=1.5,
            provided_capacity_per_store=700.0,
        )
        label = f"arena_config(overhead_reference_stores={overhead})"
    elif balanced:
        config, label = arena_balanced_config(), ARENA_BALANCED_ENV_LABEL
    else:
        config, label = arena_lock_config(), ARENA_ENV_LABEL

    warehouse_overrides: dict[str, float | int] = {
        field: value
        for field, value in (
            ("provided_capacity_per_store", provided_capacity_per_store),
            ("capacity_per_store", capacity_per_store),
            ("penalty_severity_scale", penalty_severity_scale),
            ("penalty_frac_max", penalty_frac_max),
            ("ramp_ticks", ramp_ticks),
        )
        if value is not None
    }
    # restructure_after_ticks/restructure_keep_stores are genuinely int (WageConfig's own
    # fields), unlike every other override here (float).
    wage_overrides: dict[str, float | int] = {
        field: value
        for field, value in (
            ("overdraft_rate", overdraft_rate),
            ("writeoff_frac_max", writeoff_frac_max),
            ("wage_own_share_coeff", wage_own_share_coeff),
            ("overhead_reference_stores", overhead_reference_stores),
            ("restructure_after_ticks", restructure_after_ticks),
            ("restructure_keep_stores", restructure_keep_stores),
        )
        if value is not None
    }
    # Regional-share-ceiling override dict (None keeps the chosen world's own value).
    demand_overrides: dict[str, float] = {
        field: value
        for field, value in (("max_share_per_region", max_share_per_region),)
        if value is not None
    }
    marketing_overrides: dict[str, float] = {
        field: value
        for field, value in (("marketing_cost_exponent", marketing_cost_exponent),)
        if value is not None
    }
    promotion_overrides: dict[str, float] = {
        field: value
        for field, value in (
            ("promotion_contest_weight", promotion_contest_weight),
            ("promo_cost_per_unit", promo_cost_per_unit),
            ("stockpile_build", stockpile_build),
            ("stockpile_decay", stockpile_decay),
        )
        if value is not None
    }
    if warehouse_overrides:
        config = replace(config, warehouse=_apply(config.warehouse, warehouse_overrides))
    if wage_overrides:
        config = replace(config, wage=_apply(config.wage, wage_overrides))
    if demand_overrides:
        config = replace(config, demand=_apply(config.demand, demand_overrides))
    if marketing_overrides:
        config = replace(config, marketing=_apply(config.marketing, marketing_overrides))
    if promotion_overrides:
        config = replace(config, promotion=_apply(config.promotion, promotion_overrides))
    return (
        config,
        label,
        {
            **warehouse_overrides,
            **wage_overrides,
            **demand_overrides,
            **marketing_overrides,
            **promotion_overrides,
        },
    )


def _validate_seats(seats: Sequence[ArenaSeatSpec]) -> None:
    """Reject a seat list that cannot produce a well-formed match (boundary validation)."""
    if len(seats) < MIN_SEATS:
        raise ValueError(f"a match needs at least {MIN_SEATS} seats, got {len(seats)}")
    names = [seat.name for seat in seats]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        # Seat names key the standings/wire schema, so duplicates would silently merge.
        raise ValueError(f"seat names must be unique, but {duplicated} appear more than once")
    for index, seat in enumerate(seats):
        if seat.kind == "rl":
            if not seat.checkpoint:
                raise ValueError(f"seat {index} ({seat.name!r}) is rl but carries no checkpoint")
        elif seat.kind == "npc":
            if seat.archetype not in GEO_ARCHETYPE_LABELS:
                raise ValueError(
                    f"seat {index} ({seat.name!r}) has unknown archetype {seat.archetype!r}; "
                    f"choose from {list(GEO_ARCHETYPE_LABELS)}"
                )
        else:
            raise ValueError(f"seat {index} ({seat.name!r}) has unknown kind {seat.kind!r}")


class ArenaMatch:
    """One unbroken arena match over a continuing world, advanced one round per ``step()``.

    Construction seats the field, materializes every policy and resets the world with ``seed``; nothing is lazy afterwards. ``homes`` defaults to :func:`~retail_simulator.envs.arena.arena_home_regions` (one distinct METRO home per seat).
    """

    def __init__(
        self,
        config: CoreConfig,
        seats: Sequence[ArenaSeatSpec],
        *,
        seed: int,
        homes: Sequence[int] | None = None,
    ) -> None:
        seat_specs = tuple(seats)
        _validate_seats(seat_specs)
        n_seats = len(seat_specs)
        if homes is None:
            resolved_homes = arena_home_regions(n_seats, n_regions=len(config.demand.regions))
        else:
            resolved_homes = tuple(int(region) for region in homes)
            if len(resolved_homes) != n_seats:
                raise ValueError(f"homes names {len(resolved_homes)} regions for {n_seats} seats")

        self._config = config
        self._seat_specs = seat_specs
        self._homes = resolved_homes
        self._masked = tuple(index for index, seat in enumerate(seat_specs) if seat.kind == "rl")

        self._env = RetailParallelEnv(
            config,
            n_learning_agents=n_seats,
            seed=seed,
            home_regions=resolved_homes,
            mask_expansion_seats=self._masked,
        )
        space = self._env.action_space(self._env.possible_agents[0])

        # Checkpoints load ONCE here (_materialize_predicts owns the lazy SB3 import).
        checkpoints = {seat.name: str(seat.checkpoint) for seat in seat_specs if seat.kind == "rl"}
        predicts = _materialize_predicts(checkpoints, action_space=space, config=config)
        region_table = config.demand.regions
        roster = geo_archetypes(len(region_table), region_table) if self._has_npc() else {}
        self._policies: list[Callable[..., Any]] = [
            predicts[seat.name] if seat.kind == "rl" else roster[str(seat.archetype)]
            for seat in seat_specs
        ]

        self._obs, _ = self._env.reset(seed=seed)
        self._round = 0

    # -- introspection --

    @property
    def homes(self) -> tuple[int, ...]:
        """The home region seated per seat, in seat order."""
        return self._homes

    @property
    def n_regions(self) -> int:
        """The world's region count (the region table's own length, the codebase's SSOT)."""
        return len(self._config.demand.regions)

    @property
    def seat_specs(self) -> tuple[ArenaSeatSpec, ...]:
        """The seats as given, in env seat order."""
        return self._seat_specs

    @property
    def masked_seats(self) -> tuple[int, ...]:
        """The seat indices playing with expansion masking on (exactly the rl seats)."""
        return self._masked

    @property
    def rounds_played(self) -> int:
        """How many rounds have been stepped so far."""
        return self._round

    # -- the match --

    def step(self) -> RoundResult:
        """Play ONE more round of the same continuing match and report what happened."""
        agents = self._env.possible_agents
        episode_start = self._round == 0
        raw = {
            agent: np.asarray(
                self._policies[seat](self._obs[agent], episode_start=episode_start),
                dtype=np.float32,
            ).ravel()
            for seat, agent in enumerate(agents)
        }
        levers = [self._effective_levers(seat, raw[agent]) for seat, agent in enumerate(agents)]
        stores_before = [retailer.stores_per_region for retailer in self._state().retailers]

        self._obs, rewards, _, _, infos = self._env.step(raw)
        self._round += 1

        state = self._state()
        seat_rounds = tuple(
            self._seat_round(seat, agent, rewards, infos, state, stores_before[seat], levers[seat])
            for seat, agent in enumerate(agents)
        )
        n_seats = len(agents)
        region_share = tuple(
            tuple(
                float(state.retailers[seat].last_market_share_per_region[region])
                for seat in range(n_seats)
            )
            for region in range(self.n_regions)
        )
        region_stores = tuple(
            tuple(int(state.retailers[seat].stores_per_region[region]) for seat in range(n_seats))
            for region in range(self.n_regions)
        )
        return RoundResult(
            round=self._round,
            seats=seat_rounds,
            region_share=region_share,
            region_stores=region_stores,
        )

    # -- internals --

    def _has_npc(self) -> bool:
        return any(seat.kind == "npc" for seat in self._seat_specs)

    def _state(self) -> WorldState:
        """The live world state — this seam's own subject, so the private read is deliberate."""
        state = self._env._state
        if state is None:  # pragma: no cover — the constructor resets the env
            raise RuntimeError("the arena env has no world state; it was never reset")
        return state

    def _effective_levers(self, seat: int, action: npt.NDArray[np.float32]) -> dict[str, float]:
        """Decode the decision the world will actually execute for this seat.

        A masked seat's action is mirrored through the env's OWN masking helper with the PRE-step mask (the mask available at decision time) — the post-step mask would report a successful open as illegal.
        """
        if seat in self._masked:
            action = mask_expansion_logits(
                action, compute_expansion_mask(self._state(), seat, self._config)
            )
        decoded = decode_action(action).levers
        return {lever: float(decoded[lever]) for lever in LOGGED_LEVERS if lever in decoded}

    @staticmethod
    def _seat_round(
        seat: int,
        agent: str,
        rewards: dict[str, Any],
        infos: dict[str, Any],
        state: Any,
        stores_before: tuple[int, ...],
        levers: dict[str, float],
    ) -> SeatRound:
        components = infos[agent]["components"]
        retailer = state.retailers[seat]
        opened = tuple(
            region
            for region, count in enumerate(retailer.stores_per_region)
            if count > 0 and stores_before[region] == 0
        )
        return SeatRound(
            reward=float(rewards[agent]),
            revenue=float(components["revenue"]),
            profit=float(components["profit"]),
            market_share=float(components["market_share"]),
            cash=float(retailer.cash),
            employee_happiness=float(retailer.employee_happiness),
            stores=int(np.sum(retailer.stores_per_region)),
            opened_regions=opened,
            levers=levers,
        )
