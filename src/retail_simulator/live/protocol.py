"""Wire protocol: JSON-over-WebSocket message dataclasses.

The single source of truth for the live wrapper's wire format; server, client
and replay tool all import their message types from here. Pure stdlib + numpy
+ the core schema — no asyncio, no ``websockets`` (those are lazy-imported
inside ``server.py`` / ``client.py``).

Wire format: one JSON object per WebSocket text frame, UTF-8. Each message
type is a frozen dataclass with ``to_json`` / ``from_json`` round-trippers;
the top-level ``kind`` field is the discriminator (:func:`from_json_any`
dispatches, raising ``ValueError`` at the boundary on malformed input).
Action / observation arrays cross as Python lists, not numpy arrays (the
caller converts to / from ``np.ndarray``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Self

import numpy as np
import numpy.typing as npt

PROTOCOL_VERSION: int = 2


def _build_registry_default_action() -> npt.NDArray[np.float32]:
    """Build the registry-default action vector (every enabled lever at its default) as flat float32."""
    # Imported inside the helper so this module's top-level imports stay stdlib + numpy only.
    from retail_simulator.core.encoding import StructuredAction, encode_action
    from retail_simulator.core.schema import LEVER_REGISTRY, default_action_schema

    structured = StructuredAction(
        levers={spec.name: spec.default for spec in LEVER_REGISTRY if spec.enabled}
    )
    return encode_action(structured, default_action_schema())


REGISTRY_DEFAULT_ACTION: npt.NDArray[np.float32] = _build_registry_default_action()
# Read-only so a caller cannot mutate the singleton and corrupt every future tick-0 fallback.
REGISTRY_DEFAULT_ACTION.flags.writeable = False


def _payload_without_kind(obj: Any) -> dict[str, Any]:
    """``dataclasses.asdict`` over a frozen message (ClassVar fields are excluded automatically)."""
    payload: dict[str, Any] = asdict(obj)
    return payload


def _require(payload: dict[str, Any], key: str, expected_type: type) -> Any:
    """Validate a payload key exists and has the expected type; raise ``ValueError`` with the key name."""
    if key not in payload:
        raise ValueError(f"missing required field {key!r}")
    value = payload[key]
    if not isinstance(value, expected_type):
        raise ValueError(
            f"field {key!r} has type {type(value).__name__}, expected {expected_type.__name__}"
        )
    return value


def _check_kind(payload: dict[str, Any], expected: str) -> None:
    """Pop the ``kind`` discriminator and verify it matches; raise with both values on mismatch."""
    got = payload.pop("kind", None)
    if got != expected:
        raise ValueError(f"kind mismatch: expected {expected!r}, got {got!r}")


@dataclass(frozen=True)
class Hello:
    """Client to server: registration hello (name, token, protocol/schema versions)."""

    agent_name: str
    token: str | None
    protocol_version: int
    schema_version: int
    client_name: str
    kind: ClassVar[str] = "hello"

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, **_payload_without_kind(self)})

    @classmethod
    def from_json(cls, s: str) -> Self:
        payload = json.loads(s)
        _check_kind(payload, cls.kind)
        agent_name = _require(payload, "agent_name", str)
        protocol_version = _require(payload, "protocol_version", int)
        schema_version = _require(payload, "schema_version", int)
        client_name = _require(payload, "client_name", str)
        # ``token`` is nullable; ``_require`` would reject ``None``.
        if "token" not in payload:
            raise ValueError("missing required field 'token'")
        token = payload["token"]
        if token is not None and not isinstance(token, str):
            raise ValueError(f"field 'token' has type {type(token).__name__}, expected str or None")
        return cls(
            agent_name=agent_name,
            token=token,
            protocol_version=protocol_version,
            schema_version=schema_version,
            client_name=client_name,
        )


@dataclass(frozen=True)
class SeatAssigned:
    """Server to client: confirms the seat the client occupies.

    ``npc_archetypes``: scripted-NPC names on seats ``[n_agents, n_agents +
    n_npcs)``. ``agent_id`` is the env's seat identifier (e.g. ``"retailer_0"``).
    ``tick_interval_ns`` is the wall-clock tick interval; the per-tick truth is
    ``deadline_ns`` on each :class:`ObservationMessage`. ``scenario_name`` lets
    the client reconstruct its own replay.
    """

    seat_index: int
    n_agents: int
    n_npcs: int
    npc_archetypes: tuple[str, ...]
    n_ticks: int
    game_id: str
    agent_id: str
    tick_interval_ns: int
    scenario_name: str
    kind: ClassVar[str] = "seat-assigned"

    def to_json(self) -> str:
        # Tuples are not JSON-native; ``json.dumps`` emits lists and ``from_json`` rebuilds the tuple.
        return json.dumps({"kind": self.kind, **_payload_without_kind(self)})

    @classmethod
    def from_json(cls, s: str) -> Self:
        payload = json.loads(s)
        _check_kind(payload, cls.kind)
        seat_index = _require(payload, "seat_index", int)
        n_agents = _require(payload, "n_agents", int)
        n_npcs = _require(payload, "n_npcs", int)
        archetypes_raw = _require(payload, "npc_archetypes", list)
        if not all(isinstance(name, str) for name in archetypes_raw):
            raise ValueError("field 'npc_archetypes' must be a list of strings")
        n_ticks = _require(payload, "n_ticks", int)
        game_id = _require(payload, "game_id", str)
        agent_id = _require(payload, "agent_id", str)
        tick_interval_ns = _require(payload, "tick_interval_ns", int)
        scenario_name = _require(payload, "scenario_name", str)
        return cls(
            seat_index=seat_index,
            n_agents=n_agents,
            n_npcs=n_npcs,
            npc_archetypes=tuple(archetypes_raw),
            n_ticks=n_ticks,
            game_id=game_id,
            agent_id=agent_id,
            tick_interval_ns=tick_interval_ns,
            scenario_name=scenario_name,
        )


@dataclass(frozen=True)
class ObservationMessage:
    """Server to client: per-tick observation broadcast.

    ``observation`` is the flat obs vector as a Python list of floats (numpy
    conversion is the caller's job). ``info`` is the per-tick info dict. ``seat_index``
    is the explicit seat carried in the message. ``action_mask`` is the
    per-action-dim valid mask (None when masking is disabled). ``deadline_ns`` is
    the absolute monotonic-clock deadline before the server's timeout-fallback
    fires. ``components`` is the per-component reward breakdown for diagnostics.
    """

    tick: int
    observation: list[float]
    reward: float
    info: dict[str, Any]
    seat_index: int
    action_mask: list[float] | None
    deadline_ns: int
    components: dict[str, float]
    kind: ClassVar[str] = "observation"

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, **_payload_without_kind(self)})

    @classmethod
    def from_json(cls, s: str) -> Self:
        payload = json.loads(s)
        _check_kind(payload, cls.kind)
        tick = _require(payload, "tick", int)
        observation_raw = _require(payload, "observation", list)
        # Accept ints as well as floats (JSON has no int/float distinction); cast at the boundary.
        if not all(isinstance(x, (int, float)) for x in observation_raw):
            raise ValueError("field 'observation' must be a list of numbers")
        observation = [float(x) for x in observation_raw]
        # JSON parses ``42`` as int and ``42.0`` as float; accept both then cast.
        reward_raw = payload.get("reward")
        if not isinstance(reward_raw, (int, float)):
            raise ValueError(
                f"field 'reward' has type {type(reward_raw).__name__}, expected number"
            )
        info = _require(payload, "info", dict)
        seat_index = _require(payload, "seat_index", int)
        # ``action_mask`` is optional / nullable; allow ``None`` explicitly.
        if "action_mask" not in payload:
            raise ValueError("missing required field 'action_mask'")
        action_mask_raw = payload["action_mask"]
        if action_mask_raw is None:
            action_mask: list[float] | None = None
        else:
            if not isinstance(action_mask_raw, list):
                raise ValueError(
                    f"field 'action_mask' has type {type(action_mask_raw).__name__}, "
                    "expected list or None"
                )
            if not all(isinstance(x, (int, float)) for x in action_mask_raw):
                raise ValueError("field 'action_mask' must be a list of numbers or None")
            action_mask = [float(x) for x in action_mask_raw]
        deadline_ns = _require(payload, "deadline_ns", int)
        components_raw = _require(payload, "components", dict)
        for k, v in components_raw.items():
            if not isinstance(k, str):
                raise ValueError("field 'components' keys must be strings")
            if not isinstance(v, (int, float)):
                raise ValueError("field 'components' values must be numbers")
        components = {k: float(v) for k, v in components_raw.items()}
        return cls(
            tick=tick,
            observation=observation,
            reward=float(reward_raw),
            info=info,
            seat_index=seat_index,
            action_mask=action_mask,
            deadline_ns=deadline_ns,
            components=components,
        )


@dataclass(frozen=True)
class ActionMessage:
    """Client to server: the submitted action for a tick."""

    tick: int
    action: list[float]
    kind: ClassVar[str] = "action"

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, **_payload_without_kind(self)})

    @classmethod
    def from_json(cls, s: str) -> Self:
        payload = json.loads(s)
        _check_kind(payload, cls.kind)
        tick = _require(payload, "tick", int)
        action_raw = _require(payload, "action", list)
        if not all(isinstance(x, (int, float)) for x in action_raw):
            raise ValueError("field 'action' must be a list of numbers")
        return cls(tick=tick, action=[float(x) for x in action_raw])


@dataclass(frozen=True)
class FinalScores:
    """Server to client: end-of-game scores and per-seat reward components."""

    final_scores: dict[str, float]
    winner_seat: int | None
    per_seat_components: dict[str, dict[str, float]]
    total_ticks: int
    missed_deadlines_per_seat: dict[str, int]
    wall_clock_variance: float
    kind: ClassVar[str] = "final-scores"

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, **_payload_without_kind(self)})

    @classmethod
    def from_json(cls, s: str) -> Self:
        payload = json.loads(s)
        _check_kind(payload, cls.kind)
        scores_raw = _require(payload, "final_scores", dict)
        for k, v in scores_raw.items():
            if not isinstance(k, str):
                raise ValueError("field 'final_scores' keys must be strings")
            if not isinstance(v, (int, float)):
                raise ValueError("field 'final_scores' values must be numbers")
        final_scores = {k: float(v) for k, v in scores_raw.items()}
        if "winner_seat" not in payload:
            raise ValueError("missing required field 'winner_seat'")
        winner_seat = payload["winner_seat"]
        if winner_seat is not None and not isinstance(winner_seat, int):
            raise ValueError(
                f"field 'winner_seat' has type {type(winner_seat).__name__}, expected int or None"
            )
        components_raw = _require(payload, "per_seat_components", dict)
        per_seat_components: dict[str, dict[str, float]] = {}
        for seat_key, comp_map in components_raw.items():
            if not isinstance(seat_key, str):
                raise ValueError("field 'per_seat_components' keys must be strings")
            if not isinstance(comp_map, dict):
                raise ValueError(
                    "field 'per_seat_components' values must be dicts of component name to number"
                )
            seat_components: dict[str, float] = {}
            for comp_key, comp_value in comp_map.items():
                if not isinstance(comp_key, str):
                    raise ValueError("field 'per_seat_components' inner keys must be strings")
                if not isinstance(comp_value, (int, float)):
                    raise ValueError("field 'per_seat_components' inner values must be numbers")
                seat_components[comp_key] = float(comp_value)
            per_seat_components[seat_key] = seat_components
        total_ticks = _require(payload, "total_ticks", int)
        missed_raw = _require(payload, "missed_deadlines_per_seat", dict)
        for k, v in missed_raw.items():
            if not isinstance(k, str):
                raise ValueError("field 'missed_deadlines_per_seat' keys must be strings")
            if not isinstance(v, int) or isinstance(v, bool):
                # ``bool`` is a subclass of ``int``; reject it explicitly so True does not coerce to 1.
                raise ValueError("field 'missed_deadlines_per_seat' values must be ints")
        missed_deadlines_per_seat = {k: int(v) for k, v in missed_raw.items()}
        variance_raw = payload.get("wall_clock_variance")
        if not isinstance(variance_raw, (int, float)):
            raise ValueError(
                f"field 'wall_clock_variance' has type "
                f"{type(variance_raw).__name__}, expected number"
            )
        return cls(
            final_scores=final_scores,
            winner_seat=winner_seat,
            per_seat_components=per_seat_components,
            total_ticks=total_ticks,
            missed_deadlines_per_seat=missed_deadlines_per_seat,
            wall_clock_variance=float(variance_raw),
        )


@dataclass(frozen=True)
class ErrorMessage:
    """Either direction: error notification; the connection may close after.

    ``error_kind`` is a stable machine-readable discriminator (e.g.
    ``"auth_failed"``, ``"protocol_mismatch"``); ``message`` is the operator-facing
    detail. ``recoverable`` is ``True`` if the client may continue.
    """

    error_kind: str
    message: str
    recoverable: bool
    kind: ClassVar[str] = "error"

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, **_payload_without_kind(self)})

    @classmethod
    def from_json(cls, s: str) -> Self:
        payload = json.loads(s)
        _check_kind(payload, cls.kind)
        error_kind = _require(payload, "error_kind", str)
        message = _require(payload, "message", str)
        # ``bool`` is a subclass of ``int`` in Python; this check accepts exactly True/False.
        recoverable = _require(payload, "recoverable", bool)
        return cls(error_kind=error_kind, message=message, recoverable=recoverable)


# The discriminator -> class registry; ``from_json_any`` dispatches via it.
_MESSAGE_REGISTRY: dict[
    str,
    type[Hello | SeatAssigned | ObservationMessage | ActionMessage | FinalScores | ErrorMessage],
] = {
    Hello.kind: Hello,
    SeatAssigned.kind: SeatAssigned,
    ObservationMessage.kind: ObservationMessage,
    ActionMessage.kind: ActionMessage,
    FinalScores.kind: FinalScores,
    ErrorMessage.kind: ErrorMessage,
}


def from_json_any(
    s: str,
) -> Hello | SeatAssigned | ObservationMessage | ActionMessage | FinalScores | ErrorMessage:
    """Parse any wire message, dispatching on the top-level ``kind`` field.

    Raises ``ValueError`` on malformed JSON, missing/unknown ``kind``, or any
    per-class field validation failure.
    """
    try:
        data = json.loads(s)
    except json.JSONDecodeError as e:
        # Re-raise as ValueError so callers handle one exception type.
        raise ValueError(f"from_json_any: malformed JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(
            f"from_json_any: expected a JSON object at the top level, got {type(data).__name__}"
        )
    kind = data.get("kind")
    if kind not in _MESSAGE_REGISTRY:
        raise ValueError(
            f"from_json_any: unknown kind {kind!r}; expected one of {sorted(_MESSAGE_REGISTRY)}"
        )
    cls = _MESSAGE_REGISTRY[kind]
    return cls.from_json(s)


__all__ = [
    "PROTOCOL_VERSION",
    "REGISTRY_DEFAULT_ACTION",
    "ActionMessage",
    "ErrorMessage",
    "FinalScores",
    "Hello",
    "ObservationMessage",
    "SeatAssigned",
    "from_json_any",
]
