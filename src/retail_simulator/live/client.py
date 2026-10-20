"""Reference Python WebSocket client for the live server.

One :class:`LiveClient` per asyncio task; not thread-safe — the mutable
internal slots are touched only inside the owning event loop.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from retail_simulator.core.encoding import flatten_action_space_layout
from retail_simulator.live.protocol import (
    PROTOCOL_VERSION,
    ActionMessage,
    ErrorMessage,
    FinalScores,
    Hello,
    ObservationMessage,
    SeatAssigned,
    from_json_any,
)

logger = logging.getLogger(__name__)

# Derived from the current action schema's flat layout, not hardcoded.
_EXPECTED_ACTION_SHAPE: tuple[int, ...] = (flatten_action_space_layout().total_dim,)


@dataclass
class LiveClient:
    """Reference Python WebSocket client for the live server.

    Constructor:
        uri: ``ws://host:port`` — the server's WebSocket endpoint.
        agent_name: identifies this client to the server (echoed in the
            :class:`Hello` frame and surfaced in server logs).
        token: shared-token auth; ``None`` when the server runs with
            ``--no-auth``. Forwarded verbatim into the :class:`Hello` frame.
    """

    uri: str
    agent_name: str
    token: str | None = None

    # Internal state populated by ``connect()`` and mutated by the other async
    # methods; ``repr=False`` keeps debug prints from dumping the connection or
    # the full :class:`SeatAssigned` payload.
    _seat_assigned: SeatAssigned | None = field(default=None, init=False, repr=False)
    _connection: Any = field(default=None, init=False, repr=False)
    # Advances inside ``submit_action``; starts at 0 so the first submission
    # after ``connect()`` carries ``tick=0``, matching the server's numbering.
    _current_tick: int = field(default=0, init=False, repr=False)
    # End-of-game payload captured by ``iter_observations``; read via the
    # ``final_scores`` property after the ``async for`` loop completes.
    _final_scores: FinalScores | None = field(default=None, init=False, repr=False)

    @property
    def seat_assigned(self) -> SeatAssigned | None:
        """The server's :class:`SeatAssigned` payload, populated by :meth:`connect`."""
        return self._seat_assigned

    @property
    def final_scores(self) -> FinalScores | None:
        """The end-of-game :class:`FinalScores`; ``None`` until the game ends."""
        return self._final_scores

    async def connect(self) -> SeatAssigned:
        """Open the WebSocket, send :class:`Hello`, receive :class:`SeatAssigned`.

        Caches the response on :attr:`seat_assigned` before returning it.

        Raises:
            ConnectionError: server unreachable, transport closed during the
                handshake, or the server rejected the connection.
            ValueError: server sent a well-formed frame of an unexpected type.
        """
        import websockets
        import websockets.exceptions

        # Lazy imports: websockets is only needed at connect time.
        import retail_simulator
        from retail_simulator.core.schema import ACTION_SCHEMA_VERSION

        try:
            self._connection = await websockets.connect(self.uri)
        except (OSError, websockets.exceptions.WebSocketException) as e:
            # Funnel both failure modes into one ConnectionError for callers.
            raise ConnectionError(f"failed to open WebSocket {self.uri!r}: {e}") from e

        hello = Hello(
            agent_name=self.agent_name,
            token=self.token,
            protocol_version=PROTOCOL_VERSION,
            schema_version=ACTION_SCHEMA_VERSION,
            client_name=f"retail-simulator-py/{retail_simulator.__version__}",
        )
        try:
            await self._connection.send(hello.to_json())
            raw = await self._connection.recv()
        except (OSError, websockets.exceptions.ConnectionClosedError) as e:
            # Best-effort cleanup before re-raising.
            await self._safe_close()
            raise ConnectionError(f"connection closed during handshake on {self.uri!r}: {e}") from e

        # The protocol is text-frame JSON; decode bytes frames instead of rejecting them.
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        msg = from_json_any(raw)
        if isinstance(msg, ErrorMessage):
            await self._safe_close()
            raise ConnectionError(f"server rejected connection: {msg.error_kind}: {msg.message}")
        if not isinstance(msg, SeatAssigned):
            # The transport is healthy; a wrong frame type is a server-side protocol
            # bug, so this is a ValueError, not a ConnectionError.
            await self._safe_close()
            raise ValueError(f"expected SeatAssigned during handshake, got {type(msg).__name__}")
        self._seat_assigned = msg
        return msg

    async def iter_observations(self) -> AsyncIterator[ObservationMessage]:
        """Yield one :class:`ObservationMessage` per tick.

        Stops on :class:`FinalScores` (captured on :attr:`final_scores`). A
        recoverable :class:`ErrorMessage` is logged and skipped; a fatal one
        closes the socket and raises.

        Raises:
            RuntimeError: :meth:`connect` was never called.
            ConnectionError: server sent a fatal :class:`ErrorMessage`.
        """
        if self._connection is None:
            raise RuntimeError("LiveClient.iter_observations: must call connect() first")

        async for raw in self._connection:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            msg = from_json_any(raw)
            if isinstance(msg, ObservationMessage):
                yield msg
            elif isinstance(msg, FinalScores):
                # End-of-game sentinel: capture it, then stop; the server sends its
                # own close frame, so we do not close first.
                self._final_scores = msg
                return
            elif isinstance(msg, ErrorMessage):
                if not msg.recoverable:
                    # Fatal: close our side too and raise, so an aborted game is
                    # distinguishable from a normal end.
                    await self._safe_close()
                    raise ConnectionError(
                        f"server error during play: {msg.error_kind}: {msg.message}"
                    )
                # Recoverable: the server keeps the connection open; log and continue.
                logger.warning("live server warning: %s: %s", msg.error_kind, msg.message)
            else:
                # A wrong frame type during play is a server-side bug; log and continue.
                logger.warning("unexpected message type during play: %s", type(msg).__name__)

    async def submit_action(self, action: npt.NDArray[np.float32]) -> None:
        """Send one :class:`ActionMessage` for the current tick.

        ``action`` must be a flat float32 vector of the current action
        schema's width; shape and dtype are validated before the wire.

        Raises:
            ValueError: wrong shape or dtype.
            RuntimeError: :meth:`connect` was never called.
            ConnectionError: socket closed while sending.
        """
        # Validate at the boundary: the common policy bugs are a missing
        # float32 cast and a (1, N) shape.
        if action.shape != _EXPECTED_ACTION_SHAPE:
            raise ValueError(
                f"submit_action: expected shape {_EXPECTED_ACTION_SHAPE}, got {action.shape}"
            )
        if action.dtype != np.float32:
            raise ValueError(f"submit_action: expected dtype float32, got {action.dtype}")
        if self._connection is None:
            raise RuntimeError("LiveClient.submit_action: must call connect() first")
        if self._seat_assigned is None:
            # Defensive: ``connect()`` sets ``_seat_assigned`` before returning, so
            # this is unreachable today.
            raise RuntimeError("LiveClient.submit_action: handshake did not complete")

        action_msg = ActionMessage(tick=self._current_tick, action=action.tolist())
        try:
            await self._connection.send(action_msg.to_json())
        except OSError as e:
            # Re-raise as ConnectionError so callers handle one boundary exception
            # type; OSError covers both socket failures and websockets close errors.
            raise ConnectionError(
                f"submit_action: send failed at tick {self._current_tick}: {e}"
            ) from e

        # Increment only on a successful send, so a retry resends the same tick.
        self._current_tick += 1

    async def close(self) -> None:
        """Close the WebSocket gracefully.

        Idempotent: a no-op before :meth:`connect` or after a prior :meth:`close`.
        """
        await self._safe_close()

    async def _safe_close(self) -> None:
        """Close the underlying connection if open; never raise.

        A raising cleanup would mask the real error in the paths that call it.
        """
        if self._connection is None:
            return
        connection = self._connection
        # Null first so a re-entrant call sees the closed state and short-circuits.
        self._connection = None
        try:
            await connection.close()
        except Exception:  # noqa: BLE001  # intentional: cleanup must not raise
            # Swallowed: if the peer already tore down the socket, close() may raise;
            # the surrounding exception (if any) is the real one to surface.
            logger.debug("LiveClient._safe_close: ignored exception during close", exc_info=True)


__all__ = ["LiveClient"]
