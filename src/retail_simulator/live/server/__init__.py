"""The asyncio WebSocket server: the third transport over ``World.step``.

Hosts one :class:`RetailParallelEnv` per game, drives ``env.step`` on a
wall-clock schedule, and accepts joint actions from N remote
:class:`LiveClient` connections. The seam is called exactly once per tick; no
economics are reimplemented here. One :func:`serve_game` call drives one game
from registration through final scores.

Flat subpackage (``retail_simulator.live.server`` is this package, so imports
of its public names are unchanged):

- :mod:`config` — :class:`ServeConfig`, :class:`ServeResult`, per-connection
  state, and the pure wire-shaping helpers.
- :mod:`replay` — the action-log writer, atomic text writes, the
  :class:`ReplayManifest` snapshot, and the end-of-game score reducers.
- :mod:`game` — the per-game server state machine, the WebSocket handlers, and
  the :func:`serve_game` entry point.
"""

from retail_simulator.live.server.config import ServeConfig, ServeResult
from retail_simulator.live.server.game import serve_game

__all__ = ["ServeConfig", "ServeResult", "serve_game"]
