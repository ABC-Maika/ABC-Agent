"""The live transport: a third, sibling view over the single ``World.step`` seam.

Reimplements no economics — the seam is called exactly once per tick by the
server, mirroring the ``envs/`` relationship to ``core/``.

Lazy-import contract: the optional ``websockets`` dependency is imported only
inside ``server.py`` / ``client.py`` at the point of first use, so
``import retail_simulator.live`` stays cheap (stdlib + numpy only).
"""

from __future__ import annotations

from retail_simulator.live.client import LiveClient
from retail_simulator.live.play import play_one_game
from retail_simulator.live.protocol import (
    PROTOCOL_VERSION,
    REGISTRY_DEFAULT_ACTION,
    ActionMessage,
    ErrorMessage,
    FinalScores,
    Hello,
    ObservationMessage,
    SeatAssigned,
    from_json_any,
)
from retail_simulator.live.replay import (
    ActionLogRecord,
    ReplayManifest,
    ReplayResult,
    read_action_log,
    read_manifest,
    replay_game_dir,
)
from retail_simulator.live.replay import replay as run_replay
from retail_simulator.live.series import SeriesGameResult, SeriesResult, run_series
from retail_simulator.live.server import ServeConfig, ServeResult, serve_game

# Re-export the replay function as ``run_replay`` so the attribute does not
# shadow the ``replay`` submodule.
# wrapper use ``run_replay``.


__all__ = [
    "ActionLogRecord",
    "ActionMessage",
    "ErrorMessage",
    "FinalScores",
    "Hello",
    "LiveClient",
    "ObservationMessage",
    "PROTOCOL_VERSION",
    "REGISTRY_DEFAULT_ACTION",
    "play_one_game",
    "ReplayManifest",
    "ReplayResult",
    "SeatAssigned",
    "SeriesGameResult",
    "SeriesResult",
    "ServeConfig",
    "ServeResult",
    "from_json_any",
    "run_series",
    "read_action_log",
    "read_manifest",
    "replay_game_dir",
    "run_replay",
    "serve_game",
]
