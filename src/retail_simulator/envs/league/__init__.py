"""PFSP self-play league: frozen opponent pools, training rounds, and win-rate evaluation.

This package is split into five modules by responsibility:

- ``pool``: league definitions and the :class:`FrozenPolicyPool` snapshot pool
- ``winrate``: win-rate matrix and opponent-sampling distributions
- ``training``: PettingZoo wrapper and checkpoint loading for league training
- ``runner``: :func:`run_league_training` — the multi-round PFSP training loop
- ``evaluation``: head-to-head scoring, pool-improvement and generalization reports

All public names are re-exported here so ``from retail_simulator.envs.league import ...``
keeps working unchanged.
"""

from .evaluation import (
    _position_controlled_head_to_head as _position_controlled_head_to_head,
)
from .evaluation import (
    default_holdout_scenarios,
    evaluate_pool_snapshot_improvement,
    run_real_ppo_generalization_diagnostic,
)
from .pool import (
    DEFAULT_LEAGUE_NAME,
    POOL_SPIKE_ARCHETYPE_LABELS,
    FrozenPolicyPool,
    PoolMember,
    build_league,
    known_leagues,
    make_seeded_pool,
)
from .runner import run_league_training
from .training import LeagueTrainingResult
from .winrate import (
    OpponentDistribution,
    PFSPOpponentDistribution,
    UniformOpponentDistribution,
)

__all__ = [
    "DEFAULT_LEAGUE_NAME",
    "POOL_SPIKE_ARCHETYPE_LABELS",
    "FrozenPolicyPool",
    "LeagueTrainingResult",
    "OpponentDistribution",
    "PFSPOpponentDistribution",
    "PoolMember",
    "UniformOpponentDistribution",
    "build_league",
    "default_holdout_scenarios",
    "evaluate_pool_snapshot_improvement",
    "known_leagues",
    "make_seeded_pool",
    "run_league_training",
    "run_real_ppo_generalization_diagnostic",
]
# _position_controlled_head_to_head is re-exported above (redundant alias =
# ruff F401 re-export marker). It stays out of __all__ on purpose: it is a
# private name that external modules import explicitly.
