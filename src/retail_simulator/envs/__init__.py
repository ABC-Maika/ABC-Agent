"""RL adapter layer: Gymnasium Spaces, action translation, RetailEnv, RetailParallelEnv.

Importing this package imports gymnasium and registers the gym id ``RetailSim-v0``.
"""

from __future__ import annotations

from retail_simulator.envs.gym_env import ENV_ID, RetailEnv, register_env
from retail_simulator.envs.parallel_env import RetailParallelEnv

# Register on import: any path that reaches envs/ makes the gym id available.
register_env()

__all__ = ["ENV_ID", "RetailEnv", "RetailParallelEnv", "register_env"]
