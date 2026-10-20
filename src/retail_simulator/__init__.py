"""retail-simulator: a deterministic multi-agent RL environment for retail.

Public surface (Gym/PettingZoo-idiomatic; ``core/`` is private): RetailEnv,
RetailParallelEnv, ScenarioConfig, ``__version__``. Importing registers the gym id
``RetailSim-v0`` when gymnasium is installed (the adapter layer itself imports
lazily, so a bare ``import retail_simulator`` stays numpy-only).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

__all__ = ["RetailEnv", "RetailParallelEnv", "ScenarioConfig", "__version__"]

if TYPE_CHECKING:  # import only for type-checkers; never at runtime import time
    from retail_simulator.envs import RetailEnv, RetailParallelEnv
    from retail_simulator.scenarios import ScenarioConfig


def register_environments() -> bool:
    """Register ``RetailSim-v0`` with Gymnasium. Returns False if gymnasium is not installed."""
    try:
        import retail_simulator.envs  # noqa: F401  (import for its registration side effect)
    except ImportError:
        # gymnasium not installed (bare core install): nothing to register.
        return False
    return True


def __getattr__(name: str) -> Any:
    """Lazily expose RetailEnv / RetailParallelEnv / ScenarioConfig without importing gymnasium at module load."""
    if name == "RetailEnv":
        from retail_simulator.envs import RetailEnv

        return RetailEnv
    if name == "RetailParallelEnv":
        from retail_simulator.envs import RetailParallelEnv

        return RetailParallelEnv
    if name == "ScenarioConfig":
        from retail_simulator.scenarios import ScenarioConfig

        return ScenarioConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Best-effort registration on import so `import retail_simulator; gym.make(...)`
# works per the DX contract. Silent no-op when gymnasium is absent.
register_environments()
