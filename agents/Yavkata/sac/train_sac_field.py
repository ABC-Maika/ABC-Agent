"""SAC trained against Orcho and Krischo (real checkpoints plus scripted stand-ins) and the NPCs.

Implements section 4.1 of ``COUNTER_STRATEGY.md``. ``NpcFieldEnv`` seats one expansion-masked learner
against four opponents; here the four are drawn per episode from:
  * the real Orcho checkpoint (MLP) or an "orcho-like" scripted seat (store cap 8, 10 or 12),
  * the real Krischo checkpoint (LSTM, own recurrent state) or a "krischo-like" scripted seat (cap 6, 8 or 10),
  * the four self-check NPCs.
Every episode has at least one Orcho-type opponent; the rest is filled at random so the policy does not
overfit one rival. Reward stays profit only. Scripted seats are unmasked in the world, so they check cash
before opening a store.

Everything else is the SAC path of ``scripts/train_vs_npcs.py`` (bounded logits, obs scale folded into the
network, profit x 1e-3), reached by swapping the env class it uses. Run from the repo root:
    python agents/Yavkata/sac/train_sac_field.py --steps 50000 --out agents/Yavkata/sac/checkpoint_field.zip
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

import train_vs_npcs  # noqa: E402
from retail_simulator.core.encoding import encode_action  # noqa: E402
from retail_simulator.core.schema import observation_schema  # noqa: E402
from retail_simulator.envs.geo_archetypes import _structured_action, geo_archetypes  # noqa: E402
from retail_simulator.envs.ladder import _materialize_predicts  # noqa: E402
from retail_simulator.envs.spaces import build_action_space  # noqa: E402

ORCHO_ZIP = REPO / "agents/orcho/market-share/checkpoint.zip"
KRISCHO_ZIP = REPO / "agents/Krischo/rppo-lstm/checkpoint.zip"
EXPANSION_CAPEX = 5_000.0
P_REAL = 0.5  # chance an Orcho / Krischo seat is the real checkpoint rather than the scripted copy
P_KRISCHO = 0.7  # chance a Krischo-type seat is present in an episode
ORCHO_CAPS = (8, 10, 12)
KRISCHO_CAPS = (6, 8, 10)

Predict = Callable[..., npt.NDArray[np.float32]]


def scripted_seat(
    cfg: Any,
    *,
    cap: int,
    order: Sequence[int],
    price: float,
    research: float,
    service: float,
    warehouse: float,
    wage: float,
) -> Predict:
    """Constant levers; open the next region of ``order`` while under ``cap`` stores and cash >= capex."""
    n = len(cfg.demand.regions)
    idx = {f.name: i for i, f in enumerate(observation_schema(n))}

    def predict(obs: npt.NDArray[np.float32], *, episode_start: bool = False, **_: Any):
        target = 0
        if obs[idx["stores"]] < cap and obs[idx["cash"]] >= EXPANSION_CAPEX:
            target = next((r for r in order if obs[idx[f"stores_region_{r}"]] == 0), 0)
        return encode_action(
            _structured_action(
                price=price,
                marketing=1.0,
                assortment=1.0,
                promotion=1.0,
                expansion_choice=target,
                research=research,
                service_level=service,
                automation_tier=0,
                loyalty_spend=0.0,
                n_regions=n,
                wage_spend=wage,
                warehouse_invest=warehouse,
            )
        )

    return predict


class FieldEnv(train_vs_npcs.NpcFieldEnv):
    """``NpcFieldEnv`` with Orcho / Krischo opponents mixed into the roster."""

    _real: dict[str, Predict] | None = None

    def _real_predicts(self) -> dict[str, Predict]:
        if FieldEnv._real is None:  # loaded once per process; the closures reset on episode_start
            space = build_action_space()
            FieldEnv._real = _materialize_predicts(
                {"Orcho": str(ORCHO_ZIP), "Krischo": str(KRISCHO_ZIP)},
                action_space=space,
                config=self._config,
            )
        return FieldEnv._real

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        out = super().reset(seed=seed, options=options)
        rng = self.np_random
        regions = self._regions
        by_demand = sorted(range(1, len(regions)), key=lambda r: (-regions[r].base_regional_demand, r))
        metro = [r for r in by_demand if (r - 1) % 4 == 0]
        affl = [r for r in by_demand if (r - 1) % 4 == 1]
        real = self._real_predicts()
        npcs = geo_archetypes(len(regions), regions)

        def orcho() -> Predict:
            if rng.random() < P_REAL:
                return real["Orcho"]
            return scripted_seat(
                self._config, cap=int(rng.choice(ORCHO_CAPS)), order=by_demand, price=2.0, research=0.2,
                service=1.0, warehouse=0.0, wage=0.5,
            )

        def krischo() -> Predict:
            if rng.random() < P_REAL:
                return real["Krischo"]
            return scripted_seat(
                self._config, cap=int(rng.choice(KRISCHO_CAPS)), order=metro + affl, price=1.98, research=0.05,
                service=0.95, warehouse=0.03, wage=0.5,
            )

        seats = [orcho()]
        if rng.random() < P_KRISCHO:
            seats.append(krischo())
        names = list(train_vs_npcs.NPC_FIELD)
        while len(seats) < len(self._npcs):
            seats.append(npcs[str(rng.choice(names))])
        order = rng.permutation(len(seats))
        self._npcs = [seats[i] for i in order]
        return out


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    train_vs_npcs.NpcFieldEnv = FieldEnv  # make_env_fns and estimate_obs_scale resolve it by name
    return train_vs_npcs.main(["yavkata_sac_field", "--algo", "sac", *argv])


if __name__ == "__main__":
    raise SystemExit(main())
