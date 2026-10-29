"""Evaluate a candidate seat in the five-seat league field, over rotated homes and seeds, in parallel.

Field: candidate + Orcho + Krischo + Misho's jack + Yavkata's recurrent-ppo (the checkpoints currently in
the repo; Orcho is whatever ``agents/orcho/market-share/checkpoint.zip`` is on this branch). No NPCs.
A candidate is either ``("zip", path)`` or ``("params", dict)`` for the scripted strategy below
(``scripted``), so the same harness scores a trained checkpoint and a parameter-search candidate.

    python agents/Yavkata/sac/field_eval.py agents/Yavkata/sac/checkpoint.zip
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OPPONENTS = [
    ("Orcho", "agents/orcho/market-share/checkpoint.zip"),
    ("Krischo", "agents/Krischo/rppo-lstm/checkpoint.zip"),
    ("Jack", "agents/misho/jack/checkpoint.zip"),
    ("YavkataRPPO", "agents/Yavkata/recurrent-ppo/checkpoint.zip"),
]
METRO_HOMES = [1, 5, 9, 13, 17, 21]
# Six rotations of the published homes over five seats; seeds cycle 1, 2, 3.
HOMES = [tuple(METRO_HOMES[(i + k) % 6] for k in range(5)) for i in range(6)]
SEEDS = [1, 2, 3, 1, 2, 3]
DEFAULTS: dict[str, Any] = {
    "cap": 12, "order": "demand", "price": 2.0, "marketing": 1.0, "assortment": 1.0, "promotion": 1.0,
    "service": 1.0, "research": 0.2, "warehouse": 0.0, "wage_base": 0.45, "wage_gain": 4.0, "loyalty": 0.0,
}
EXPANSION_CAPEX = 5_000.0


def scripted(config: Any, params: dict[str, Any]):
    """Constant levers plus an expansion rule; the orcho-like recipe is ``DEFAULTS``."""
    from retail_simulator.core.encoding import encode_action
    from retail_simulator.core.schema import observation_schema
    from retail_simulator.envs.geo_archetypes import _structured_action

    p = {**DEFAULTS, **params}
    regions = config.demand.regions
    n = len(regions)
    idx = {f.name: i for i, f in enumerate(observation_schema(n))}
    by_demand = sorted(range(1, n), key=lambda r: (-regions[r].base_regional_demand, r))
    if p["order"] == "metro_first":
        order = [r for r in by_demand if (r - 1) % 4 == 0] + [r for r in by_demand if (r - 1) % 4 != 0]
    else:
        order = by_demand

    def predict(obs: Any, *, episode_start: bool = False, **_: Any):
        target = 0
        if obs[idx["stores"]] < p["cap"] and obs[idx["cash"]] >= EXPANSION_CAPEX:
            target = next((r for r in order if obs[idx[f"stores_region_{r}"]] == 0), 0)
        wage = min(1.0, max(0.0, p["wage_base"] + p["wage_gain"] * (1.0 - float(obs[idx["employee_happiness"]]))))
        return encode_action(
            _structured_action(
                price=p["price"], marketing=p["marketing"], assortment=p["assortment"],
                promotion=p["promotion"], expansion_choice=target, research=p["research"],
                service_level=p["service"], automation_tier=0, loyalty_spend=p["loyalty"],
                n_regions=n, wage_spend=wage, warehouse_invest=p["warehouse"],
            )
        )

    return predict


def run_match(job: tuple[tuple[str, Any], int, tuple[int, ...], int]) -> list[float]:
    """One 1000-round match; returns every seat's total profit, candidate first."""
    (kind, spec), seed, homes, rounds = job
    from retail_simulator.envs.arena import arena_balanced_config
    from retail_simulator.envs.arena_match import ArenaMatch, ArenaSeatSpec

    cfg = arena_balanced_config()
    first = str(REPO / spec) if kind == "zip" and not Path(spec).is_absolute() else str(spec)
    seats = [ArenaSeatSpec("cand", "rl", checkpoint=first if kind == "zip" else REPO / OPPONENTS[0][1])]
    seats += [ArenaSeatSpec(name, "rl", checkpoint=REPO / path) for name, path in OPPONENTS]
    match = ArenaMatch(cfg, seats, seed=seed, homes=homes)
    if kind == "params":
        match._policies[0] = scripted(cfg, spec)
    total = np.zeros(len(seats))
    for _ in range(rounds):
        result = match.step()
        total += [s.profit for s in result.seats]
    return total.tolist()


def evaluate(candidate: tuple[str, Any], rounds: int = 1000, workers: int = 6, pool: Any = None) -> dict[str, Any]:
    jobs = [(candidate, s, h, rounds) for s, h in zip(SEEDS, HOMES)]
    runs = list(pool.map(run_match, jobs)) if pool else list(map(run_match, jobs))
    arr = np.array(runs)
    ranks = (-arr).argsort(axis=1).argsort(axis=1)[:, 0] + 1
    return {
        "mean": float(arr[:, 0].mean()), "min": float(arr[:, 0].min()), "mean_rank": float(ranks.mean()),
        "orcho": float(arr[:, 1].mean()), "krischo": float(arr[:, 2].mean()), "per_match": arr[:, 0].round().tolist(),
    }


def fmt(name: str, r: dict[str, Any]) -> str:
    return (f"{name:34s} mean {r['mean']:>10,.0f}  min {r['min']:>10,.0f}  rank {r['mean_rank']:.2f}  "
            f"Orcho {r['orcho']:>9,.0f}  Krischo {r['krischo']:>9,.0f}")


if __name__ == "__main__":
    cands: list[tuple[str, tuple[str, Any]]] = []
    for arg in sys.argv[1:] or ["agents/Yavkata/sac/checkpoint.zip"]:
        cands.append((Path(arg).name, ("zip", arg)))
    with ProcessPoolExecutor(6) as pool:
        for name, cand in cands:
            print(fmt(name, evaluate(cand, pool=pool)), flush=True)
