"""Play one arena match and write a self-contained, self-explaining HTML replay (Bulgaria map).

Same field options as ``scripts/arena_competition.py`` (balanced world only). The page explains
the competition, then replays the match: region map, leaderboard, each seat's live strategy,
an event feed and score / share / store charts. Open the written file in any browser.

    python scripts/arena_replay.py --rounds 300 --npc cherry-picker --npc fortress \
        --npc fast-follower --npc sprawler --out runs/replay.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from retail_simulator.envs.arena import arena_home_regions
from retail_simulator.envs.arena_match import ArenaMatch, ArenaSeatSpec, resolve_arena_world
from aggressive_npcs import AGGRESSIVE_DESCRIPTIONS, register

# Region 0 is stranded (never enterable); regions 1..24 cycle these four classes in order.
SEATS = 6  # team rule: every simulation seats exactly six agents
REGION_CLASSES = ["METRO", "AFFLUENT", "LOYAL", "FRONTIER"]

NPC_LABELS = register()  # organizer archetypes + our aggressive variants

ARCHETYPES = {
    "sprawler": "Low price, minimal spend on everything, opens the lowest-index empty region every tick.",
    "fortress": "Max marketing, assortment and loyalty spend, mid price. Stops at 3 stores.",
    "cherry-picker": "Premium price, only expands into the big METRO regions.",
    "fast-follower": "Mid price, follows rivals into contested regions, else the largest empty one.",
}

ASSETS = Path(__file__).parent / "arena_replay_assets"
# Region outlines and province names follow the region indices of the 25-region arena world.
# Geometry (bulgaria_regions.json) was copied from the competition site's leaderboard map.
LEVER_FMT = {"price_index": {"price": 1}, "automation": {"tier": 1}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--player", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--npc", action="append", default=[], choices=list(NPC_LABELS))
    parser.add_argument("--out", type=Path, default=Path("runs/replay.html"))
    args = parser.parse_args()

    seats = []
    metas: dict[str, dict] = {}
    for spec in args.player:
        label, _, path = spec.partition("=")
        seats.append(ArenaSeatSpec(name=label, kind="rl", checkpoint=path))
        # train_vs_npcs.py writes <checkpoint>.json (algo, reward, steps); absent for other trainers.
        sidecar = Path(path).with_suffix(".json")
        if sidecar.exists():
            metas[label] = json.loads(sidecar.read_text())
    seats += [ArenaSeatSpec(name=n, kind="npc", archetype=n) for n in args.npc]
    if len(seats) != SEATS:
        parser.error(f"every simulation runs with exactly {SEATS} seats, got {len(seats)}")

    cfg, _, _ = resolve_arena_world(balanced=True)
    homes = arena_home_regions(len(seats))
    match = ArenaMatch(cfg, seats, seed=args.seed, homes=homes)
    lever_names: list[str] = []
    rounds = []
    for _ in range(args.rounds):
        res = match.step()
        if not lever_names:
            lever_names = [n for n in res.seats[0].levers if n != "expansion"]
        rounds.append(
            {
                "score": [round(s.reward, 2) for s in res.seats],
                "profit": [round(s.profit, 1) for s in res.seats],
                "revenue": [round(s.revenue, 1) for s in res.seats],
                "happy": [round(s.employee_happiness, 3) for s in res.seats],
                "cash": [round(s.cash, 1) for s in res.seats],
                "share": [round(s.market_share, 4) for s in res.seats],
                "stores": [s.stores for s in res.seats],
                "opened": [list(s.opened_regions) for s in res.seats],
                "levers": [{k: round(v, 3) for k, v in s.levers.items()} for s in res.seats],
                "rshare": [[round(v, 3) for v in row] for row in res.region_share],
                "rstores": [list(row) for row in res.region_stores],
            }
        )
    data = {
        "seats": [
            {"name": s.name, "kind": s.kind, "home": h, "meta": metas.get(s.name)}
            for s, h in zip(seats, homes)
        ],
        "classes": ["STRANDED"]
        + [REGION_CLASSES[(k - 1) % 4] for k in range(1, match.n_regions)],
        "price": cfg.expansion.expansion_capex,
        "demand": [r.base_regional_demand for r in cfg.demand.regions],
        "leverFmt": LEVER_FMT,
        "archetypes": {**ARCHETYPES, **AGGRESSIVE_DESCRIPTIONS},
        "leverNames": lever_names,
        "rounds": rounds,
    }
    page = (ASSETS / "template.html").read_text(encoding="utf-8")
    geometry = (ASSETS / "bulgaria_regions.json").read_text(encoding="utf-8")
    for key, value in {
        "__DATA__": json.dumps(data),
        "__MAP__": geometry,
        "__VB__": json.loads(geometry)["viewBox"],
        "__SEED__": str(args.seed),
        "__ROUNDS__": str(args.rounds),
        "__PRICE__": f"{cfg.expansion.expansion_capex:,.0f}",
    }.items():
        page = page.replace(key, value)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"replay -> {args.out} ({match.n_regions} regions, {len(seats)} seats, {args.rounds} rounds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
