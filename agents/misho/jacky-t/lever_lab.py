"""A scripted seat with chosen lever numbers, for sweeps and replays (registered like scripts/aggressive_npcs).

Expansion is the 'biggest class first, up to 12 stores' rule of orcho-recipe2; every lever is a number.
Params come from the LAB_PARAMS environment variable (JSON), else the defaults below = Orcho v5's measured numbers.
"""
import json, os, sys
from typing import Any

sys.path.insert(0, "scripts")
from aggressive_npcs import AGGRESSIVE_DESCRIPTIONS, register  # noqa: E402

from retail_simulator.core.encoding import encode_action  # noqa: E402
from retail_simulator.envs import arena_match  # noqa: E402
from retail_simulator.envs import geo_archetypes as geo  # noqa: E402

V5 = dict(price=2.0, marketing=1.0, assortment=1.0, promotion=1.0, service=1.0, loyalty=0.0,
          research=0.18, wage_base=0.45, wage_gain=4.0, warehouse=0.0, cap=12, order="high", nclass=2, prio=None, frugal_ticks=0, save_band=0.0, early_ticks=0, early_mkt=1.0, early_ass=1.0, crowd_k=0.0, buffer=0.0, buf_from=4, wage_max=1.0, sw_tick=None, sw_thresh=99, alt=None, vr=False, vr_a=0.495, vr_b=172.0, vr_g=0.5, vr_theta=40.0, collapse_price=None, war_cash=0.0, war_price=1.6, endcut_t=None, endcut_mkt=0.5, auto_cash=0.0, auto_tier=1, stop_t=None, crowd_price=0.0, crowd_floor=1.5, ad_tick=None, ad_stores=5, ad_profit=0.0, ad_alt=None, stuck=None, war_profit=0.0)


def params(extra: dict | None = None) -> dict:
    p = dict(V5)
    if os.environ.get("LAB_PARAMS"):
        p.update(json.loads(os.environ["LAB_PARAMS"]))
    if extra:
        p.update(extra)
    return p


def _lab_predict(base: dict, n_regions: int) -> Any:
    idx = geo._name_to_index(n_regions)
    p = dict(base)  # re-set each match; ``home_over`` = {"21": {...}} swaps in numbers for that home
    tick = [0]
    decided = [False]
    adapted = [False]
    war = [False]

    def target(obs):
        if float(obs[idx["stores"]]) >= p["cap"] or (p["stop_t"] is not None and tick[0] > p["stop_t"]):
            return 0  # tick[0] has already advanced past this round
        if p["buffer"] and float(obs[idx["stores"]]) >= p["buf_from"] and float(obs[idx["cash"]]) < 5000.0 + p["buffer"]:
            return 0  # keep a cash cushion beyond the 5,000 gate once the opening is done
        empty = list(geo._own_empty_regions(obs, idx=idx, candidates=range(1, n_regions)))
        if p["vr"]:
            # Value rule (calibrated by runs/calib.py: profit per round = 0.495 x sum(demand x share) - 172 x stores):
            # open the region with the best expected net profit per round, if it clears ``vr_theta``. The share
            # I can expect in a region with n rivals present is min(0.6, 1 / (n + 1 + vr_g)).
            demand = {0: 1400.0, 1: 900.0, 2: 600.0, 3: 350.0}
            best, best_v = 0, p["vr_theta"]
            for r in empty:
                n = float(obs[idx[f"others_present_region_{r}"]])
                share = min(0.6, 1.0 / (n + 1.0 + p["vr_g"]))
                v = p["vr_a"] * demand[(r - 1) % 4] * share - p["vr_b"]
                if v > best_v or (v == best_v and best and r > best):
                    best, best_v = r, v
            return best
        if p["crowd_k"]:
            # crowd-aware: score = class demand / (1 + k x rivals already present), highest score first
            demand = {0: 1400.0, 1: 900.0, 2: 600.0, 3: 350.0}
            empty = [r for r in empty if (r - 1) % 4 < p["nclass"]]
            return max(empty, key=lambda r: (demand[(r - 1) % 4] / (1.0 + p["crowd_k"] * float(obs[idx[f"others_present_region_{r}"]])), r)) if empty else 0
        if p["prio"] is not None:
            # explicit class priority, e.g. [1, 0] = AFFLUENT before METRO (class = (region - 1) % 4:
            # 0 METRO, 1 AFFLUENT, 2 LOYAL, 3 FRONTIER); region number high or low first inside a class
            empty = [r for r in empty if (r - 1) % 4 in p["prio"]]
            sign = -1 if p["order"] == "high" else 1
            return min(empty, key=lambda r: (p["prio"].index((r - 1) % 4), sign * r)) if empty else 0
        if p["order"] == "high":
            # orcho-recipe4: METRO then AFFLUENT only, HIGHEST region number first (most agents open the
            # lowest number first, so this dodges their crowd in the opening cash race).
            empty = [r for r in empty if (r - 1) % 4 < p["nclass"]]  # 2: METRO+AFFLUENT, 3: + LOYAL
            return min(empty, key=lambda r: ((r - 1) % 4, -r)) if empty else 0
        return min(empty, key=lambda r: ((r - 1) % 4, r)) if empty else 0

    def predict(obs, *, episode_start=False, **_kw):
        if episode_start:
            tick[0] = 0
        t = tick[0]; tick[0] += 1
        if t == 0:
            p.clear(); p.update(base); decided[0] = False; adapted[0] = False
            if base.get("home_over"):
                empty0 = set(geo._own_empty_regions(obs, idx=idx, candidates=range(1, n_regions)))
                owned = [r for r in range(1, n_regions) if r not in empty0]
                p.update(base["home_over"].get(str(owned[0]) if owned else "", {}))
        if os.environ.get("LAB_DEBUG") and 1 <= t <= 8:
            em = set(geo._own_empty_regions(obs, idx=idx, candidates=range(1, n_regions)))
            own = [r for r in range(1, n_regions) if r not in em]
            crowd = sum(float(obs[idx[f"others_present_region_{r}"]]) for r in own)
            print(f"   dbg t{t} stores {len(own)} crowd {crowd:.0f} cash {float(obs[idx['cash']]):,.0f}", flush=True)
        if p["ad_tick"] is not None and t >= p["ad_tick"] and t < p["ad_tick"] + 300 and not adapted[0]:
            # Adaptive switch: if I am still small and losing money, the recipe is failing; play the other one.
            if float(obs[idx["stores"]]) <= p["ad_stores"] and float(obs[idx["last_profit"]]) < p["ad_profit"]:
                adapted[0] = True
                p.update(p["ad_alt"])
        if p["sw_tick"] is not None and t == p["sw_tick"] and not decided[0]:
            # Rival-aware switch: if rivals already sit in my regions by round ``sw_tick``, the METRO-first
            # opening is being split; play the ``alt`` opening (e.g. AFFLUENT, then LOYAL, with a smaller cap).
            decided[0] = True
            em = set(geo._own_empty_regions(obs, idx=idx, candidates=range(1, n_regions)))
            crowd = sum(float(obs[idx[f"others_present_region_{r}"]]) for r in range(1, n_regions) if r not in em)
            if crowd >= p["sw_thresh"]:
                p.update(p["alt"])
        wage = min(p["wage_max"], p["wage_base"] + p["wage_gain"] * (1.0 - float(obs[idx["employee_happiness"]])))
        research = p["research"]
        cash = float(obs[idx["cash"]])
        # Cash race: a store costs 5,000, and missing the gate by ~100 in round 2 cost the 4th store (homes 1
        # and 5, where rivals crowd the lab's home first). No wage and no research (a) in the first
        # ``frugal_ticks`` rounds and (b) whenever cash is within ``save_band`` below the gate, until the store opens.
        saving = t < p["frugal_ticks"] or (p["save_band"] and 5000.0 - p["save_band"] <= cash < 5000.0 and float(obs[idx["stores"]]) < p["cap"])
        if saving:
            wage, research = 0.0, 0.0
        price = p["price"]
        stores_now = float(obs[idx["stores"]])
        if p["war_cash"]:
            # price war with idle cash: once full size with a big pile, undercut until the pile halves
            if stores_now >= p["cap"] and cash >= p["war_cash"] and float(obs[idx["last_profit"]]) >= p["war_profit"]:
                war[0] = True
            elif cash < p["war_cash"] / 2:
                war[0] = False
            if war[0]:
                price = p["war_price"]
        if p["crowd_price"]:
            em2 = set(geo._own_empty_regions(obs, idx=idx, candidates=range(1, n_regions)))
            own2 = [r for r in range(1, n_regions) if r not in em2]
            if own2:
                mean_riv = sum(float(obs[idx[f"others_present_region_{r}"]]) for r in own2) / len(own2)
                price = max(p["crowd_floor"], price - p["crowd_price"] * max(0.0, mean_riv - 1.0))
        marketing_now = p["early_mkt"] if t < p["early_ticks"] else p["marketing"]
        if p["endcut_t"] is not None and t >= p["endcut_t"]:
            marketing_now = p["endcut_mkt"]  # the last rounds: spend that only pays later is wasted
            research = 0.0
        tier = p["auto_tier"] if (p["auto_cash"] and cash >= p["auto_cash"]) else 0
        if p["collapse_price"] is not None and t > 120 and float(obs[idx["stores"]]) <= 3 and cash < 0:
            price = p["collapse_price"]  # restructured and out of the race: undercut the rivals in the regions I still hold
        so = p["stuck"]
        in_stuck = so is not None and t > 120 and stores_now <= 4 and cash < 0
        if in_stuck:
            price = so.get("price", price); marketing_now = so.get("marketing", marketing_now)
            wage = so["wage"] if "wage" in so else wage
            research = so.get("research", research)
        return encode_action(geo._structured_action(
            price=price, marketing=marketing_now,
            assortment=((so or {}).get("assortment", p["assortment"]) if in_stuck else (p["early_ass"] if t < p["early_ticks"] else p["assortment"])), promotion=((so or {}).get("promotion", p["promotion"]) if in_stuck else p["promotion"]),
            expansion_choice=target(obs), research=research, service_level=((so or {}).get("service", p["service"]) if in_stuck else p["service"]),
            automation_tier=tier, loyalty_spend=p["loyalty"], n_regions=n_regions,
            wage_spend=wage, warehouse_invest=p["warehouse"]))
    return predict


LAB_LABELS = {"lab-v5": dict(), "lab-opt": None, "lab-x": None}


def register_lab(opt: dict | None = None) -> None:
    register()
    if "lab-v5" in arena_match.GEO_ARCHETYPE_LABELS:
        return
    base = arena_match.geo_archetypes

    def with_lab(n_regions, region_table):
        roster = dict(base(n_regions, region_table))
        roster["lab-v5"] = _lab_predict(dict(V5), n_regions)
        roster["lab-opt"] = _lab_predict(params(opt), n_regions)
        roster["lab-x"] = _lab_predict(params(), n_regions)
        return roster

    arena_match.geo_archetypes = with_lab
    arena_match.GEO_ARCHETYPE_LABELS = (*arena_match.GEO_ARCHETYPE_LABELS, "lab-v5", "lab-opt", "lab-x")
    AGGRESSIVE_DESCRIPTIONS["lab-v5"] = ("LAB seat playing Orcho v5's measured numbers: price 2.0, marketing/assortment/"
        "promotion/service 1.0, loyalty 0, research 0.18, wage 0.45 (+4x unhappiness), no warehouse; biggest class first up to 12 stores.")
    AGGRESSIVE_DESCRIPTIONS["lab-opt"] = "LAB seat playing the swept best lever numbers (see runs/lever_report.md)."
    AGGRESSIVE_DESCRIPTIONS["lab-x"] = "LAB seat with swept parameters."
