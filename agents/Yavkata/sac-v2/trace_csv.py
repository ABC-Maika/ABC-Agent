"""Summarise an arena_competition.py --csv decision log per seat (stores, cash, levers over time, expansion order)."""
import csv, sys
from collections import defaultdict
rows = defaultdict(dict)
for r in csv.DictReader(open(sys.argv[1])):
    rows[r["name"]][int(r["round"])] = r
marks = [1, 2, 3, 4, 5, 8, 12, 20, 30, 50, 100, 200, 300, 500, 700, 1000]
cols = ["stores", "cash", "market_share", "employee_happiness", "price_index", "marketing", "promotion", "service_level", "research", "wage_spend", "warehouse_invest", "loyalty_spend"]
for nm, g in rows.items():
    print("==", nm, "total profit", round(sum(float(x["profit"]) for x in g.values())))
    print("round " + " ".join(f"{c[:8]:>9s}" for c in cols) + "   cumprofit")
    cum = 0; cums = {}
    for k in sorted(g):
        cum += float(g[k]["profit"]); cums[k] = cum
    for m in marks:
        if m in g:
            print(f"{m:5d} " + " ".join(f"{float(g[m][c]):9.2f}" for c in cols) + f" {cums[m]:11.0f}")
    ex = [(k, int(float(g[k]["expansion"]))) for k in sorted(g) if float(g[k]["opened"]) > float(g.get(k - 1, g[k])["opened"]) ] if False else []
    prev = 0; ev = []
    for k in sorted(g):
        o = int(float(g[k]["opened"]))
        if o > prev: ev.append((k, int(float(g[k]["expansion"])))); prev = o
    print("opened (round, region):", ev)
