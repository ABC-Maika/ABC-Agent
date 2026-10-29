"""Summarise an arena_competition CSV: per seat store count over time, profit per phase, mean levers."""
import csv, sys
from collections import OrderedDict
rows = list(csv.DictReader(open(sys.argv[1])))
by = OrderedDict()
for r in rows:
    by.setdefault(r["name"], []).append(r)
cols = ["price_index","marketing","assortment","promotion","service_level","loyalty_spend","research","wage_spend","warehouse_invest","employee_happiness","market_share"]
f = lambda g, c, lo, hi: sum(float(x[c]) for x in g if lo <= int(x["round"]) <= hi) / max(1, sum(1 for x in g if lo <= int(x["round"]) <= hi))
for name, g in by.items():
    at = {int(x["round"]): x for x in g}
    print("==", name)
    print(" stores@", {r: int(at[r]["stores"]) for r in (10,50,100,200,300,500,800,1000) if r in at})
    print(" cash@", {r: int(float(at[r]["cash"])) for r in (10,100,300,500,800,1000) if r in at})
    print(" profit/round per 200:", [round(f(g,"profit",i*200+1,i*200+200)) for i in range(5)])
    for c in cols:
        print("   %-18s early %.3f  late %.3f" % (c, f(g,c,1,200), f(g,c,801,1000)))
    ex = [(int(x["round"]), int(float(x["expansion"]))) for x in g if float(x["expansion"]) > 0]
    print(" expansion picks (round,region):", ex[:16])
