"""Second search batch: levers/cap around the demand-order demonstrator."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from concurrent.futures import ProcessPoolExecutor
from script_search import evaluate
cands = [{"cap": 10}, {"cap": 14}, {"cap": 16}, {"research": 0.0}, {"wage_base": 0.3}, {"wage_base": 0.6},
         {"reserve": 5000.0}, {"price": 1.9}, {"loyalty": 0.3}, {"warehouse": 0.1}, {"service": 0.7}, {"cap": 14, "reserve": 5000.0}]
if __name__ == "__main__":
    with ProcessPoolExecutor(12) as pool:
        evaluate(pool, cands)
