"""Third batch: combine the winners of batch 2 (service 0.7, price 1.9, research 0)."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from concurrent.futures import ProcessPoolExecutor
from script_search import evaluate
cands = [{"service": 0.7, "price": 1.9, "research": 0.0}, {"service": 0.5}, {"service": 0.5, "price": 1.9, "research": 0.0},
         {"service": 0.7, "research": 0.0}, {"service": 0.3, "price": 1.9, "research": 0.0}, {"service": 0.7, "price": 1.8, "research": 0.0}]
if __name__ == "__main__":
    with ProcessPoolExecutor(12) as pool:
        evaluate(pool, cands)
