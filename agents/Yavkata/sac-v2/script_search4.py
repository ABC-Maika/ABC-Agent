"""Fourth batch: how low can service go."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from concurrent.futures import ProcessPoolExecutor
from script_search import evaluate
cands = [{"service": 0.15, "price": 1.9, "research": 0.0}, {"service": 0.0, "price": 1.9, "research": 0.0}]
if __name__ == "__main__":
    with ProcessPoolExecutor(6) as pool:
        evaluate(pool, cands)
