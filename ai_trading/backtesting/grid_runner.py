from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, Iterable
from itertools import product

try:
    from joblib import Parallel, delayed  # type: ignore
except Exception:
    Parallel = None
    def delayed(f): return f

def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

def run_one(evaluator, params: Dict[str, Any]) -> Dict[str, Any]:
    metrics = evaluator(params)
    return {"params": params, "metrics": metrics}

def grid_search(evaluator, param_grid: Dict[str, Iterable[Any]], n_jobs: int = -1) -> Dict[str, Any]:
    keys = list(param_grid.keys())
    combos = [dict(zip(keys, vals)) for vals in product(*[param_grid[k] for k in keys])]
    if Parallel and n_jobs != 1:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(run_one)(evaluator, p) for p in combos)
    else:
        results = [run_one(evaluator, p) for p in combos]
    return {"results": results, "count": len(results)}

def persist_artifacts(run: Dict[str, Any], out_dir: str) -> str:
    ts = _timestamp()
    run_dir = os.path.join(out_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    return run_dir