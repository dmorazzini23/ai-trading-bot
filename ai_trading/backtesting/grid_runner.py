from __future__ import annotations
import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from itertools import product
from typing import Any
from ai_trading.utils import optional_import, module_ok  # AI-AGENT-REF: unify optional deps

_joblib = optional_import(
    "joblib", purpose="parallel grid search", extra="backtest"
)  # AI-AGENT-REF: extras hint uses key
if module_ok(_joblib):  # joblib available
    Parallel = _joblib.Parallel
    delayed = _joblib.delayed
else:
    Parallel = None

    def delayed(f):
        return f

def _timestamp() -> str:
    return datetime.now(UTC).strftime('%Y%m%d_%H%M%S')

def run_one(evaluator, params: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluator(params)
    return {'params': params, 'metrics': metrics}

def grid_search(evaluator, param_grid: dict[str, Iterable[Any]], n_jobs: int=-1) -> dict[str, Any]:
    keys = list(param_grid.keys())
    combos = [dict(zip(keys, vals, strict=False)) for vals in product(*[param_grid[k] for k in keys])]
    if Parallel and n_jobs != 1:
        results = Parallel(n_jobs=n_jobs, prefer='processes')((delayed(run_one)(evaluator, p) for p in combos))
    else:
        results = [run_one(evaluator, p) for p in combos]
    return {'results': results, 'count': len(results)}

def persist_artifacts(run: dict[str, Any], out_dir: str) -> str:
    ts = _timestamp()
    run_dir = os.path.join(out_dir, f'run_{ts}')
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(run, f, indent=2)
    return run_dir