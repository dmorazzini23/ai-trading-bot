"""
Model registry and evaluation store.

Keeps a lightweight JSON registry in `MODELS_DIR/registry.json` with
per-symbol active version and metadata. Evaluations are appended to
`MODELS_DIR/eval/{symbol}.jsonl` to track OOS metrics for promotion.

All filesystem interactions are best-effort and gated at call time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ai_trading.paths import MODELS_DIR
from ai_trading.logging import get_logger

logger = get_logger(__name__)


_REGISTRY_PATH = MODELS_DIR / "registry.json"
_EVAL_DIR = MODELS_DIR / "eval"


def _load_registry() -> dict[str, Any]:
    try:
        if _REGISTRY_PATH.exists():
            return json.loads(_REGISTRY_PATH.read_text())
    except Exception:
        logger.debug("MODEL_REGISTRY_LOAD_FAILED", exc_info=True)
    return {}


def _save_registry(reg: dict[str, Any]) -> None:
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _REGISTRY_PATH.write_text(json.dumps(reg, indent=2))
    except Exception:
        logger.debug("MODEL_REGISTRY_SAVE_FAILED", exc_info=True)


def register_model(symbol: str, version: str, path: Path, meta: dict[str, Any] | None = None, activate: bool = True) -> None:
    """Register a model version for a symbol and optionally activate it."""
    reg = _load_registry()
    entry = reg.get(symbol) or {"versions": {}, "active": None}
    entry["versions"][version] = {
        "path": str(path),
        "meta": meta or {},
        "registered_at": datetime.now(UTC).isoformat(),
    }
    if activate:
        entry["active"] = version
    reg[symbol] = entry
    _save_registry(reg)


def set_active_model(symbol: str, version: str) -> None:
    reg = _load_registry()
    if symbol in reg and version in reg[symbol].get("versions", {}):
        reg[symbol]["active"] = version
        _save_registry(reg)


def get_active_model_meta(symbol: str) -> dict[str, Any] | None:
    reg = _load_registry()
    entry = reg.get(symbol)
    if not entry:
        return None
    ver = entry.get("active")
    if not ver:
        return None
    return entry.get("versions", {}).get(ver)


def record_evaluation(symbol: str, metrics: dict[str, Any]) -> None:
    """Append evaluation metrics for a symbol in JSONL format."""
    try:
        _EVAL_DIR.mkdir(parents=True, exist_ok=True)
        payload = {"symbol": symbol, "ts": datetime.now(UTC).isoformat(), **metrics}
        with (_EVAL_DIR / f"{symbol}.jsonl").open("a") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        logger.debug("MODEL_EVAL_WRITE_FAILED", exc_info=True)


def list_evaluations(symbol: str, limit: int = 100) -> list[dict[str, Any]]:
    try:
        path = _EVAL_DIR / f"{symbol}.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines()[-limit:]
        return [json.loads(l) for l in lines if l.strip()]
    except Exception:
        logger.debug("MODEL_EVAL_READ_FAILED", exc_info=True)
        return []

