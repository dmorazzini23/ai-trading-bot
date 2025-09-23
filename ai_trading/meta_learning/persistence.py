"""Persistence helpers for meta-learning trade history.

Provides lightweight append and load utilities that persist fills to a
canonical parquet store while remaining tolerant of missing optional
dependencies in lean deployments.
"""

from __future__ import annotations

import os

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from ai_trading.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing aid
    import pandas as pd

_CANONICAL_PATH = Path(
    Path.cwd()
    / Path(
        # Allow callers/tests to override through environment while keeping a
        # stable default inside the repository workspace.
        os.getenv("AI_TRADING_TRADE_HISTORY_PATH", "artifacts/trade_history.parquet")
    )
)

_PANDAS_MISSING_LOGGED = False


def _ensure_parent(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("TRADE_HISTORY_PARENT_CREATE_FAILED", exc_info=True)


def _coerce_mapping(record: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if is_dataclass(record):
        return asdict(record)
    return dict(getattr(record, "__dict__", {}))


def _normalise_record(raw: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(raw)
    # Normalise timestamps to datetime for parquet; accept iso strings.
    ts = record.get("entry_time")
    if isinstance(ts, str):
        try:
            record["entry_time"] = datetime.fromisoformat(ts)
        except ValueError:
            record["entry_time"] = datetime.now(UTC)
    elif ts is None:
        record["entry_time"] = datetime.now(UTC)

    if "exit_time" in record and isinstance(record["exit_time"], str):
        try:
            record["exit_time"] = datetime.fromisoformat(record["exit_time"])
        except ValueError:
            record["exit_time"] = None

    defaults: dict[str, Any] = {
        "entry_price": None,
        "exit_price": None,
        "qty": 0,
        "side": "",
        "strategy": "",
        "classification": "",
        "signal_tags": "",
        "confidence": 0.0,
        "reward": None,
        "order_id": record.get("order_id"),
        "fill_id": record.get("fill_id"),
    }
    for key, value in defaults.items():
        record.setdefault(key, value)
    return record


def _read_parquet(path: Path) -> "pd.DataFrame" | None:
    global _PANDAS_MISSING_LOGGED
    try:
        import pandas as pd
    except ImportError:
        if not _PANDAS_MISSING_LOGGED:
            logger.warning("TRADE_HISTORY_PANDAS_MISSING")
            _PANDAS_MISSING_LOGGED = True
        return None
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError) as exc:  # pragma: no cover - corrupt file guard
        logger.warning(
            "TRADE_HISTORY_READ_FAILED",
            extra={"path": str(path), "cause": exc.__class__.__name__, "detail": str(exc)},
        )
        return None


def _write_parquet(path: Path, frame: "pd.DataFrame") -> None:
    _ensure_parent(path)
    try:
        frame.to_parquet(path, index=False)
    except (OSError, ValueError) as exc:
        logger.warning(
            "TRADE_HISTORY_WRITE_FAILED",
            extra={"path": str(path), "cause": exc.__class__.__name__, "detail": str(exc)},
        )


def record_trade_fill(record: Mapping[str, Any] | Any) -> None:
    """Append a single fill record to the canonical parquet history."""

    payload = _coerce_mapping(record)
    data = _normalise_record(payload)
    try:
        import pandas as pd
    except ImportError:
        global _PANDAS_MISSING_LOGGED
        if not _PANDAS_MISSING_LOGGED:
            logger.warning("TRADE_HISTORY_PANDAS_MISSING")
            _PANDAS_MISSING_LOGGED = True
        return

    existing = _read_parquet(_CANONICAL_PATH)
    new_frame = pd.DataFrame([data])
    if existing is not None and not existing.empty:
        combined = pd.concat([existing, new_frame], ignore_index=True)
        if {"order_id", "fill_id"}.issubset(combined.columns):
            combined = combined.drop_duplicates(subset=["order_id", "fill_id"], keep="last")
    else:
        combined = new_frame
    _write_parquet(_CANONICAL_PATH, combined)


def _extract_attr(obj: Any, *candidates: str) -> Any:
    for name in candidates:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def _broker_rows(broker: Any) -> Iterable[dict[str, Any]]:
    if broker is None:
        return []
    orders: Iterable[Any]
    fetch_methods = (
        "list_orders",
        "get_orders",
        "list_trades",
        "get_fills",
    )
    for name in fetch_methods:
        if hasattr(broker, name):
            try:
                candidate = getattr(broker, name)
                orders = candidate() if name == "get_fills" else candidate(status="all")
                break
            except TypeError:
                try:
                    orders = getattr(broker, name)()
                    break
                except Exception:
                    continue
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug(
                    "TRADE_HISTORY_BROKER_FETCH_FAILED",
                    exc_info=True,
                    extra={"method": name, "detail": str(exc)},
                )
                return []
    else:
        return []

    rows: list[dict[str, Any]] = []
    for order in orders or []:
        status = str(_extract_attr(order, "status") or "").lower()
        if "filled" not in status:
            continue
        qty_raw = _extract_attr(order, "filled_qty", "quantity", "qty")
        price_raw = _extract_attr(order, "filled_avg_price", "average_price", "price")
        timestamp = _extract_attr(order, "filled_at", "executed_at", "updated_at")
        strategy = _extract_attr(order, "strategy", "strategy_id") or ""
        try:
            qty = int(float(qty_raw or 0))
        except (TypeError, ValueError):
            continue
        try:
            price = float(price_raw or 0)
        except (TypeError, ValueError):
            price = None
        if qty <= 0:
            continue
        side_raw = _extract_attr(order, "side") or ""
        side = str(getattr(side_raw, "value", side_raw)).lower()
        order_id = _extract_attr(order, "id", "order_id")
        rows.append(
            {
                "symbol": _extract_attr(order, "symbol") or "",
                "entry_time": timestamp,
                "entry_price": price,
                "qty": qty,
                "side": side,
                "strategy": strategy,
                "signal_tags": "",
                "confidence": 0.0,
                "order_id": order_id,
                "fill_id": _extract_attr(order, "fill_id"),
            }
        )
    return rows


def load_trade_history(
    *, sync_from_broker: bool = False, broker: Any | None = None
) -> tuple["pd.DataFrame" | None, str | None]:
    """Load canonical history with optional broker reconciliation."""

    frame = _read_parquet(_CANONICAL_PATH)
    source: str | None = None
    if frame is not None and not frame.empty:
        source = "canonical"

    broker_frame = None
    if sync_from_broker:
        rows = list(_broker_rows(broker))
        if rows:
            try:
                import pandas as pd
            except ImportError:
                rows = []
            else:
                broker_frame = pd.DataFrame([_normalise_record(r) for r in rows])
                if broker_frame.empty:
                    broker_frame = None

    if broker_frame is not None:
        try:
            import pandas as pd
        except ImportError:
            broker_frame = None
        else:
            if frame is not None and not frame.empty:
                combined = pd.concat([frame, broker_frame], ignore_index=True)
                if {"order_id", "fill_id"}.issubset(combined.columns):
                    combined = combined.drop_duplicates(subset=["order_id", "fill_id"], keep="last")
                frame = combined
                source = "merged"
            else:
                frame = broker_frame
                source = "broker"
            _write_parquet(_CANONICAL_PATH, frame)

    return frame, source


__all__ = ["load_trade_history", "record_trade_fill"]
