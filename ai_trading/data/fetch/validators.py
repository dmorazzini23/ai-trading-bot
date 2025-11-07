"""Lightweight parameter validators and diagnostics for data fetchers."""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Deque, Dict, Mapping

from ai_trading.logging import get_logger


logger = get_logger(__name__)

VALID_FEEDS = {"iex", "sip", "yahoo", "finnhub"}
VALID_ADJUSTMENTS = {"raw", "split", "all"}


@dataclass(slots=True)
class GapSample:
    """Sampled coverage metrics captured when a gap event occurs."""

    window_start: str | None = None
    window_end: str | None = None
    missing_after: int = 0
    expected: int = 0
    gap_ratio: float = 0.0
    initial_gap_ratio: float = 0.0
    initial_missing: int = 0


@dataclass(slots=True)
class GapStatistics:
    """Aggregated minute-bar gap statistics per symbol."""

    symbol: str
    provider: str | None = None
    gap_ratio: float = 0.0
    missing_after: int = 0
    expected: int = 0
    used_backup: bool = False
    residual_gap: bool = False
    fallback_provider: str | None = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    samples: Deque[GapSample] = field(default_factory=lambda: deque(maxlen=5))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation for tests and logging."""

        return {
            "symbol": self.symbol,
            "provider": self.provider,
            "gap_ratio": self.gap_ratio,
            "missing_after": self.missing_after,
            "expected": self.expected,
            "used_backup": self.used_backup,
            "residual_gap": self.residual_gap,
            "fallback_provider": self.fallback_provider,
            "last_updated": self.last_updated.isoformat(),
            "samples": [asdict(sample) for sample in self.samples],
        }


_GAP_STATS: Dict[str, GapStatistics] = {}


def _normalise_symbol(symbol: str | None) -> str | None:
    if symbol is None:
        return None
    try:
        candidate = symbol.strip()
    except AttributeError:
        return None
    return candidate or None


def validate_feed(feed: str | None) -> None:
    if feed is None:
        return
    if feed not in VALID_FEEDS:
        raise ValueError(f"invalid feed: {feed}")


def validate_adjustment(adj: str | None) -> None:
    if adj is None:
        return
    if adj not in VALID_ADJUSTMENTS:
        raise ValueError(f"invalid adjustment: {adj}")


def record_gap_statistics(symbol: str, metadata: Mapping[str, Any]) -> None:
    """Persist structured coverage diagnostics for ``symbol``."""

    symbol_norm = _normalise_symbol(symbol)
    if not symbol_norm:
        return
    if not isinstance(metadata, Mapping):
        return

    stats = _GAP_STATS.get(symbol_norm)
    if stats is None:
        stats = GapStatistics(symbol=symbol_norm)
        _GAP_STATS[symbol_norm] = stats

    provider_val = metadata.get("provider")
    if isinstance(provider_val, str) and provider_val.strip():
        stats.provider = provider_val.strip()

    fallback_provider_val = metadata.get("fallback_provider")
    if isinstance(fallback_provider_val, str) and fallback_provider_val.strip():
        stats.fallback_provider = fallback_provider_val.strip()

    def _coerce_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    stats.expected = max(_coerce_int(metadata.get("expected")), 0)
    stats.missing_after = max(_coerce_int(metadata.get("missing_after")), 0)
    stats.gap_ratio = max(_coerce_float(metadata.get("gap_ratio")), 0.0)
    stats.used_backup = bool(metadata.get("used_backup"))
    stats.residual_gap = bool(metadata.get("residual_gap"))
    stats.last_updated = datetime.now(UTC)

    start_iso: str | None = None
    end_iso: str | None = None
    window_start = metadata.get("window_start")
    window_end = metadata.get("window_end")
    if isinstance(window_start, datetime):
        start_iso = window_start.astimezone(UTC).isoformat()
    elif isinstance(window_start, str):
        start_iso = window_start
    if isinstance(window_end, datetime):
        end_iso = window_end.astimezone(UTC).isoformat()
    elif isinstance(window_end, str):
        end_iso = window_end

    try:
        initial_missing_val = int(metadata.get("initial_missing", stats.missing_after) or 0)
    except (TypeError, ValueError):
        initial_missing_val = stats.missing_after
    try:
        initial_gap_ratio_val = float(metadata.get("initial_gap_ratio", stats.gap_ratio) or 0.0)
    except (TypeError, ValueError):
        initial_gap_ratio_val = stats.gap_ratio

    stats.samples.append(
        GapSample(
            window_start=start_iso,
            window_end=end_iso,
            missing_after=stats.missing_after,
            expected=stats.expected,
            gap_ratio=stats.gap_ratio,
            initial_gap_ratio=initial_gap_ratio_val,
            initial_missing=initial_missing_val,
        )
    )

    logger.debug(
        "GAP_STATS_RECORDED",
        extra={
            "symbol": symbol_norm,
            "gap_ratio": stats.gap_ratio,
            "missing_after": stats.missing_after,
            "expected": stats.expected,
            "used_backup": stats.used_backup,
            "initial_gap_ratio": initial_gap_ratio_val,
            "initial_missing": initial_missing_val,
        },
    )


def get_gap_statistics(symbol: str | None = None) -> dict[str, dict[str, Any]]:
    """Return captured gap statistics, optionally scoped to ``symbol``."""

    if symbol is not None:
        symbol_norm = _normalise_symbol(symbol)
        if not symbol_norm:
            return {}
        stats = _GAP_STATS.get(symbol_norm)
        return {symbol_norm: stats.as_dict()} if stats else {}
    return {key: value.as_dict() for key, value in _GAP_STATS.items()}


def reset_gap_statistics() -> None:
    """Clear captured gap statistics (used by regression tests)."""

    _GAP_STATS.clear()


__all__ = [
    "VALID_FEEDS",
    "VALID_ADJUSTMENTS",
    "GapStatistics",
    "GapSample",
    "validate_feed",
    "validate_adjustment",
    "record_gap_statistics",
    "get_gap_statistics",
    "reset_gap_statistics",
]
