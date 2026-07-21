"""Governed, research-only historical one-minute bar acquisition.

This module deliberately writes to a provenance-keyed research cache.  It does
not write runtime order, fill, trade, or promotion artifacts.  The materialized
CSV files are compatible with :func:`ai_trading.data.historical_bars.load_historical_bars`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

import pandas as pd

from ai_trading.alpaca_api import (
    get_stock_bars_request_cls,
    get_timeframe_cls,
)
from ai_trading.logging import get_logger
from ai_trading.utils.market_calendar import is_trading_day, session_info


logger = get_logger(__name__)

GOVERNED_HISTORICAL_SYMBOLS: frozenset[str] = frozenset({"AAPL", "AMZN", "MSFT"})
HISTORICAL_EVIDENCE_TYPE = "historical_research"
HISTORICAL_TIMEFRAME = "1Min"
_CSV_COLUMNS: tuple[str, ...] = (
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "data_provider",
    "feed",
    "adjustment",
    "timeframe",
)
_AUTHORITY: dict[str, Any] = {
    "research_only": True,
    "evidence_type": HISTORICAL_EVIDENCE_TYPE,
    "promotion_eligible": False,
    "promotion_authority": False,
    "live_money_authority": False,
    "runtime_fill_authority": False,
}


class HistoricalBarsClient(Protocol):
    """Minimal Alpaca historical-client surface used by the backfill."""

    def get_stock_bars(self, request: Any) -> Any:
        """Return historical bars for ``request``."""


class HistoricalBackfillError(RuntimeError):
    """Base error for governed historical backfill failures."""


class HistoricalBackfillIncompleteError(HistoricalBackfillError):
    """Raised after materialization when completeness exceeds the fail threshold."""

    def __init__(self, result: "HistoricalBackfillResult") -> None:
        self.result = result
        failed = [item.symbol for item in result.symbols if not item.quality_passed]
        super().__init__(
            "Historical backfill completeness failed for: " + ",".join(failed)
        )


@dataclass(frozen=True, slots=True)
class HistoricalBackfillSpec:
    """Immutable acquisition and cache contract."""

    symbols: tuple[str, ...]
    start_date: date
    end_date: date
    output_root: Path
    provider: str = "alpaca"
    feed: str = "iex"
    adjustment: str = "raw"
    timeframe: str = HISTORICAL_TIMEFRAME
    window_sessions: int = 10
    request_limit: int = 10_000
    max_missing_ratio: float = 0.02
    fail_on_incomplete: bool = True
    sdk_version: str | None = None

    def __post_init__(self) -> None:
        normalized_symbols = normalize_governed_symbols(self.symbols)
        object.__setattr__(self, "symbols", normalized_symbols)
        object.__setattr__(self, "output_root", Path(self.output_root).expanduser())
        provider = str(self.provider or "").strip().lower()
        feed = str(self.feed or "").strip().lower()
        adjustment = str(self.adjustment or "").strip().lower()
        timeframe = str(self.timeframe or "").strip()
        if provider != "alpaca":
            raise ValueError("historical backfill provider must be alpaca")
        if not feed:
            raise ValueError("historical backfill feed is required")
        if not adjustment:
            raise ValueError("historical backfill adjustment is required")
        if timeframe.lower() not in {"1min", "1m", "minute"}:
            raise ValueError("historical backfill timeframe must be 1Min")
        if self.start_date > self.end_date:
            raise ValueError("historical backfill start_date must not exceed end_date")
        if not 1 <= int(self.window_sessions) <= 20:
            raise ValueError("window_sessions must be between 1 and 20")
        if not 1 <= int(self.request_limit) <= 10_000:
            raise ValueError("request_limit must be between 1 and 10000")
        if not 0.0 <= float(self.max_missing_ratio) <= 1.0:
            raise ValueError("max_missing_ratio must be between 0 and 1")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "feed", feed)
        object.__setattr__(self, "adjustment", adjustment)
        object.__setattr__(self, "timeframe", HISTORICAL_TIMEFRAME)


@dataclass(frozen=True, slots=True)
class HistoricalFetchWindow:
    """A deterministic, bounded group of NYSE sessions."""

    window_id: str
    session_dates: tuple[date, ...]
    start_utc: datetime
    end_utc: datetime
    expected_rows: int
    early_close_dates: tuple[date, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "session_dates": [item.isoformat() for item in self.session_dates],
            "start_utc": self.start_utc.isoformat(),
            "end_utc": self.end_utc.isoformat(),
            "expected_rows": int(self.expected_rows),
            "early_close_dates": [item.isoformat() for item in self.early_close_dates],
        }


@dataclass(frozen=True, slots=True)
class SymbolBackfillResult:
    symbol: str
    csv_path: Path
    provenance_path: Path
    row_count: int
    content_sha256: str
    fetched_windows: int
    resumed_windows: int
    quality_passed: bool
    completeness: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["csv_path"] = str(self.csv_path)
        payload["provenance_path"] = str(self.provenance_path)
        payload["completeness"] = dict(self.completeness)
        return payload


@dataclass(frozen=True, slots=True)
class HistoricalBackfillResult:
    dataset_dir: Path
    manifest_path: Path
    cache_key: str
    symbols: tuple[SymbolBackfillResult, ...]
    quality_passed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "manifest_path": str(self.manifest_path),
            "cache_key": self.cache_key,
            "symbols": [item.as_dict() for item in self.symbols],
            "quality_passed": bool(self.quality_passed),
            "authority": dict(_AUTHORITY),
        }


def normalize_governed_symbols(symbols: Sequence[str] | str) -> tuple[str, ...]:
    """Return deterministic governed symbols or raise for any ungoverned input."""

    raw_symbols = symbols.split(",") if isinstance(symbols, str) else symbols
    normalized = tuple(
        sorted({str(symbol).strip().upper() for symbol in raw_symbols if str(symbol).strip()})
    )
    if not normalized:
        raise ValueError("at least one governed historical symbol is required")
    ungoverned = sorted(set(normalized) - GOVERNED_HISTORICAL_SYMBOLS)
    if ungoverned:
        raise ValueError(
            "historical backfill symbols must be limited to AAPL,AMZN,MSFT; "
            f"received ungoverned symbols: {','.join(ungoverned)}"
        )
    return normalized


def _alpaca_sdk_version() -> str:
    try:
        return str(importlib_metadata.version("alpaca-py"))
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _iso_utc(value: datetime) -> str:
    normalized = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return normalized.astimezone(UTC).isoformat()


def _session_expected_index(session_date: date) -> tuple[pd.DatetimeIndex, bool]:
    session = session_info(session_date)
    start = pd.Timestamp(session.start_utc)
    end = pd.Timestamp(session.end_utc)
    expected = pd.date_range(start=start, end=end - pd.Timedelta(minutes=1), freq="1min")
    return expected, bool(session.is_early_close)


def build_fetch_windows(
    start_date: date,
    end_date: date,
    *,
    window_sessions: int,
) -> tuple[HistoricalFetchWindow, ...]:
    """Plan monotonic, non-overlapping NYSE-session request windows."""

    if start_date > end_date:
        raise ValueError("start_date must not exceed end_date")
    session_dates: list[date] = []
    cursor = start_date
    while cursor <= end_date:
        if is_trading_day(cursor):
            session_dates.append(cursor)
        cursor += timedelta(days=1)
    windows: list[HistoricalFetchWindow] = []
    chunk_size = max(1, int(window_sessions))
    previous_start: datetime | None = None
    previous_end: datetime | None = None
    for offset in range(0, len(session_dates), chunk_size):
        chunk = tuple(session_dates[offset : offset + chunk_size])
        first = session_info(chunk[0])
        last = session_info(chunk[-1])
        expected_rows = 0
        early_closes: list[date] = []
        for session_date in chunk:
            expected, early = _session_expected_index(session_date)
            expected_rows += int(len(expected))
            if early:
                early_closes.append(session_date)
        if previous_start is not None and first.start_utc <= previous_start:
            raise HistoricalBackfillError("historical fetch window cursor did not progress")
        if previous_end is not None and first.start_utc < previous_end:
            raise HistoricalBackfillError("historical fetch windows overlap")
        window_id = f"{chunk[0].isoformat()}_{chunk[-1].isoformat()}"
        windows.append(
            HistoricalFetchWindow(
                window_id=window_id,
                session_dates=chunk,
                start_utc=first.start_utc,
                end_utc=last.end_utc,
                expected_rows=expected_rows,
                early_close_dates=tuple(early_closes),
            )
        )
        previous_start = first.start_utc
        previous_end = last.end_utc
    return tuple(windows)


class AlpacaHistoricalBarFetcher:
    """Small injectable adapter around Alpaca's historical client."""

    def __init__(
        self,
        client: HistoricalBarsClient,
        *,
        request_factory: Callable[..., Any] | None = None,
        minute_timeframe: Any | None = None,
    ) -> None:
        self.client = client
        self._request_factory = request_factory
        self._minute_timeframe = minute_timeframe

    def _request_components(self) -> tuple[Callable[..., Any], Any]:
        request_factory = self._request_factory or get_stock_bars_request_cls()
        minute_timeframe = self._minute_timeframe
        if minute_timeframe is None:
            timeframe_cls = get_timeframe_cls()
            minute_timeframe = getattr(timeframe_cls, "Minute")
        return request_factory, minute_timeframe

    def fetch_window(
        self,
        *,
        symbol: str,
        window: HistoricalFetchWindow,
        feed: str,
        adjustment: str,
        limit: int,
    ) -> Any:
        request_factory, minute_timeframe = self._request_components()
        request = request_factory(
            symbol_or_symbols=symbol,
            timeframe=minute_timeframe,
            start=window.start_utc,
            end=window.end_utc,
            feed=feed,
            adjustment=adjustment,
            limit=int(limit),
            sort="asc",
        )
        return self.client.get_stock_bars(request)


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce", utc=True)


def _response_frame(response: Any, *, symbol: str) -> pd.DataFrame:
    raw = getattr(response, "df", response)
    if isinstance(raw, pd.DataFrame):
        frame = raw.copy()
    elif isinstance(raw, Mapping):
        symbol_rows = raw.get(symbol) or raw.get(symbol.upper()) or []
        frame = pd.DataFrame(
            [
                dict(row) if isinstance(row, Mapping) else vars(row)
                for row in symbol_rows
            ]
        )
    else:
        try:
            frame = pd.DataFrame(raw)
        except (TypeError, ValueError):
            frame = pd.DataFrame()
    if frame.empty:
        return pd.DataFrame(columns=_CSV_COLUMNS)
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()
    elif not isinstance(frame.index, pd.RangeIndex):
        index_name = str(frame.index.name or "timestamp")
        frame = frame.reset_index(names=index_name)
    frame = frame.rename(columns={column: str(column).strip().lower() for column in frame.columns})
    if "timestamp" not in frame.columns:
        for candidate in ("time", "datetime", "index"):
            if candidate in frame.columns:
                frame = frame.rename(columns={candidate: "timestamp"})
                break
    if "symbol" not in frame.columns:
        frame["symbol"] = symbol
    required = {"timestamp", "open", "high", "low", "close"}
    if not required.issubset(frame.columns):
        missing = ",".join(sorted(required - set(frame.columns)))
        raise HistoricalBackfillError(f"historical response missing columns: {missing}")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    frame = frame.loc[frame["symbol"] == symbol].copy()
    frame["timestamp"] = _parse_timestamp_series(frame["timestamp"])
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "open", "high", "low", "close"])
    positive = (frame[["open", "high", "low", "close"]] > 0.0).all(axis=1)
    return frame.loc[positive, ["symbol", "timestamp", "open", "high", "low", "close", "volume"]].copy()


def _normalize_response(
    response: Any,
    *,
    symbol: str,
    expected_timestamps: pd.DatetimeIndex,
    provider: str,
    feed: str,
    adjustment: str,
    timeframe: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = _response_frame(response, symbol=symbol)
    if frame.empty:
        return pd.DataFrame(columns=_CSV_COLUMNS), {
            "raw_rows": 0,
            "duplicate_rows": 0,
            "out_of_session_rows": 0,
        }
    raw_rows = int(len(frame))
    frame = frame.sort_values("timestamp", kind="mergesort")
    before_dedupe = int(len(frame))
    frame = frame.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    duplicate_rows = before_dedupe - int(len(frame))
    expected_set = set(expected_timestamps)
    in_session = frame["timestamp"].isin(expected_set)
    out_of_session_rows = int((~in_session).sum())
    frame = frame.loc[in_session].copy()
    frame["data_provider"] = provider
    frame["feed"] = feed
    frame["adjustment"] = adjustment
    frame["timeframe"] = timeframe
    frame = frame.loc[:, list(_CSV_COLUMNS)].sort_values("timestamp", kind="mergesort")
    return frame.reset_index(drop=True), {
        "raw_rows": raw_rows,
        "duplicate_rows": duplicate_rows,
        "out_of_session_rows": out_of_session_rows,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.loc[:, list(_CSV_COLUMNS)].to_csv(temp, index=False)
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()
    return _sha256_file(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temp.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def _dataset_identity(spec: HistoricalBackfillSpec, *, sdk_version: str) -> dict[str, Any]:
    return {
        "provider": spec.provider,
        "feed": spec.feed,
        "adjustment": spec.adjustment,
        "timeframe": spec.timeframe,
        "symbols": list(spec.symbols),
        "request_interval": {
            "start_date": spec.start_date.isoformat(),
            "end_date": spec.end_date.isoformat(),
            "end_inclusive": True,
        },
        "sdk": {"distribution": "alpaca-py", "version": sdk_version},
    }


def _symbol_identity(
    dataset_identity: Mapping[str, Any],
    *,
    symbol: str,
) -> dict[str, Any]:
    return {**dict(dataset_identity), "symbol": symbol}


def _empty_cache_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_CSV_COLUMNS)


def _load_valid_cache(
    csv_path: Path,
    provenance_path: Path,
    *,
    expected_identity: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    """Return verified cached rows/state; corrupt or partial pairs recover empty."""

    if not csv_path.exists() or not provenance_path.exists():
        return _empty_cache_frame(), {}, False
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return _empty_cache_frame(), {}, False
    if not isinstance(provenance, Mapping):
        return _empty_cache_frame(), {}, False
    if provenance.get("identity") != dict(expected_identity):
        raise HistoricalBackfillError(
            f"cache provenance mismatch at {provenance_path}"
        )
    expected_hash = str(provenance.get("content_sha256") or "")
    try:
        actual_hash = _sha256_file(csv_path)
    except OSError:
        return _empty_cache_frame(), {}, False
    if not expected_hash or actual_hash != expected_hash:
        return _empty_cache_frame(), {}, False
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, ValueError, pd.errors.ParserError):
        return _empty_cache_frame(), {}, False
    if not set(_CSV_COLUMNS).issubset(frame.columns):
        return _empty_cache_frame(), {}, False
    frame["timestamp"] = _parse_timestamp_series(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"])
    return frame.loc[:, list(_CSV_COLUMNS)].copy(), dict(provenance), True


def _merge_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if existing.empty:
        combined = incoming.copy()
    elif incoming.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, incoming], axis=0, ignore_index=True)
    if combined.empty:
        return _empty_cache_frame(), 0
    combined["timestamp"] = _parse_timestamp_series(combined["timestamp"])
    combined = combined.dropna(subset=["timestamp"])
    before = int(len(combined))
    combined = combined.sort_values(["symbol", "timestamp"], kind="mergesort")
    combined = combined.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    duplicate_rows = before - int(len(combined))
    return combined.loc[:, list(_CSV_COLUMNS)].reset_index(drop=True), duplicate_rows


def _expected_timestamps(windows: Sequence[HistoricalFetchWindow]) -> pd.DatetimeIndex:
    indexes: list[pd.DatetimeIndex] = []
    for window in windows:
        for session_date in window.session_dates:
            expected, _ = _session_expected_index(session_date)
            indexes.append(expected)
    if not indexes:
        return pd.DatetimeIndex([], tz=UTC)
    combined = indexes[0]
    for index in indexes[1:]:
        combined = combined.append(index)
    return combined.drop_duplicates().sort_values()


def _completeness_report(
    frame: pd.DataFrame,
    *,
    windows: Sequence[HistoricalFetchWindow],
    duplicate_rows: int,
    out_of_session_rows: int,
    max_missing_ratio: float,
) -> dict[str, Any]:
    observed_index = pd.DatetimeIndex(
        _parse_timestamp_series(frame["timestamp"]) if not frame.empty else []
    ).dropna().drop_duplicates()
    sessions: list[dict[str, Any]] = []
    expected_total = 0
    observed_total = 0
    missing_total = 0
    early_close_dates: list[str] = []
    for window in windows:
        for session_date in window.session_dates:
            expected, early = _session_expected_index(session_date)
            observed = int(expected.isin(observed_index).sum())
            missing = int(len(expected) - observed)
            expected_total += int(len(expected))
            observed_total += observed
            missing_total += missing
            if early:
                early_close_dates.append(session_date.isoformat())
            sessions.append(
                {
                    "session_date": session_date.isoformat(),
                    "expected_rows": int(len(expected)),
                    "observed_rows": observed,
                    "missing_rows": missing,
                    "is_early_close": bool(early),
                    "complete": missing == 0,
                }
            )
    missing_ratio = float(missing_total / expected_total) if expected_total else 0.0
    quality_passed = bool(expected_total > 0 and missing_ratio <= float(max_missing_ratio))
    return {
        "expected_rows": int(expected_total),
        "observed_rows": int(observed_total),
        "missing_rows": int(missing_total),
        "missing_ratio": missing_ratio,
        "duplicate_rows": int(max(duplicate_rows, 0)),
        "out_of_session_rows": int(max(out_of_session_rows, 0)),
        "session_count": int(len(sessions)),
        "early_close_count": int(len(early_close_dates)),
        "early_close_dates": early_close_dates,
        "max_missing_ratio": float(max_missing_ratio),
        "quality_passed": quality_passed,
        "interpolation_used": False,
        "sessions": sessions,
    }


def _window_expected_timestamps(window: HistoricalFetchWindow) -> pd.DatetimeIndex:
    return _expected_timestamps((window,))


def _provenance_payload(
    *,
    identity: Mapping[str, Any],
    cache_key: str,
    csv_path: Path,
    content_sha256: str,
    row_count: int,
    completed_windows: Sequence[str],
    windows: Sequence[HistoricalFetchWindow],
    window_results: Sequence[Mapping[str, Any]],
    duplicate_rows: int,
    out_of_session_rows: int,
    completeness: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "identity": dict(identity),
        "cache_key": cache_key,
        "csv_path": str(csv_path),
        "fetched_at": datetime.now(UTC).isoformat(),
        "row_count": int(row_count),
        "content_sha256": content_sha256,
        "completed_windows": list(completed_windows),
        "request_windows": [window.as_dict() for window in windows],
        "window_results": [dict(item) for item in window_results],
        "duplicate_rows": int(max(duplicate_rows, 0)),
        "out_of_session_rows": int(max(out_of_session_rows, 0)),
        "completeness": dict(completeness or {}),
        "source_authority": "alpaca_historical_market_data",
        "authority": dict(_AUTHORITY),
    }


def materialize_historical_backfill(
    spec: HistoricalBackfillSpec,
    *,
    fetcher: AlpacaHistoricalBarFetcher,
) -> HistoricalBackfillResult:
    """Fetch, verify, and atomically materialize a provenance-isolated dataset."""

    sdk_version = str(spec.sdk_version or _alpaca_sdk_version())
    identity = _dataset_identity(spec, sdk_version=sdk_version)
    dataset_cache_key = _json_hash(identity)
    dataset_dir = spec.output_root / dataset_cache_key[:20]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    windows = build_fetch_windows(
        spec.start_date,
        spec.end_date,
        window_sessions=spec.window_sessions,
    )
    if not windows:
        raise HistoricalBackfillError("historical backfill range contains no trading sessions")

    symbol_results: list[SymbolBackfillResult] = []
    for symbol in spec.symbols:
        symbol_identity = _symbol_identity(identity, symbol=symbol)
        symbol_cache_key = _json_hash(symbol_identity)
        csv_path = dataset_dir / f"{symbol}.csv"
        provenance_path = dataset_dir / f"{symbol}.provenance.json"
        frame, prior, cache_valid = _load_valid_cache(
            csv_path,
            provenance_path,
            expected_identity=symbol_identity,
        )
        completed_windows = {
            str(item)
            for item in prior.get("completed_windows", ())
            if str(item)
        }
        window_results = [
            dict(item)
            for item in prior.get("window_results", ())
            if isinstance(item, Mapping)
        ]
        duplicate_rows = int(prior.get("duplicate_rows", 0) or 0)
        out_of_session_rows = int(prior.get("out_of_session_rows", 0) or 0)
        fetched_windows = 0
        resumed_windows = 0
        if not cache_valid:
            frame = _empty_cache_frame()
            completed_windows = set()
            window_results = []
            duplicate_rows = 0
            out_of_session_rows = 0

        for window in windows:
            if window.window_id in completed_windows:
                resumed_windows += 1
                continue
            response = fetcher.fetch_window(
                symbol=symbol,
                window=window,
                feed=spec.feed,
                adjustment=spec.adjustment,
                limit=spec.request_limit,
            )
            incoming, fetch_counts = _normalize_response(
                response,
                symbol=symbol,
                expected_timestamps=_window_expected_timestamps(window),
                provider=spec.provider,
                feed=spec.feed,
                adjustment=spec.adjustment,
                timeframe=spec.timeframe,
            )
            frame, merge_duplicates = _merge_frames(frame, incoming)
            duplicate_rows += int(fetch_counts["duplicate_rows"]) + int(merge_duplicates)
            out_of_session_rows += int(fetch_counts["out_of_session_rows"])
            completed_windows.add(window.window_id)
            fetched_windows += 1
            window_results = [
                item
                for item in window_results
                if str(item.get("window_id") or "") != window.window_id
            ]
            window_results.append(
                {
                    "window_id": window.window_id,
                    "request_start_utc": _iso_utc(window.start_utc),
                    "request_end_utc": _iso_utc(window.end_utc),
                    "raw_rows": int(fetch_counts["raw_rows"]),
                    "accepted_rows": int(len(incoming)),
                    "duplicate_rows": int(fetch_counts["duplicate_rows"]),
                    "out_of_session_rows": int(fetch_counts["out_of_session_rows"]),
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            )
            content_sha256 = _atomic_write_csv(csv_path, frame)
            checkpoint = _provenance_payload(
                identity=symbol_identity,
                cache_key=symbol_cache_key,
                csv_path=csv_path,
                content_sha256=content_sha256,
                row_count=len(frame),
                completed_windows=sorted(completed_windows),
                windows=windows,
                window_results=window_results,
                duplicate_rows=duplicate_rows,
                out_of_session_rows=out_of_session_rows,
                completeness=None,
            )
            _atomic_write_json(provenance_path, checkpoint)

        if not csv_path.exists():
            content_sha256 = _atomic_write_csv(csv_path, frame)
        else:
            content_sha256 = _sha256_file(csv_path)
        completeness = _completeness_report(
            frame,
            windows=windows,
            duplicate_rows=duplicate_rows,
            out_of_session_rows=out_of_session_rows,
            max_missing_ratio=spec.max_missing_ratio,
        )
        final_provenance = _provenance_payload(
            identity=symbol_identity,
            cache_key=symbol_cache_key,
            csv_path=csv_path,
            content_sha256=content_sha256,
            row_count=len(frame),
            completed_windows=sorted(completed_windows),
            windows=windows,
            window_results=window_results,
            duplicate_rows=duplicate_rows,
            out_of_session_rows=out_of_session_rows,
            completeness=completeness,
        )
        _atomic_write_json(provenance_path, final_provenance)
        symbol_results.append(
            SymbolBackfillResult(
                symbol=symbol,
                csv_path=csv_path,
                provenance_path=provenance_path,
                row_count=int(len(frame)),
                content_sha256=content_sha256,
                fetched_windows=fetched_windows,
                resumed_windows=resumed_windows,
                quality_passed=bool(completeness["quality_passed"]),
                completeness=completeness,
            )
        )

    quality_passed = all(item.quality_passed for item in symbol_results)
    manifest_path = dataset_dir / "dataset.provenance.json"
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_identity": identity,
        "dataset_cache_key": dataset_cache_key,
        "dataset_dir": str(dataset_dir),
        "symbols": [item.as_dict() for item in symbol_results],
        "quality_passed": quality_passed,
        "authority": dict(_AUTHORITY),
    }
    _atomic_write_json(manifest_path, manifest)
    result = HistoricalBackfillResult(
        dataset_dir=dataset_dir,
        manifest_path=manifest_path,
        cache_key=dataset_cache_key,
        symbols=tuple(symbol_results),
        quality_passed=quality_passed,
    )
    logger.info(
        "HISTORICAL_BACKFILL_MATERIALIZED",
        extra={
            "dataset_dir": str(dataset_dir),
            "cache_key": dataset_cache_key,
            "symbols": list(spec.symbols),
            "quality_passed": quality_passed,
            "rows": sum(item.row_count for item in symbol_results),
            "research_only": True,
            "promotion_authority": False,
        },
    )
    if not quality_passed and spec.fail_on_incomplete:
        raise HistoricalBackfillIncompleteError(result)
    return result


__all__ = [
    "AlpacaHistoricalBarFetcher",
    "GOVERNED_HISTORICAL_SYMBOLS",
    "HISTORICAL_EVIDENCE_TYPE",
    "HISTORICAL_TIMEFRAME",
    "HistoricalBackfillError",
    "HistoricalBackfillIncompleteError",
    "HistoricalBackfillResult",
    "HistoricalBackfillSpec",
    "HistoricalBarsClient",
    "HistoricalFetchWindow",
    "SymbolBackfillResult",
    "build_fetch_windows",
    "materialize_historical_backfill",
    "normalize_governed_symbols",
]
