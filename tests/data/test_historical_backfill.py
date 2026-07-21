from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd
import pytest

from ai_trading.data.historical_backfill import (
    AlpacaHistoricalBarFetcher,
    HistoricalBackfillIncompleteError,
    HistoricalBackfillSpec,
    build_fetch_windows,
    materialize_historical_backfill,
    normalize_governed_symbols,
)
from ai_trading.data.historical_bars import load_historical_bars
from ai_trading.tools import historical_training_backfill as cli
from ai_trading.utils.market_calendar import session_info


@dataclass
class _Response:
    df: pd.DataFrame


class _Request:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Client:
    def __init__(self, responder: Callable[[Any], pd.DataFrame]) -> None:
        self.responder = responder
        self.calls: list[Any] = []

    def get_stock_bars(self, request: Any) -> _Response:
        self.calls.append(request)
        return _Response(self.responder(request))


def _session_frame(
    symbol: str,
    session_date: date,
    *,
    missing_offsets: set[int] | None = None,
    duplicate_offset: int | None = None,
    include_out_of_session: bool = False,
) -> pd.DataFrame:
    session = session_info(session_date)
    timestamps = pd.date_range(
        start=session.start_utc,
        end=session.end_utc - pd.Timedelta(minutes=1),
        freq="1min",
    )
    missing = missing_offsets or set()
    timestamps = pd.DatetimeIndex(
        [timestamp for offset, timestamp in enumerate(timestamps) if offset not in missing]
    )
    rows = pd.DataFrame(
        {
            "symbol": symbol,
            "timestamp": timestamps,
            "open": [100.0 + (offset / 100.0) for offset in range(len(timestamps))],
            "high": [100.2 + (offset / 100.0) for offset in range(len(timestamps))],
            "low": [99.8 + (offset / 100.0) for offset in range(len(timestamps))],
            "close": [100.1 + (offset / 100.0) for offset in range(len(timestamps))],
            "volume": [1_000 + offset for offset in range(len(timestamps))],
        }
    )
    if duplicate_offset is not None:
        rows = pd.concat([rows, rows.iloc[[duplicate_offset]]], ignore_index=True)
    if include_out_of_session:
        extra = rows.iloc[[0]].copy()
        extra["timestamp"] = pd.Timestamp(session.start_utc) - pd.Timedelta(minutes=1)
        rows = pd.concat([rows, extra], ignore_index=True)
    return rows.set_index(["symbol", "timestamp"])


def _frame_for_request(symbol: str, request: Any) -> pd.DataFrame:
    current = request.start.astimezone(UTC).date()
    end_date = request.end.astimezone(UTC).date()
    frames: list[pd.DataFrame] = []
    while current <= end_date:
        try:
            session = session_info(current)
        except ValueError:
            current += pd.Timedelta(days=1)
            continue
        if session.start_utc >= request.start and session.end_utc <= request.end:
            frames.append(_session_frame(symbol, current))
        current += pd.Timedelta(days=1)
    return pd.concat(frames) if frames else pd.DataFrame()


def _spec(tmp_path: Path, **overrides: Any) -> HistoricalBackfillSpec:
    values: dict[str, Any] = {
        "symbols": ("AAPL",),
        "start_date": date(2025, 8, 20),
        "end_date": date(2025, 8, 20),
        "output_root": tmp_path,
        "window_sessions": 1,
        "max_missing_ratio": 0.0,
        "sdk_version": "0.42.1",
    }
    values.update(overrides)
    return HistoricalBackfillSpec(**values)


def _fetcher(client: _Client) -> AlpacaHistoricalBarFetcher:
    return AlpacaHistoricalBarFetcher(
        client,
        request_factory=_Request,
        minute_timeframe="1Min",
    )


def test_governed_symbols_are_deterministic_and_reject_ungoverned() -> None:
    assert normalize_governed_symbols("msft,AAPL,msft") == ("AAPL", "MSFT")
    with pytest.raises(ValueError, match="GOOGL"):
        normalize_governed_symbols("AAPL,GOOGL")


def test_fetch_windows_are_monotonic_and_honor_early_close() -> None:
    windows = build_fetch_windows(
        date(2024, 11, 29),
        date(2024, 12, 2),
        window_sessions=1,
    )

    assert [window.window_id for window in windows] == [
        "2024-11-29_2024-11-29",
        "2024-12-02_2024-12-02",
    ]
    assert windows[0].expected_rows == 210
    assert windows[0].early_close_dates == (date(2024, 11, 29),)
    assert windows[1].start_utc > windows[0].end_utc


def test_materializes_loader_compatible_csv_and_provenance(tmp_path: Path) -> None:
    client = _Client(lambda request: _frame_for_request("AAPL", request))
    result = materialize_historical_backfill(_spec(tmp_path), fetcher=_fetcher(client))

    assert result.quality_passed is True
    assert len(client.calls) == 1
    request = client.calls[0]
    assert request.symbol_or_symbols == "AAPL"
    assert request.timeframe == "1Min"
    assert request.feed == "iex"
    assert request.adjustment == "raw"
    assert request.limit == 10_000
    assert request.sort == "asc"

    symbol_result = result.symbols[0]
    frame, load_report = load_historical_bars(symbol_result.csv_path)
    assert len(frame) == 390
    assert load_report.timestamp_authoritative is True
    assert load_report.source_providers == ("alpaca",)

    provenance = json.loads(symbol_result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["identity"]["provider"] == "alpaca"
    assert provenance["identity"]["feed"] == "iex"
    assert provenance["identity"]["adjustment"] == "raw"
    assert provenance["identity"]["timeframe"] == "1Min"
    assert provenance["identity"]["symbol"] == "AAPL"
    assert provenance["identity"]["sdk"] == {
        "distribution": "alpaca-py",
        "version": "0.42.1",
    }
    assert provenance["content_sha256"] == symbol_result.content_sha256
    assert provenance["authority"] == {
        "evidence_type": "historical_research",
        "live_money_authority": False,
        "promotion_authority": False,
        "promotion_eligible": False,
        "research_only": True,
        "runtime_fill_authority": False,
    }


def test_rerun_uses_verified_checkpoint_without_refetching(tmp_path: Path) -> None:
    client = _Client(lambda request: _frame_for_request("AAPL", request))
    spec = _spec(tmp_path)
    first = materialize_historical_backfill(spec, fetcher=_fetcher(client))
    first_bytes = first.symbols[0].csv_path.read_bytes()

    second = materialize_historical_backfill(spec, fetcher=_fetcher(client))

    assert len(client.calls) == 1
    assert second.symbols[0].fetched_windows == 0
    assert second.symbols[0].resumed_windows == 1
    assert second.symbols[0].csv_path.read_bytes() == first_bytes
    assert second.symbols[0].content_sha256 == first.symbols[0].content_sha256


def test_feed_and_adjustment_provenance_do_not_share_cache(tmp_path: Path) -> None:
    client = _Client(lambda request: _frame_for_request("AAPL", request))
    iex = materialize_historical_backfill(
        _spec(tmp_path, feed="iex", adjustment="raw"),
        fetcher=_fetcher(client),
    )
    sip = materialize_historical_backfill(
        _spec(tmp_path, feed="sip", adjustment="all"),
        fetcher=_fetcher(client),
    )

    assert iex.dataset_dir != sip.dataset_dir
    assert iex.cache_key != sip.cache_key
    assert len(client.calls) == 2


def test_completeness_reports_duplicates_missing_and_out_of_session(
    tmp_path: Path,
) -> None:
    response = _session_frame(
        "AAPL",
        date(2025, 8, 20),
        missing_offsets={5},
        duplicate_offset=10,
        include_out_of_session=True,
    )
    client = _Client(lambda _request: response)
    result = materialize_historical_backfill(
        _spec(tmp_path, max_missing_ratio=0.01),
        fetcher=_fetcher(client),
    )

    completeness = result.symbols[0].completeness
    assert completeness["expected_rows"] == 390
    assert completeness["observed_rows"] == 389
    assert completeness["missing_rows"] == 1
    assert completeness["duplicate_rows"] == 1
    assert completeness["out_of_session_rows"] == 1
    assert completeness["interpolation_used"] is False
    assert completeness["quality_passed"] is True


def test_completeness_threshold_fails_after_writing_diagnostics(tmp_path: Path) -> None:
    response = _session_frame(
        "AAPL",
        date(2025, 8, 20),
        missing_offsets={5},
    )
    client = _Client(lambda _request: response)

    with pytest.raises(HistoricalBackfillIncompleteError) as error:
        materialize_historical_backfill(
            _spec(tmp_path, max_missing_ratio=0.0),
            fetcher=_fetcher(client),
        )

    symbol_result = error.value.result.symbols[0]
    assert symbol_result.quality_passed is False
    assert symbol_result.csv_path.exists()
    provenance = json.loads(symbol_result.provenance_path.read_text(encoding="utf-8"))
    assert provenance["completeness"]["missing_rows"] == 1
    assert provenance["authority"]["promotion_eligible"] is False


def test_interrupted_run_resumes_only_unfinished_window(tmp_path: Path) -> None:
    spec = _spec(
        tmp_path,
        start_date=date(2025, 8, 20),
        end_date=date(2025, 8, 21),
        window_sessions=1,
    )
    attempts = {"count": 0}

    def _flaky(request: Any) -> pd.DataFrame:
        attempts["count"] += 1
        if attempts["count"] == 2:
            raise TimeoutError("fixture interruption")
        return _frame_for_request("AAPL", request)

    first_client = _Client(_flaky)
    with pytest.raises(TimeoutError, match="fixture interruption"):
        materialize_historical_backfill(spec, fetcher=_fetcher(first_client))
    assert len(first_client.calls) == 2

    resume_client = _Client(lambda request: _frame_for_request("AAPL", request))
    result = materialize_historical_backfill(spec, fetcher=_fetcher(resume_client))

    assert len(resume_client.calls) == 1
    assert resume_client.calls[0].start.astimezone(UTC).date() == date(2025, 8, 21)
    assert result.symbols[0].fetched_windows == 1
    assert result.symbols[0].resumed_windows == 1
    assert result.symbols[0].row_count == 780


def test_corrupt_csv_invalidates_checkpoint_and_refetches(tmp_path: Path) -> None:
    client = _Client(lambda request: _frame_for_request("AAPL", request))
    spec = _spec(tmp_path)
    first = materialize_historical_backfill(spec, fetcher=_fetcher(client))
    first.symbols[0].csv_path.write_text("corrupt,partial\n1", encoding="utf-8")

    recovered = materialize_historical_backfill(spec, fetcher=_fetcher(client))

    assert len(client.calls) == 2
    assert recovered.symbols[0].fetched_windows == 1
    frame, _ = load_historical_bars(recovered.symbols[0].csv_path)
    assert len(frame) == 390


def test_cli_rejects_ungoverned_symbol_before_client_creation(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "_build_alpaca_historical_client",
        lambda: pytest.fail("client must not be created for ungoverned symbols"),
    )

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--symbols",
                "AAPL,GOOGL",
                "--start",
                "2025-08-20",
                "--end",
                "2025-08-20",
            ]
        )

    assert error.value.code == 2


def test_cli_client_hydrates_managed_alpaca_credentials(monkeypatch) -> None:
    hydrated: list[tuple[str, ...]] = []
    constructed: list[dict[str, str]] = []

    class _HistoricalClient:
        def __init__(self, **kwargs: str) -> None:
            constructed.append(kwargs)

    values = {
        "ALPACA_OAUTH": "",
        "ALPACA_API_KEY": "managed-key",
        "ALPACA_SECRET_KEY": "managed-secret",
    }
    monkeypatch.setattr(
        cli,
        "hydrate_managed_secrets",
        lambda *, required_keys: hydrated.append(tuple(required_keys)),
    )
    monkeypatch.setattr(cli, "get_data_client_cls", lambda: _HistoricalClient)
    monkeypatch.setattr(
        cli,
        "get_env",
        lambda key, default="", **_kwargs: values.get(key, default),
    )

    client = cli._build_alpaca_historical_client()

    assert isinstance(client, _HistoricalClient)
    assert hydrated == [("ALPACA_API_KEY", "ALPACA_SECRET_KEY")]
    assert constructed == [
        {"api_key": "managed-key", "secret_key": "managed-secret"}
    ]


def test_dataset_manifest_cannot_claim_runtime_or_promotion_authority(
    tmp_path: Path,
) -> None:
    client = _Client(lambda request: _frame_for_request("AAPL", request))
    result = materialize_historical_backfill(_spec(tmp_path), fetcher=_fetcher(client))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["authority"]["research_only"] is True
    assert manifest["authority"]["evidence_type"] == "historical_research"
    assert manifest["authority"]["promotion_eligible"] is False
    assert manifest["authority"]["promotion_authority"] is False
    assert manifest["authority"]["live_money_authority"] is False
    assert manifest["authority"]["runtime_fill_authority"] is False
    assert "fill" not in result.dataset_dir.parts
    assert "trade" not in result.dataset_dir.parts
