from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

pd = pytest.importorskip("pandas")

from ai_trading.meta_learning import persistence as mp


@dataclass
class _Fill:
    symbol: str
    entry_time: str
    entry_price: str
    qty: str
    side: str


def test_record_normalization_dedup_and_pickle_sidecar(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = tmp_path / "history.parquet"
    monkeypatch.setattr(mp, "_CANONICAL_PATH", path)
    monkeypatch.setattr(mp, "_pytest_active", lambda: True)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, *_args, **_kwargs: (_ for _ in ()).throw(ImportError("no engine")))

    mp.record_trade_fill(
        _Fill(
            symbol="AAPL",
            entry_time="2026-04-27T16:00:00+00:00",
            entry_price="100.5",
            qty="3",
            side="buy",
        )
    )
    mp.record_trade_fill(
        {
            "symbol": "AAPL",
            "entry_time": "2026-04-27T16:00:00+00:00",
            "entry_price": "100.5",
            "qty": "3",
            "side": "buy",
            "order_id": "order-1",
            "fill_id": "fill-1",
            "exit_time": "bad-time",
        }
    )
    mp.record_trade_fill(
        {
            "symbol": "AAPL",
            "entry_time": "2026-04-27T16:00:00+00:00",
            "entry_price": "100.5",
            "qty": "3",
            "side": "buy",
            "order_id": "order-1",
            "fill_id": "fill-1",
            "confidence": "0.7",
        }
    )

    sidecar = mp._pickle_sidecar_path(path)  # noqa: SLF001
    assert sidecar.exists()
    loaded = pd.read_pickle(sidecar)
    normalized = mp._drop_duplicate_fills(mp._normalise_frame(loaded))  # noqa: SLF001
    assert len(normalized) == 1
    assert normalized["qty"].iloc[0] == 3
    assert normalized["entry_price"].iloc[0] == 100.5


def test_read_parquet_and_broker_merge(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = tmp_path / "history.parquet"
    expected = pd.DataFrame([{"symbol": "MSFT", "qty": 2, "entry_time": "2026-04-27"}])
    path.write_bytes(b"PAR1")
    monkeypatch.setattr(mp, "_CANONICAL_PATH", path)
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: expected.copy())
    monkeypatch.setattr(pd.DataFrame, "to_parquet", lambda self, *_args, **_kwargs: None)

    frame = mp._read_parquet(path)  # noqa: SLF001
    assert frame is not None
    assert frame["symbol"].iloc[0] == "MSFT"

    orders = [
        SimpleNamespace(status="filled", filled_qty="4", filled_avg_price="101.2", filled_at="2026-04-27T17:00:00+00:00", side="buy", symbol="AAPL", id="o1", fill_id="f1"),
        SimpleNamespace(status="new", filled_qty="4", filled_avg_price="101.2", side="buy", symbol="SKIP"),
        SimpleNamespace(status="filled", filled_qty="0", filled_avg_price="101.2", side="buy", symbol="ZERO"),
    ]
    broker = SimpleNamespace(list_orders=lambda status="all": orders)

    merged, source = mp.load_trade_history(sync_from_broker=True, broker=broker)

    assert source == "merged"
    assert merged is not None
    assert set(merged["symbol"]) == {"MSFT", "AAPL"}
    assert list(mp._broker_rows(None)) == []  # noqa: SLF001
    assert list(mp._broker_rows(SimpleNamespace())) == []  # noqa: SLF001


def test_pickle_migration_helpers_are_explicit(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    path = tmp_path / "legacy.parquet"
    pd.DataFrame([{"symbol": "OLD"}]).to_pickle(path)
    assert mp._looks_like_pickle(path) is True  # noqa: SLF001
    assert mp._is_parquet_path(path) is True  # noqa: SLF001
    assert mp._pickle_sidecar_path(path).name == "legacy.parquet.pkl"  # noqa: SLF001

    monkeypatch.setattr(mp, "_pytest_active", lambda: False)
    monkeypatch.delenv("AI_TRADING_ALLOW_TRUSTED_PICKLE_TRADE_HISTORY_MIGRATION", raising=False)
    assert mp._read_pickle_for_explicit_migration(path, pd) is None  # noqa: SLF001

    monkeypatch.setenv("AI_TRADING_ALLOW_TRUSTED_PICKLE_TRADE_HISTORY_MIGRATION", "1")
    migrated = mp._read_pickle_for_explicit_migration(path, pd)  # noqa: SLF001
    assert migrated is not None
    assert migrated["symbol"].iloc[0] == "OLD"
