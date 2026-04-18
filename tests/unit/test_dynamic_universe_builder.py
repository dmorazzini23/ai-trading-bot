from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from ai_trading.data import dynamic_universe as du
from ai_trading.data.alpaca_screener import MarketMover, MarketMoversSnapshot


def _runtime_with_assets(asset_map: dict[str, dict], price_map: dict[str, tuple[float, float]]):
    class DummyAPI:
        def get_asset(self, symbol: str):
            return asset_map[symbol]

    class DummyFetcher:
        def get_daily_df(self, runtime, symbol: str):
            price, volume = price_map[symbol]
            return pd.DataFrame(
                [{"open": price, "high": price, "low": price, "close": price, "volume": volume}]
            )

    return SimpleNamespace(api=DummyAPI(), data_fetcher=DummyFetcher())


def test_build_dynamic_universe_prepends_and_dedupes(monkeypatch, tmp_path):
    runtime = _runtime_with_assets(
        asset_map={
            "AAPL": {"tradable": True, "marginable": True, "shortable": True, "easy_to_borrow": True},
            "NVDA": {"tradable": True, "marginable": True, "shortable": True, "easy_to_borrow": True},
            "TSLA": {"tradable": True, "marginable": True, "shortable": True, "easy_to_borrow": True},
        },
        price_map={
            "AAPL": (180.0, 2_000_000),
            "NVDA": (900.0, 1_000_000),
            "TSLA": (170.0, 900_000),
        },
    )
    monkeypatch.setattr(
        du,
        "fetch_market_movers",
        lambda *args, **kwargs: MarketMoversSnapshot(
            gainers=[
                MarketMover(symbol="NVDA", percent_change=5.0, change=20.0, price=900.0),
                MarketMover(symbol="AAPL", percent_change=2.0, change=3.0, price=180.0),
            ],
            losers=[MarketMover(symbol="TSLA", percent_change=-4.0, change=-7.0, price=170.0)],
            market_type="stocks",
            last_updated=pd.Timestamp("2026-04-17T15:10:00Z").to_pydatetime(),
            used_fallback=False,
        ),
    )
    monkeypatch.setattr(du, "_short_overlay_enabled", lambda: True)
    config = du.DynamicUniverseConfig(
        enabled=True,
        refresh_sec=300,
        gainers_top=2,
        losers_top=1,
        min_price=5.0,
        min_dollar_volume=5_000_000.0,
        min_volume=100_000.0,
        prepend=True,
        snapshot_path=str(tmp_path / "dynamic_universe.jsonl"),
    )

    result = du.build_dynamic_universe(runtime, ["AAPL", "MSFT"], config=config)

    assert result.merged_symbols == ["NVDA", "TSLA", "AAPL", "MSFT"]
    assert [candidate.symbol for candidate in result.additions] == ["NVDA", "TSLA"]
    snapshot_rows = (tmp_path / "dynamic_universe.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert snapshot_rows
    payload = json.loads(snapshot_rows[-1])
    assert payload["dynamic_symbols"] == ["NVDA", "TSLA"]


def test_build_dynamic_universe_skips_short_overlay_when_disabled(monkeypatch, tmp_path):
    runtime = _runtime_with_assets(
        asset_map={
            "NVDA": {"tradable": True, "marginable": True, "shortable": True, "easy_to_borrow": True},
            "TSLA": {"tradable": True, "marginable": True, "shortable": True, "easy_to_borrow": True},
        },
        price_map={"NVDA": (900.0, 1_000_000), "TSLA": (170.0, 900_000)},
    )
    monkeypatch.setattr(
        du,
        "fetch_market_movers",
        lambda *args, **kwargs: MarketMoversSnapshot(
            gainers=[MarketMover(symbol="NVDA", percent_change=5.0, change=20.0, price=900.0)],
            losers=[MarketMover(symbol="TSLA", percent_change=-4.0, change=-7.0, price=170.0)],
            market_type="stocks",
            last_updated=pd.Timestamp("2026-04-17T15:15:00Z").to_pydatetime(),
            used_fallback=False,
        ),
    )
    monkeypatch.setattr(du, "_short_overlay_enabled", lambda: False)
    config = du.DynamicUniverseConfig(
        enabled=True,
        gainers_top=1,
        losers_top=1,
        snapshot_path=str(tmp_path / "dynamic_universe.jsonl"),
    )

    result = du.build_dynamic_universe(runtime, ["MSFT"], config=config)

    assert result.merged_symbols == ["NVDA", "MSFT"]
    assert [candidate.symbol for candidate in result.additions] == ["NVDA"]


def test_build_dynamic_universe_filters_non_etb_shorts(monkeypatch, tmp_path):
    runtime = _runtime_with_assets(
        asset_map={
            "TSLA": {"tradable": True, "marginable": True, "shortable": True, "easy_to_borrow": False},
        },
        price_map={"TSLA": (170.0, 900_000)},
    )
    monkeypatch.setattr(
        du,
        "fetch_market_movers",
        lambda *args, **kwargs: MarketMoversSnapshot(
            gainers=[],
            losers=[MarketMover(symbol="TSLA", percent_change=-4.0, change=-7.0, price=170.0)],
            market_type="stocks",
            last_updated=pd.Timestamp("2026-04-17T15:20:00Z").to_pydatetime(),
            used_fallback=False,
        ),
    )
    monkeypatch.setattr(du, "_short_overlay_enabled", lambda: True)
    config = du.DynamicUniverseConfig(
        enabled=True,
        gainers_top=0,
        losers_top=1,
        require_etb_shorts=True,
        snapshot_path=str(tmp_path / "dynamic_universe.jsonl"),
    )

    result = du.build_dynamic_universe(runtime, ["MSFT"], config=config)

    assert result.merged_symbols == ["MSFT"]
    assert result.additions == []
