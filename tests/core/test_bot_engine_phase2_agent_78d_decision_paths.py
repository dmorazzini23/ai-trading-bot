from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from ai_trading.core import bot_engine


def _patch_get_env(monkeypatch: pytest.MonkeyPatch, values: dict[str, Any]) -> None:
    def _fake_get_env(key: str, default: Any = None, *, cast: Any = None, **_: Any) -> Any:
        value = values.get(key, default)
        if isinstance(value, BaseException):
            raise value
        if cast is not None and value is not None:
            return cast(value)
        return value

    monkeypatch.setattr(bot_engine, "get_env", _fake_get_env)


def _bars_df(close: float = 101.0) -> pd.DataFrame:
    idx = pd.date_range("2026-04-20", periods=2, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": [close - 2.0, close - 1.0],
            "high": [close + 1.0, close + 2.0],
            "low": [close - 3.0, close - 2.0],
            "close": [close - 0.5, close],
            "volume": [1000, 1200],
        },
        index=idx,
    )


def _fetcher() -> bot_engine.DataFetcher:
    fetcher = object.__new__(bot_engine.DataFetcher)
    fetcher.prefer = None
    fetcher.force_feed = None
    fetcher.settings = SimpleNamespace(
        alpaca_api_key="key",
        alpaca_secret_key_plain="secret",
        alpaca_execution_feed="iex",
        alpaca_reference_feed="iex",
        alpaca_adjustment=None,
        data_daily_fetch_min_interval_s=60.0,
    )
    fetcher._daily_cache = {}
    fetcher._daily_cache_reference = {}
    fetcher._minute_cache = {}
    fetcher._minute_cache_reference = {}
    fetcher._minute_timestamps = {}
    fetcher._minute_timestamps_reference = {}
    fetcher._warn_seen = {}
    fetcher._daily_cache_hit_logged = False
    fetcher._daily_error_state = {}
    return fetcher


def test_adaptive_order_cap_exercises_budget_headroom_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_env(
        monkeypatch,
        {
            "AI_TRADING_ADAPTIVE_ORDER_CAP_ENABLED": True,
            "AI_TRADING_INTERVAL": 60.0,
            "INTERVAL_WHEN_CLOSED": 30.0,
            "AI_TRADING_ADAPTIVE_ORDER_CAP_WARN_HEADROOM_RATIO": 0.5,
            "AI_TRADING_ADAPTIVE_ORDER_CAP_CRIT_HEADROOM_RATIO": 0.2,
            "AI_TRADING_ADAPTIVE_ORDER_CAP_WARN_VALUE": 4,
            "AI_TRADING_ADAPTIVE_ORDER_CAP_CRIT_VALUE": 2,
        },
    )
    budget = SimpleNamespace(remaining=lambda: 6.0)
    cycle_budget = SimpleNamespace(interval_s=60.0, budget=budget)

    cap, details = bot_engine._resolve_adaptive_order_cap(
        cycle_budget=cycle_budget,
        last_loop_duration_s=55.0,
    )

    assert cap == 2
    assert details["mode"] == "critical"
    assert details["headroom_ratio"] == pytest.approx(5.0 / 60.0)

    cap, details = bot_engine._resolve_adaptive_order_cap(
        cycle_budget=SimpleNamespace(interval_s=60.0, budget=SimpleNamespace(remaining=lambda: 40.0)),
        last_loop_duration_s=40.0,
    )
    assert cap == 4
    assert details["mode"] == "warning"

    cap, details = bot_engine._resolve_adaptive_order_cap(
        cycle_budget=None,
        last_loop_duration_s=1.0,
    )
    assert cap is None
    assert details["mode"] == "normal"


def test_slo_derisk_mode_relaxes_only_friction_breaches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_env(
        monkeypatch,
        {
            "AI_TRADING_DERISK_SLO_SCALE_MULT": 0.4,
            "AI_TRADING_DERISK_SLO_BLOCK_RELAX_ENABLED": True,
            "AI_TRADING_DERISK_SLO_BLOCK_RELAX_SCALE_MULT": 0.65,
            "AI_TRADING_DERISK_SLO_BLOCK_RELAX_PACING_SEVERE_PCT": 80.0,
            "AI_TRADING_DERISK_SLO_BLOCK_RELAX_PENDING_SEVERE_SEC": 600.0,
            "AI_TRADING_DERISK_SLO_BLOCK_RELAX_MAX_FRICTION_BREACHES": 2,
        },
    )

    mode, scale, details = bot_engine._resolve_slo_derisk_effective_mode(
        configured_mode="adaptive",
        reject_breached=False,
        drift_breached=False,
        slippage_breached=False,
        calibration_ece_breached=False,
        calibration_brier_breached=False,
        feature_drift_breached=False,
        label_drift_breached=False,
        residual_drift_breached=False,
        pacing_breached=True,
        pending_breached=False,
        pacing_hit_rate_pct=25.0,
        pending_oldest_age_sec=0.0,
    )
    assert (mode, scale) == ("scale", 0.65)
    assert details["block_relax_reason"] == "friction_breach"

    mode, scale, details = bot_engine._resolve_slo_derisk_effective_mode(
        configured_mode="adaptive",
        reject_breached=True,
        drift_breached=False,
        slippage_breached=False,
        calibration_ece_breached=False,
        calibration_brier_breached=False,
        feature_drift_breached=False,
        label_drift_breached=False,
        residual_drift_breached=False,
        pacing_breached=True,
        pending_breached=False,
        pacing_hit_rate_pct=25.0,
        pending_oldest_age_sec=0.0,
    )
    assert (mode, scale) == ("block", 1.0)
    assert details["block_relax_reason"] == "core_breach"

    mode, scale, details = bot_engine._resolve_slo_derisk_effective_mode(
        configured_mode="block",
        reject_breached=False,
        drift_breached=False,
        slippage_breached=False,
        calibration_ece_breached=False,
        calibration_brier_breached=False,
        feature_drift_breached=False,
        label_drift_breached=False,
        residual_drift_breached=False,
        pacing_breached=False,
        pending_breached=True,
        pacing_hit_rate_pct=0.0,
        pending_oldest_age_sec=1200.0,
    )
    assert (mode, scale) == ("block", 1.0)
    assert details["block_relax_reason"] == "friction_severe"


def test_capacity_throttle_adaptive_modes_and_exception_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_env(
        monkeypatch,
        {
            "AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_ENABLED": ValueError("missing"),
            "AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_MIN_SAMPLES": ValueError("missing"),
            "AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_PACING_CLEAR_PCT": ValueError("missing"),
            "AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_PACING_TIGHTEN_PCT": ValueError("missing"),
            "AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_REJECT_CLEAR_PCT": ValueError("missing"),
            "AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_RELAX_MULT": ValueError("missing"),
            "AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_TIGHTEN_MULT": ValueError("missing"),
            "AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_RELAX_MIN_SCALE_ADD": ValueError("missing"),
            "AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_TIGHTEN_MIN_SCALE_MULT": ValueError("missing"),
            "AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_PENDING_SOFT_SEC": ValueError("missing"),
            "AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_PENDING_HARD_SEC": ValueError("missing"),
            "AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_SLIPPAGE_SOFT_BPS": ValueError("missing"),
            "AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_SLIPPAGE_HARD_BPS": ValueError("missing"),
            "AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_STRESS_TIGHTEN_MULT": ValueError("missing"),
            "AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_STRESS_MIN_SCALE_MULT": ValueError("missing"),
        },
    )

    *_, min_scale, details = bot_engine._resolve_capacity_throttle_adaptive_params(
        spread_soft_bps=10.0,
        spread_hard_bps=20.0,
        volume_soft_participation=0.1,
        volume_hard_participation=0.2,
        min_scale=0.5,
        slo_derisk_details={
            "pacing_samples": 15,
            "pending_samples": 1,
            "pending_oldest_age_sec": 300.0,
            "slippage_samples": 15,
            "slippage_bps": 20.0,
        },
    )
    assert details["mode"] == "tightened"
    assert details["microstructure_severe"] is True
    assert min_scale < 0.5

    _patch_get_env(monkeypatch, {"AI_TRADING_CAPACITY_THROTTLE_ADAPTIVE_ENABLED": False})
    result = bot_engine._resolve_capacity_throttle_adaptive_params(
        spread_soft_bps=10.0,
        spread_hard_bps=20.0,
        volume_soft_participation=0.1,
        volume_hard_participation=0.2,
        min_scale=0.5,
        slo_derisk_details={},
    )
    assert result[-1]["mode"] == "disabled"
    assert result[:5] == (10.0, 20.0, 0.1, 0.2, 0.5)


def test_data_fetcher_daily_memo_paths_short_circuit_without_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetcher = _fetcher()
    df = _bars_df()
    start = datetime(2026, 4, 1, tzinfo=UTC)
    end = datetime(2026, 4, 24, 20, 0, tzinfo=UTC)
    canonical_key = ("AAPL", "1Day", start.isoformat(), end.isoformat())
    memo: dict[tuple[str, ...], Any] = {canonical_key: {"timestamp": 900.0, "df": df}}

    monkeypatch.setattr(bot_engine, "_DAILY_FETCH_MEMO", memo)
    monkeypatch.setattr(bot_engine, "_DAILY_FETCH_MEMO_TTL", 300.0)
    monkeypatch.setattr(bot_engine.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "get_daily_df",
        lambda *args, **kwargs: pytest.fail("memo should avoid provider"),
    )

    result, meta = fetcher.get_daily_df(SimpleNamespace(cfg=None), "aapl", return_meta=True)

    assert result is df
    assert meta == {"memo": True}
    assert memo[("AAPL", end.date().isoformat())][1] is df


def test_data_fetcher_daily_additional_lookup_iterator_and_cache_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetcher = _fetcher()
    df = _bars_df()
    memo: dict[tuple[str, ...], Any] = {("daily", "AAPL"): iter([(990.0, df)])}
    fetch_date = datetime.now(UTC).date()

    monkeypatch.setattr(bot_engine, "_DAILY_FETCH_MEMO", memo)
    monkeypatch.setattr(bot_engine, "_DAILY_FETCH_MEMO_TTL", 300.0)
    monkeypatch.setattr(bot_engine.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "get_daily_df",
        lambda *args, **kwargs: pytest.fail("additional memo should avoid provider"),
    )

    assert fetcher.get_daily_df(SimpleNamespace(cfg=None), "AAPL") is df

    memo.clear()
    fetcher._daily_cache["AAPL"] = (fetch_date, df)
    assert fetcher.get_daily_df(SimpleNamespace(cfg=None), "AAPL") is df


def test_data_fetcher_daily_provider_session_cache_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetcher = _fetcher()
    df = _bars_df()
    fetch_date = datetime.now(UTC).date()
    provider_key = ("execution", "iex", "alpaca_iex", fetch_date.isoformat(), "AAPL")

    monkeypatch.setattr(bot_engine, "_DAILY_FETCH_MEMO", {})
    monkeypatch.setattr(bot_engine.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(bot_engine, "is_market_open", lambda: True)
    monkeypatch.setitem(bot_engine._DAILY_PROVIDER_SESSION_CACHE, provider_key, (df, 990.0))
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "get_daily_df",
        lambda *args, **kwargs: pytest.fail("session cache should avoid provider"),
    )

    assert fetcher.get_daily_df(SimpleNamespace(cfg=None), "AAPL") is df


def test_data_fetcher_stock_bar_legacy_and_sanitizing_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, tuple[str, ...]]] = []

    def fetch_with_late_signature(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append((len(args), tuple(sorted(kwargs))))
        if "symbol" in kwargs or len(args) < 3:
            raise TypeError("unexpected keyword argument")
        return _bars_df()

    assert bot_engine.DataFetcher._legacy_fetch_stock_bars(
        fetch_with_late_signature,
        object(),
        object(),
        "AAPL",
        "legacy",
    ).equals(_bars_df())
    assert calls[0][1] == ("context", "symbol")

    raw = SimpleNamespace(
        df=pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5]})
    )
    sanitized = bot_engine.DataFetcher._sanitize_legacy_stock_bars(raw, context="test")
    assert str(sanitized.index.tz) == "UTC"
    assert sanitized.index.name == "timestamp"

    monkeypatch.setattr(bot_engine.bars, "safe_get_stock_bars", None)
    monkeypatch.setattr(
        bot_engine.bars,
        "get_stock_bars",
        lambda *args, **kwargs: (_ for _ in ()).throw(TypeError("positional argument mismatch")),
        raising=False,
    )
    with pytest.raises(AttributeError, match="fallback incompatible"):
        bot_engine.DataFetcher._call_stock_bars(object(), object(), "AAPL", "ctx")


def test_data_fetcher_get_stock_bars_handles_provider_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetcher = _fetcher()
    returned = _bars_df()

    monkeypatch.setattr(bot_engine, "_parse_timeframe", lambda timeframe: timeframe)
    monkeypatch.setattr(
        bot_engine.bars,
        "StockBarsRequest",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        bot_engine.DataFetcher,
        "_call_stock_bars",
        staticmethod(lambda client, request, symbol, context: returned),
    )
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "normalize_ohlcv_columns",
        lambda frame: frame,
    )
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "normalize_ohlcv_df",
        lambda frame: frame,
    )

    multi = fetcher._get_stock_bars(
        "sip",
        ["AAPL", "MSFT"],
        datetime(2026, 4, 1, tzinfo=UTC),
        datetime(2026, 4, 2, tzinfo=UTC),
        "1Day",
        client=object(),
    )
    assert multi is returned

    single = fetcher._get_stock_bars(
        "alpaca_iex",
        "AAPL",
        datetime(2026, 4, 1, tzinfo=UTC),
        datetime(2026, 4, 2, tzinfo=UTC),
        "1Day",
        client=object(),
    )
    assert list(single.columns) == ["open", "high", "low", "close", "volume"]

    with pytest.raises(ValueError, match="symbol list"):
        fetcher._get_stock_bars("alpaca", [], None, None, "1Day", client=object())
    with pytest.raises(ValueError, match="Unsupported provider"):
        fetcher._get_stock_bars("yahoo", "AAPL", None, None, "1Day", client=object())


def test_fetch_quote_unwraps_nested_symbol_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    quote = SimpleNamespace(bid_price=100.0, ask_price=100.2)
    request_seen: list[Any] = []

    class Client:
        def get_stock_latest_quote(self, request: Any) -> Any:
            request_seen.append(request)
            return {"quotes": {"aapl": quote}}

    monkeypatch.setattr(bot_engine, "_stock_quote_request_ready", lambda: True)
    monkeypatch.setattr(
        bot_engine,
        "StockLatestQuoteRequest",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    result = bot_engine._fetch_quote(SimpleNamespace(data_client=Client()), "AAPL", feed="iex")

    assert result is quote
    assert request_seen[0].feed == "iex"


@contextlib.contextmanager
def _span(*_: Any, **__: Any):
    yield


def _patch_safe_submit_basics(monkeypatch: pytest.MonkeyPatch, *, pytest_running: bool = True) -> None:
    _patch_get_env(
        monkeypatch,
        {
            "PYTEST_RUNNING": pytest_running,
            "PYTEST_CURRENT_TEST": "test" if pytest_running else "",
        },
    )
    monkeypatch.setattr(bot_engine, "_ensure_alpaca_classes", lambda: None)
    monkeypatch.setattr(bot_engine, "get_trading_config", lambda: SimpleNamespace(rth_only=True, allow_extended=False))
    monkeypatch.setattr(bot_engine, "_kill_switch_active", lambda cfg: (False, None))
    monkeypatch.setattr(bot_engine, "execution_span", lambda *args, **kwargs: _span())
    monkeypatch.setattr(bot_engine, "monotonic_time", lambda: 10.0)
    monkeypatch.setattr(bot_engine.time, "sleep", lambda seconds: None)


def test_price_simple_primary_trade_and_invalid_feed_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_env(monkeypatch, {"ALPACA_EXECUTION_FEED": "", "ALPACA_DATA_FEED": "", "ALPACA_SIP_UNAUTHORIZED": ""})
    monkeypatch.setattr(bot_engine, "_pytest_running", lambda: True)
    monkeypatch.setattr(bot_engine, "is_alpaca_service_available", lambda: True)
    monkeypatch.setattr(bot_engine.data_fetcher_module, "is_primary_provider_enabled", lambda: True)
    monkeypatch.setattr(bot_engine, "_prefer_feed_this_cycle", lambda: None)
    monkeypatch.setattr(bot_engine, "_get_intraday_feed", lambda: "iex")
    monkeypatch.setattr(bot_engine.price_quote_feed, "resolve", lambda symbol, feed: feed or "iex")
    monkeypatch.setattr(bot_engine, "_get_price_provider_order", lambda: ("alpaca_quote", "alpaca_trade", "yahoo", "bars"))
    monkeypatch.setattr(bot_engine, "_live_execution_blocks_yahoo_fallback", lambda: False)
    monkeypatch.setattr(bot_engine, "_attempt_alpaca_quote", lambda symbol, feed, cache: (None, "alpaca_quote_empty"))
    monkeypatch.setattr(bot_engine, "_resolve_cached_quote_bid", lambda symbol, cache: None)
    monkeypatch.setattr(bot_engine, "_attempt_alpaca_trade", lambda symbol, feed, cache: (101.5, "alpaca_trade"))

    assert bot_engine._get_latest_price_simple("AAPL") == 101.5
    assert bot_engine._PRICE_SOURCE["AAPL"] == "alpaca_trade"

    monkeypatch.setattr(bot_engine, "_attempt_yahoo_price", lambda symbol: (None, "yahoo_error"))
    monkeypatch.setattr(bot_engine, "_attempt_bars_price", lambda symbol: (22.0, "latest_close_used"))
    assert bot_engine._get_latest_price_simple("MSFT", feed="bad-feed") == 22.0
    assert bot_engine._PRICE_SOURCE["MSFT"] == "latest_close_used"


def test_price_simple_provider_disabled_and_backup_frame_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_env(monkeypatch, {"ALPACA_EXECUTION_FEED": "", "ALPACA_DATA_FEED": "", "ALPACA_SIP_UNAUTHORIZED": ""})
    monkeypatch.setattr(bot_engine, "_pytest_running", lambda: False)
    monkeypatch.setattr(bot_engine, "is_alpaca_service_available", lambda: False)
    monkeypatch.setattr(bot_engine.data_fetcher_module, "is_primary_provider_enabled", lambda: True)
    monkeypatch.setattr(bot_engine, "_prefer_feed_this_cycle", lambda: None)
    monkeypatch.setattr(bot_engine, "_get_intraday_feed", lambda: "iex")
    monkeypatch.setattr(bot_engine, "_sip_lockout_active", lambda: False)
    monkeypatch.setattr(bot_engine, "_get_price_provider_order", lambda: ("alpaca_quote",))
    monkeypatch.setattr(bot_engine, "_live_execution_blocks_yahoo_fallback", lambda: True)

    assert bot_engine._get_latest_price_simple("AAPL") is None
    assert bot_engine._PRICE_SOURCE["AAPL"] == "alpaca_disabled"

    monkeypatch.setattr(bot_engine, "is_alpaca_service_available", lambda: True)
    monkeypatch.setattr(bot_engine.data_fetcher_module, "is_primary_provider_enabled", lambda: False)
    assert bot_engine._get_latest_price_simple("MSFT") is None
    assert bot_engine._PRICE_SOURCE["MSFT"] == bot_engine._ALPACA_DISABLED_SENTINEL


def test_pre_rank_execution_candidates_quality_filter_and_exploration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_env(
        monkeypatch,
        {
            "AI_TRADING_EXEC_CANDIDATE_TOP_N": 3,
            "AI_TRADING_EXEC_CANDIDATE_TOP_N_ADAPTIVE_ENABLED": True,
            "AI_TRADING_EXEC_CANDIDATE_TOP_N_PER_ORDER_CAP": 2,
            "AI_TRADING_EXEC_CANDIDATE_TOP_N_ADAPTIVE_MIN": 2,
            "AI_TRADING_EXEC_OPPORTUNITY_QUALITY_ENABLED": True,
            "AI_TRADING_EXEC_OPPORTUNITY_TOP_QUANTILE": 0.5,
            "AI_TRADING_EXEC_OPPORTUNITY_MIN_KEEP": 4,
            "AI_TRADING_EXEC_CANDIDATE_TOP_N_EXPLORATION_ENABLED": True,
            "AI_TRADING_EXEC_CANDIDATE_TOP_N_EXPLORATION_FRAC": 0.5,
            "AI_TRADING_EXEC_CANDIDATE_TOP_N_EXPLORATION_MIN": 1,
            "AI_TRADING_EXEC_CANDIDATE_TOP_N_EXPLORATION_STALE_CYCLES": 2,
            "AI_TRADING_ML_SHADOW_ENABLED": False,
        },
    )
    runtime = SimpleNamespace(
        execution_engine=SimpleNamespace(_resolve_order_submit_cap=lambda: (2, "budget")),
        execution_candidate_rank={"AAPL": 0.9, "MSFT": 0.8, "NVDA": 0.7, "TSLA": 0.6},
        execution_opportunity_quality_by_symbol={"AAPL": 0.9, "MSFT": 0.8, "NVDA": 0.4, "TSLA": 0.7},
        portfolio_weights={"AAPL": 1, "MSFT": "bad", "NVDA": 0.5, "TSLA": 0.1},
        _execution_prerank_cycle_idx=4,
        _execution_candidate_last_selected_cycle={"TSLA": 1},
    )

    selected = bot_engine._pre_rank_execution_candidates(
        ["aapl", "msft", "nvda", "tsla", "aapl"],
        runtime=runtime,
    )

    assert len(selected) == 3
    assert "AAPL" in selected
    assert runtime._execution_candidate_last_selected_cycle


def test_pre_rank_prefers_explicit_underfilled_governed_sampling_strata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_env(
        monkeypatch,
        {
            "AI_TRADING_EXEC_CANDIDATE_TOP_N": 3,
            "AI_TRADING_EXEC_CANDIDATE_TOP_N_ADAPTIVE_ENABLED": False,
            "AI_TRADING_EXEC_OPPORTUNITY_QUALITY_ENABLED": False,
            "AI_TRADING_ML_SHADOW_ENABLED": False,
        },
    )
    monkeypatch.setattr(bot_engine, "_paper_sampling_runtime_active", lambda: True)
    monkeypatch.setattr(
        bot_engine,
        "paper_sampling_deficit_snapshot",
        lambda _cfg: {
            "active": True,
            "fairness_enabled": True,
            "date": "2026-07-20",
            "session_bucket": "midday",
            "configured_symbols": ["AAPL", "AMZN", "MSFT"],
            "priority_symbols": ["AAPL", "AMZN"],
            "priority_reason": "symbol_deficit",
        },
    )
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(),
        execution_candidate_rank={
            "MSFT": 0.99,
            "NVDA": 0.95,
            "AAPL": 0.80,
            "AMZN": 0.70,
        },
    )

    selected = bot_engine._pre_rank_execution_candidates(
        ["MSFT", "NVDA", "AAPL", "AMZN"],
        runtime=runtime,
    )

    assert selected == ["AAPL", "AMZN", "MSFT"]


def test_pre_rank_never_reintroduces_quality_rejected_underfilled_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_env(
        monkeypatch,
        {
            "AI_TRADING_EXEC_CANDIDATE_TOP_N": 2,
            "AI_TRADING_EXEC_CANDIDATE_TOP_N_ADAPTIVE_ENABLED": False,
            "AI_TRADING_EXEC_OPPORTUNITY_QUALITY_ENABLED": True,
            "AI_TRADING_EXEC_OPPORTUNITY_TOP_QUANTILE": 0.75,
            "AI_TRADING_EXEC_OPPORTUNITY_MIN_KEEP": 2,
            "AI_TRADING_EXEC_CANDIDATE_TOP_N_EXPLORATION_ENABLED": False,
            "AI_TRADING_ML_SHADOW_ENABLED": False,
        },
    )
    monkeypatch.setattr(bot_engine, "_paper_sampling_runtime_active", lambda: True)
    monkeypatch.setattr(
        bot_engine,
        "paper_sampling_deficit_snapshot",
        lambda _cfg: {
            "active": True,
            "fairness_enabled": True,
            "date": "2026-07-20",
            "session_bucket": "midday",
            "configured_symbols": ["AAPL", "AMZN", "MSFT"],
            "priority_symbols": ["AAPL", "AMZN"],
            "priority_reason": "symbol_deficit",
        },
    )
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(),
        execution_candidate_rank={
            "AAPL": 0.99,
            "AMZN": 0.98,
            "MSFT": 0.90,
            "NVDA": 0.80,
            "TSLA": 0.70,
        },
        execution_opportunity_quality_by_symbol={
            "AAPL": 0.0,
            "AMZN": 0.1,
            "MSFT": 0.9,
            "NVDA": 0.8,
            "TSLA": 0.7,
        },
    )

    selected = bot_engine._pre_rank_execution_candidates(
        ["AAPL", "AMZN", "MSFT", "NVDA", "TSLA"],
        runtime=runtime,
    )

    assert selected == ["MSFT", "NVDA"]
    assert "AAPL" not in selected
    assert "AMZN" not in selected


def test_pre_rank_reserves_balanced_governed_sampling_strata_before_top_n(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_env(
        monkeypatch,
        {
            "AI_TRADING_EXEC_CANDIDATE_TOP_N": 3,
            "AI_TRADING_EXEC_CANDIDATE_TOP_N_ADAPTIVE_ENABLED": False,
            "AI_TRADING_EXEC_OPPORTUNITY_QUALITY_ENABLED": False,
            "AI_TRADING_ML_SHADOW_ENABLED": False,
        },
    )
    monkeypatch.setattr(bot_engine, "_paper_sampling_runtime_active", lambda: True)
    monkeypatch.setattr(
        bot_engine,
        "paper_sampling_deficit_snapshot",
        lambda _cfg: {
            "active": True,
            "fairness_enabled": True,
            "date": "2026-08-10",
            "session_bucket": "opening",
            "configured_symbols": ["AAPL", "AMZN", "MSFT"],
            "priority_symbols": [],
            "ranked_underfilled_symbols": ["MSFT", "AAPL", "AMZN"],
            "priority_reason": "balanced",
        },
    )
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(),
        execution_candidate_rank={
            "SPY": 1.0,
            "QQQ": 0.99,
            "NVDA": 0.98,
            "AAPL": 0.60,
            "AMZN": 0.50,
            "MSFT": 0.40,
        },
    )

    selected = bot_engine._pre_rank_execution_candidates(
        ["SPY", "QQQ", "NVDA", "AAPL", "AMZN", "MSFT"],
        runtime=runtime,
    )

    assert selected == ["MSFT", "AAPL", "AMZN"]


def test_pre_rank_non_paper_ordering_does_not_load_sampling_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_env(
        monkeypatch,
        {
            "AI_TRADING_EXEC_CANDIDATE_TOP_N": 3,
            "AI_TRADING_EXEC_CANDIDATE_TOP_N_ADAPTIVE_ENABLED": False,
            "AI_TRADING_EXEC_OPPORTUNITY_QUALITY_ENABLED": False,
            "AI_TRADING_ML_SHADOW_ENABLED": False,
        },
    )
    monkeypatch.setattr(bot_engine, "_paper_sampling_runtime_active", lambda: False)

    def _unexpected_snapshot(_cfg: Any) -> dict[str, Any]:
        raise AssertionError("non-paper ranking must not load sampling state")

    monkeypatch.setattr(
        bot_engine,
        "paper_sampling_deficit_snapshot",
        _unexpected_snapshot,
    )
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(),
        execution_candidate_rank={"MSFT": 0.9, "AAPL": 0.8, "AMZN": 0.7},
    )

    selected = bot_engine._pre_rank_execution_candidates(
        ["AAPL", "AMZN", "MSFT"],
        runtime=runtime,
    )

    assert selected == ["MSFT", "AAPL", "AMZN"]


def test_get_latest_price_primary_quote_and_degraded_bid_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_env(monkeypatch, {"ALPACA_EXECUTION_FEED": "", "ALPACA_DATA_FEED": ""})
    monkeypatch.setattr(bot_engine, "_pytest_running", lambda: True)
    monkeypatch.setattr(bot_engine.data_fetcher_module, "is_primary_provider_enabled", lambda: True)
    monkeypatch.setattr(bot_engine, "is_alpaca_service_available", lambda: True)
    monkeypatch.setattr(bot_engine, "_prefer_feed_this_cycle", lambda: None)
    monkeypatch.setattr(bot_engine, "_prefer_feed_this_cycle_helper", lambda symbol: None)
    monkeypatch.setattr(bot_engine, "_get_intraday_feed", lambda: "iex")
    monkeypatch.setattr(bot_engine, "_get_price_provider_order", lambda: ("alpaca_quote", "yahoo", "bars"))
    monkeypatch.setattr(bot_engine, "_live_execution_blocks_yahoo_fallback", lambda: False)
    monkeypatch.setattr(bot_engine, "_should_flag_delayed_slippage", lambda cache, source: False)

    def quote_ask(symbol: str, feed: str | None, cache: dict[str, Any]) -> tuple[None, str]:
        cache["quote_attempted"] = True
        cache["quote_values"] = {"alpaca_ask": 101.0}
        return None, "alpaca_quote"

    monkeypatch.setattr(bot_engine, "_attempt_alpaca_quote", quote_ask)
    assert bot_engine.get_latest_price("AAPL") == 101.0
    assert bot_engine.get_price_source("AAPL") == "alpaca_ask"

    def quote_bid(symbol: str, feed: str | None, cache: dict[str, Any]) -> tuple[None, str]:
        cache["quote_attempted"] = True
        cache["quote_ask_unusable"] = True
        cache["quote_values"] = {"alpaca_bid": 99.0, "alpaca_ask": None, "alpaca_last": None}
        return None, "alpaca_quote"

    monkeypatch.setattr(bot_engine, "_attempt_alpaca_quote", quote_bid)
    assert bot_engine.get_latest_price("MSFT") == 99.0
    assert bot_engine.get_price_source("MSFT") == "alpaca_bid"


def test_get_latest_price_disabled_invalid_and_deferred_backup_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_get_env(monkeypatch, {"ALPACA_EXECUTION_FEED": "", "ALPACA_DATA_FEED": ""})
    monkeypatch.setattr(bot_engine, "_pytest_running", lambda: False)
    monkeypatch.setattr(bot_engine, "_sip_lockout_active", lambda: False)
    monkeypatch.setattr(bot_engine, "_prefer_feed_this_cycle", lambda: None)
    monkeypatch.setattr(bot_engine, "_prefer_feed_this_cycle_helper", lambda symbol: None)
    monkeypatch.setattr(bot_engine, "_get_intraday_feed", lambda: "iex")
    monkeypatch.setattr(bot_engine, "_live_execution_blocks_yahoo_fallback", lambda: False)
    monkeypatch.setattr(bot_engine, "_should_flag_delayed_slippage", lambda cache, source: False)
    monkeypatch.setattr(bot_engine, "_resolve_cached_quote_bid", lambda symbol, cache: None)

    monkeypatch.setattr(bot_engine.data_fetcher_module, "is_primary_provider_enabled", lambda: False)
    assert bot_engine.get_latest_price("AAPL") is None
    assert bot_engine.get_price_source("AAPL") == bot_engine._ALPACA_DISABLED_SENTINEL

    monkeypatch.setattr(bot_engine.data_fetcher_module, "is_primary_provider_enabled", lambda: True)
    monkeypatch.setattr(bot_engine, "is_alpaca_service_available", lambda: True)
    monkeypatch.setattr(bot_engine, "_get_price_provider_order", lambda: ("alpaca_trade", "yahoo", "bars"))
    monkeypatch.setattr(bot_engine, "_attempt_alpaca_trade", lambda symbol, feed, cache: (None, "alpaca_empty"))
    monkeypatch.setattr(bot_engine, "_attempt_yahoo_price", lambda symbol: (55.0, "yahoo"))
    assert bot_engine.get_latest_price("MSFT") == 55.0
    assert bot_engine.get_price_source("MSFT") == "yahoo"

    monkeypatch.setattr(bot_engine, "_get_price_provider_order", lambda: ("alpaca_quote", "bars"))
    monkeypatch.setattr(bot_engine, "_attempt_yahoo_price", lambda symbol: (None, "yahoo_error"))
    monkeypatch.setattr(bot_engine, "_attempt_bars_price", lambda symbol: (44.0, "latest_close_used"))
    assert bot_engine.get_latest_price("NVDA", feed="bad-feed") == 44.0
    assert bot_engine.get_price_source("NVDA") == "latest_close_used"


def test_sentiment_fetch_cache_circuit_rate_limit_and_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_trading.config.settings as settings_mod
    from ai_trading.analysis import sentiment as sentiment_mod

    settings = SimpleNamespace(
        sentiment_api_key="key",
        azure_language_key=None,
        news_api_key=None,
        sentiment_api_url="https://news.example.test",
    )
    monkeypatch.setattr(settings_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(sentiment_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(bot_engine, "get_news_api_key", lambda: None)
    monkeypatch.setattr(bot_engine.pytime, "time", lambda: 1000.0)
    monkeypatch.setattr(sentiment_mod, "_sentiment_initialized", True)
    monkeypatch.setattr(sentiment_mod, "_device", "cpu")
    monkeypatch.setattr(sentiment_mod, "_check_sentiment_circuit_breaker", lambda: True)
    monkeypatch.setattr(sentiment_mod, "_record_sentiment_success", lambda: None)
    monkeypatch.setattr(sentiment_mod, "_record_sentiment_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sentiment_mod,
        "analyze_text",
        lambda text: {"available": True, "pos": 0.75, "neg": 0.25, "neu": 0.0},
    )
    monkeypatch.setattr(
        sentiment_mod,
        "fetch_form4_filings",
        lambda ticker: [{"type": "buy", "dollar_amount": 60_000}],
    )
    bot_engine._SENTIMENT_CACHE.clear()
    sentiment_mod._sentiment_cache.clear()
    bot_engine._SENTIMENT_CIRCUIT_BREAKER["state"] = "closed"
    sentiment_mod._sentiment_circuit_breaker["state"] = "closed"

    class Response:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"articles": [{"title": "Good", "description": "Strong demand"}]}

    monkeypatch.setattr(bot_engine.http, "get", lambda *args, **kwargs: Response())
    assert bot_engine._fetch_sentiment_ctx(SimpleNamespace(), "AAPL") == pytest.approx(0.42)

    bot_engine._SENTIMENT_CACHE["AAPL"] = (995.0, 0.7)
    assert bot_engine._fetch_sentiment_ctx(SimpleNamespace(), "AAPL") == 0.7

    class RateLimited(Response):
        status_code = 429

    bot_engine._SENTIMENT_CACHE.clear()
    sentiment_mod._sentiment_cache.clear()
    monkeypatch.setattr(bot_engine.http, "get", lambda *args, **kwargs: RateLimited())
    assert bot_engine._fetch_sentiment_ctx(SimpleNamespace(), "MSFT") == 0.0
    assert bot_engine._SENTIMENT_CACHE["MSFT"][1] == 0.0

    bot_engine._SENTIMENT_CACHE["TSLA"] = (999.0, -0.3)
    sentiment_mod._sentiment_cache["TSLA"] = (999.0, -0.3)
    monkeypatch.setattr(sentiment_mod, "_check_sentiment_circuit_breaker", lambda: False)
    assert bot_engine._fetch_sentiment_ctx(SimpleNamespace(), "TSLA") == -0.3


def test_sentiment_missing_key_and_request_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_trading.config.settings as settings_mod
    from ai_trading.analysis import sentiment as sentiment_mod

    missing_settings = SimpleNamespace(
        sentiment_api_key=None,
        azure_language_key=None,
        news_api_key=None,
        sentiment_api_url="https://news.example.test",
    )
    monkeypatch.setattr(settings_mod, "get_settings", lambda: missing_settings)
    monkeypatch.setattr(sentiment_mod, "get_settings", lambda: missing_settings)
    monkeypatch.setattr(bot_engine, "get_news_api_key", lambda: None)
    with pytest.raises(RuntimeError, match="missing_api_key"):
        bot_engine._fetch_sentiment_ctx(SimpleNamespace(), "AAPL")

    settings = SimpleNamespace(
        sentiment_api_key="key",
        azure_language_key=None,
        news_api_key=None,
        sentiment_api_url="https://news.example.test",
    )
    monkeypatch.setattr(settings_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(sentiment_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(bot_engine.pytime, "time", lambda: 2000.0)
    monkeypatch.setattr(bot_engine, "_check_sentiment_circuit_breaker", lambda: True)
    monkeypatch.setattr(bot_engine, "_record_sentiment_failure", lambda *args, **kwargs: None)
    bot_engine._SENTIMENT_CACHE.clear()
    sentiment_mod._sentiment_cache.clear()

    def raise_request(*args: Any, **kwargs: Any) -> None:
        raise bot_engine.requests.exceptions.RequestException("down")

    monkeypatch.setattr(bot_engine.http, "get", raise_request)
    assert bot_engine._fetch_sentiment_ctx(SimpleNamespace(), "MSFT") == 0.0
    assert bot_engine._SENTIMENT_CACHE["MSFT"][1] == 0.0


def test_bot_engine_sentiment_wrapper_preserves_canonical_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_trading.analysis import sentiment as sentiment_mod

    bot_engine._SENTIMENT_CACHE.clear()
    monkeypatch.setattr(
        sentiment_mod,
        "fetch_sentiment",
        lambda _ctx, _ticker: (_ for _ in ()).throw(
            RuntimeError("Sentiment unavailable: missing_api_key")
        ),
    )
    monkeypatch.setattr(sentiment_mod, "_sentiment_fail_closed", lambda: True)

    with pytest.raises(RuntimeError, match="missing_api_key"):
        bot_engine.fetch_sentiment(SimpleNamespace(), "AAPL")


def _minute_frame(end: datetime, rows: int, *, provider: str = "alpaca_iex", feed: str = "iex") -> pd.DataFrame:
    index = pd.date_range(end - timedelta(minutes=rows + 1), periods=rows, freq="min", tz=UTC)
    frame = pd.DataFrame(
        {
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.5] * rows,
            "volume": [1000] * rows,
        },
        index=index,
    )
    frame.attrs.update({"data_provider": provider, "data_feed": feed, "asset_class": "equity"})
    return frame


def test_fetch_minute_df_safe_uncached_recovers_low_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_trading.utils.market_calendar as calendar_mod
    import ai_trading.utils.base as base_mod

    now = datetime.now(UTC)
    session_start = now - timedelta(minutes=5)
    monkeypatch.setattr(base_mod, "is_market_open", lambda: True)
    monkeypatch.setattr(calendar_mod, "rth_session_utc", lambda day: (session_start, now))
    monkeypatch.setattr(calendar_mod, "previous_trading_session", lambda day: day - timedelta(days=1))
    monkeypatch.setattr(bot_engine, "_minute_data_freshness_limit", lambda: 300.0)
    monkeypatch.setattr(bot_engine, "_count_trading_minutes", lambda start, end: 4)
    monkeypatch.setattr(bot_engine.staleness, "_ensure_data_fresh", lambda *args, **kwargs: None)
    monkeypatch.setattr(bot_engine, "_sip_authorized", lambda: True)
    monkeypatch.setattr(bot_engine.data_fetcher_module, "_sip_configured", lambda: True)
    monkeypatch.setattr(bot_engine.provider_monitor, "is_disabled", lambda provider: False)
    monkeypatch.setattr(bot_engine, "asset_class_for", lambda symbol: "equity")
    monkeypatch.setattr(bot_engine, "_record_coverage_provider", lambda feed: None)
    monkeypatch.setattr(bot_engine, "_cache_cycle_fallback_feed", lambda feed, symbol=None: None)
    monkeypatch.setattr(
        bot_engine.data_fetcher_module,
        "_cache_fallback",
        lambda symbol, feed: None,
        raising=False,
    )
    monkeypatch.setattr(bot_engine.CFG, "data_feed", "iex", raising=False)
    monkeypatch.setattr(bot_engine.CFG, "minute_gap_backfill", "auto", raising=False)
    monkeypatch.setattr(bot_engine.CFG, "longest_intraday_indicator_minutes", 1, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "intraday_indicator_window_minutes", 1, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "intraday_lookback_minutes", 4, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "market_cache_enabled", False, raising=False)
    monkeypatch.setattr(bot_engine.CFG, "alpaca_feed_failover", ("sip",), raising=False)
    monkeypatch.setattr(bot_engine, "state", SimpleNamespace(minute_feed_cache={}, minute_feed_cache_ts={}))

    calls: list[dict[str, Any]] = []

    def minute_fetch(symbol: str, start: datetime, end: datetime, **kwargs: Any) -> pd.DataFrame:
        calls.append(kwargs)
        if kwargs.get("feed") == "sip":
            return _minute_frame(end, 4, provider="alpaca_sip", feed="sip")
        return _minute_frame(end, 1)

    monkeypatch.setattr(bot_engine, "get_minute_df", minute_fetch)

    result = bot_engine._fetch_minute_df_safe_uncached("AAPL")

    assert len(result) == 4
    assert any(call.get("feed") == "sip" for call in calls)

