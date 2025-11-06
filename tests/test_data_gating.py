from datetime import UTC, datetime, timedelta
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import ai_trading.core.bot_engine as bot_engine
from ai_trading.core.bot_engine import BotState
from ai_trading.config.runtime import TradingConfig


@pytest.fixture(autouse=True)
def clear_gating_caches():
    bot_engine._strict_data_gating_enabled.cache_clear()
    bot_engine._gap_ratio_gate_limit.cache_clear()
    bot_engine._fallback_gap_ratio_limit.cache_clear()
    bot_engine._fallback_quote_max_age_seconds.cache_clear()
    bot_engine._liquidity_fallback_cap.cache_clear()
    yield
    bot_engine._strict_data_gating_enabled.cache_clear()
    bot_engine._gap_ratio_gate_limit.cache_clear()
    bot_engine._fallback_gap_ratio_limit.cache_clear()
    bot_engine._fallback_quote_max_age_seconds.cache_clear()
    bot_engine._liquidity_fallback_cap.cache_clear()


def _fresh_state() -> BotState:
    state = BotState()
    state.price_reliability.clear()
    state.data_quality.clear()
    return state


def test_gap_ratio_gate_blocks(monkeypatch):
    monkeypatch.setenv("AI_TRADING_STRICT_GATING", "1")
    monkeypatch.setenv("AI_TRADING_GAP_LIMIT_BPS", "200")
    monkeypatch.setenv("AI_TRADING_FALLBACK_GAP_LIMIT_BPS", "200")
    state = _fresh_state()
    state.price_reliability["NFLX"] = (
        False,
        "gap_ratio=4.60%>limit=2.00%",
    )
    state.data_quality["NFLX"] = {
        "gap_ratio": 0.046,
        "price_reliable": False,
        "price_reliable_reason": "gap_ratio=4.60%>limit=2.00%",
    }
    ctx = SimpleNamespace(data_client=None, liquidity_annotations={})
    monkeypatch.setattr(
        bot_engine,
        "_check_fallback_quote_age",
        lambda *_, **__: (True, 0.0, None),
    )

    decision = bot_engine._evaluate_data_gating(
        ctx,
        state,
        "NFLX",
        "yahoo_close",
        prefer_backup_quote=True,
    )

    assert decision.block
    assert any("gap_ratio" in reason for reason in decision.reasons)


def test_gap_ratio_relaxed_for_fallback(monkeypatch):
    monkeypatch.setenv("AI_TRADING_STRICT_GATING", "1")
    monkeypatch.setenv("AI_TRADING_GAP_LIMIT_BPS", "50")
    monkeypatch.setenv("AI_TRADING_FALLBACK_GAP_LIMIT_BPS", "500")
    state = _fresh_state()
    state.price_reliability["XYZ"] = (
        False,
        "gap_ratio=1.80%>limit=0.50%",
    )
    state.data_quality["XYZ"] = {
        "gap_ratio": 0.018,
        "price_reliable": False,
        "price_reliable_reason": "gap_ratio=1.80%>limit=0.50%",
    }
    ctx = SimpleNamespace(data_client=None, liquidity_annotations={})
    monkeypatch.setattr(
        bot_engine,
        "_check_fallback_quote_age",
        lambda *_, **__: (False, None, "quote_timestamp_missing"),
    )

    decision = bot_engine._evaluate_data_gating(
        ctx,
        state,
        "XYZ",
        "yahoo_close",
        prefer_backup_quote=True,
    )

    assert not decision.block
    assert decision.annotations.get("gap_limit_primary") == pytest.approx(0.005)
    assert decision.annotations.get("gap_limit") == pytest.approx(0.05)
    assert decision.annotations.get("gap_ratio_relaxed") is True
    assert any("gap_ratio" in reason for reason in decision.reasons)
    assert "quote_timestamp_missing" in decision.reasons
    quality = state.data_quality["XYZ"]
    assert quality["price_reliable"] is True
    assert quality.get("fallback_gap_relaxed") is True


def test_fallback_gap_floor_relaxes_price_reliability(monkeypatch):
    monkeypatch.setenv("AI_TRADING_STRICT_GATING", "1")
    monkeypatch.setenv("AI_TRADING_GAP_LIMIT_BPS", "50")
    monkeypatch.setenv("AI_TRADING_FALLBACK_GAP_LIMIT_BPS", "200")
    state = _fresh_state()
    state.price_reliability["XYZ"] = (
        False,
        "gap_ratio=2.00%>limit=0.50%",
    )
    state.data_quality["XYZ"] = {
        "gap_ratio": 0.02,
        "price_reliable": False,
        "price_reliable_reason": "gap_ratio=2.00%>limit=0.50%",
    }
    ctx = SimpleNamespace(data_client=None, liquidity_annotations={})
    monkeypatch.setattr(
        bot_engine,
        "_check_fallback_quote_age",
        lambda *_, **__: (True, 0.25, None),
    )

    decision = bot_engine._evaluate_data_gating(
        ctx,
        state,
        "XYZ",
        "yahoo_close",
        prefer_backup_quote=True,
    )

    assert bot_engine._gap_ratio_gate_limit() == pytest.approx(0.005)
    assert bot_engine._fallback_gap_ratio_limit() == pytest.approx(0.05)
    assert not decision.block
    assert decision.annotations.get("gap_limit") == pytest.approx(0.05)
    assert decision.annotations.get("gap_limit_primary") == pytest.approx(0.005)
    assert decision.annotations.get("gap_limit_relaxed") == pytest.approx(0.05)
    assert decision.annotations.get("gap_ratio_relaxed") is True
    assert decision.annotations.get("fallback_quote_ok") is True
    quality = state.data_quality["XYZ"]
    assert quality["price_reliable"] is True
    assert quality.get("price_reliable_reason") is None
    assert quality.get("fallback_gap_relaxed") is True


def test_fallback_gap_ratio_prefers_trading_config(monkeypatch):
    # Ensure legacy environment values are present but should be ignored.
    monkeypatch.setenv("AI_TRADING_GAP_LIMIT_BPS", "20")
    monkeypatch.setenv("AI_TRADING_FALLBACK_GAP_LIMIT_BPS", "30")
    cfg = TradingConfig(
        gap_ratio_limit=0.01,
        gap_limit_bps=120,
        fallback_gap_limit_bps=900,
    )
    monkeypatch.setattr(bot_engine, "get_trading_config", lambda: cfg)
    bot_engine._fallback_gap_ratio_limit.cache_clear()

    result = bot_engine._fallback_gap_ratio_limit()

    assert result == pytest.approx(0.09)


def test_synthetic_fallback_quote_used_when_alpaca_unavailable(monkeypatch):
    monkeypatch.setenv("AI_TRADING_STRICT_GATING", "1")
    state = _fresh_state()
    symbol = "XYZ"
    last_bar = datetime.now(UTC) - timedelta(seconds=4)
    quality_entry = {
        "gap_ratio": 0.0,
        "price_reliable": True,
        "provider_canonical": "yahoo",
        "fallback_contiguous": True,
        "coverage_last_timestamp": last_bar,
    }
    state.data_quality[symbol] = quality_entry
    state.price_reliability[symbol] = (True, None)

    ctx = SimpleNamespace(data_client=None, liquidity_annotations={})
    monkeypatch.setattr(
        bot_engine,
        "_check_fallback_quote_age",
        lambda *_args, **_kwargs: (False, None, "quote_source_unavailable"),
    )

    decision = bot_engine._evaluate_data_gating(
        ctx,
        state,
        symbol,
        "yahoo_close",
        prefer_backup_quote=True,
    )

    assert decision.block is False
    assert decision.annotations.get("fallback_quote_source") == "synthetic"
    assert "quote_source_unavailable" not in decision.reasons
    quality_meta = state.data_quality[symbol]
    assert quality_meta.get("fallback_quote_source") == "synthetic"
    assert quality_meta.get("fallback_quote_error") is None


def test_missing_ohlcv_blocks(monkeypatch):
    monkeypatch.setenv("AI_TRADING_STRICT_GATING", "1")
    state = _fresh_state()
    state.price_reliability["MSFT"] = (False, "ohlcv_columns_missing")
    state.data_quality["MSFT"] = {
        "missing_ohlcv": True,
        "price_reliable": False,
        "price_reliable_reason": "ohlcv_columns_missing",
    }
    ctx = SimpleNamespace(data_client=None, liquidity_annotations={})
    monkeypatch.setattr(
        bot_engine,
        "_check_fallback_quote_age",
        lambda *_, **__: (True, 0.0, None),
    )

    decision = bot_engine._evaluate_data_gating(
        ctx,
        state,
        "MSFT",
        "yahoo_close",
        prefer_backup_quote=True,
    )

    assert decision.block
    assert "ohlcv_columns_missing" in decision.reasons


def test_liquidity_fallback_cap_applies(monkeypatch):
    monkeypatch.setenv("AI_TRADING_STRICT_GATING", "1")
    monkeypatch.setenv("AI_TRADING_LIQ_FALLBACK_CAP", "0.20")
    state = _fresh_state()
    ctx = SimpleNamespace(
        data_client=None,
        liquidity_annotations={"XYZ": {"fallback": True, "factor": 0.2}},
    )
    monkeypatch.setattr(
        bot_engine,
        "_check_fallback_quote_age",
        lambda *_, **__: (True, 0.0, None),
    )

    decision = bot_engine._evaluate_data_gating(
        ctx,
        state,
        "XYZ",
        "alpaca_quote",
        prefer_backup_quote=False,
    )

    assert not decision.block
    assert decision.size_cap == pytest.approx(0.2)
    assert "liquidity_fallback" in decision.reasons


def test_fallback_quote_age_blocks_when_stale(monkeypatch):
    monkeypatch.setenv("AI_TRADING_STRICT_GATING", "1")
    monkeypatch.setenv("AI_TRADING_FALLBACK_QUOTE_MAX_AGE_SEC", "1")

    class StubClient:
        def get_stock_latest_quote(self, _request):
            return SimpleNamespace(
                timestamp=datetime.now(UTC) - timedelta(seconds=30),
                ask_price=101.0,
                bid_price=100.0,
            )

    state = _fresh_state()
    state.data_quality["ABC"] = {"price_reliable": True}
    ctx = SimpleNamespace(data_client=StubClient(), liquidity_annotations={})

    decision = bot_engine._evaluate_data_gating(
        ctx,
        state,
        "ABC",
        "yahoo_close",
        prefer_backup_quote=True,
    )

    assert decision.block
    assert "fallback_quote_stale" in decision.reasons
    assert decision.annotations.get("fallback_quote_error") == "fallback_quote_stale"
    assert decision.annotations.get("fallback_quote_ok") is False
    assert decision.annotations.get("fallback_quote_limit") == pytest.approx(1.0)


def test_gap_limit_env_override(monkeypatch):
    monkeypatch.setenv("AI_TRADING_STRICT_GATING", "1")
    monkeypatch.setenv("AI_TRADING_GAP_LIMIT_BPS", "150")
    monkeypatch.setenv("AI_TRADING_FALLBACK_GAP_LIMIT_BPS", "150")
    state = _fresh_state()
    state.price_reliability["NFLX"] = (
        False,
        "gap_ratio=2.00%>limit=1.50%",
    )
    state.data_quality["NFLX"] = {
        "gap_ratio": 0.02,
        "price_reliable": True,
        "price_reliable_reason": "gap_ratio=2.00%>limit=1.50%",
    }
    ctx = SimpleNamespace(data_client=None, liquidity_annotations={})
    monkeypatch.setattr(
        bot_engine,
        "_check_fallback_quote_age",
        lambda *_, **__: (True, 0.5, None),
    )

    decision = bot_engine._evaluate_data_gating(
        ctx,
        state,
        "NFLX",
        "yahoo_close",
        prefer_backup_quote=True,
    )

    assert decision.block
    assert any("limit=1.50%" in reason for reason in decision.reasons)
    assert bot_engine._gap_ratio_gate_limit() == pytest.approx(0.015)


def test_strict_gating_env_toggle_runtime(monkeypatch):
    monkeypatch.setenv("AI_TRADING_STRICT_GATING", "0")
    assert not bot_engine._strict_data_gating_enabled()
    monkeypatch.setenv("AI_TRADING_STRICT_GATING", "1")
    assert bot_engine._strict_data_gating_enabled()


def test_fallback_quote_age_env_toggle_runtime(monkeypatch):
    monkeypatch.setenv("AI_TRADING_FALLBACK_QUOTE_MAX_AGE_SEC", "4.5")
    assert bot_engine._fallback_quote_max_age_seconds() == pytest.approx(4.5)
    monkeypatch.setenv("AI_TRADING_FALLBACK_QUOTE_MAX_AGE_SEC", "9.0")
    assert bot_engine._fallback_quote_max_age_seconds() == pytest.approx(9.0)


def test_gap_limit_env_toggle_runtime(monkeypatch):
    monkeypatch.setenv("AI_TRADING_GAP_LIMIT_BPS", "250")
    assert bot_engine._gap_ratio_gate_limit() == pytest.approx(0.025)
    monkeypatch.setenv("AI_TRADING_GAP_LIMIT_BPS", "125")
    assert bot_engine._gap_ratio_gate_limit() == pytest.approx(0.0125)


def test_liquidity_cap_env_toggle_runtime(monkeypatch):
    monkeypatch.setenv("AI_TRADING_LIQ_FALLBACK_CAP", "0.35")
    assert bot_engine._liquidity_fallback_cap() == pytest.approx(0.35)
    monkeypatch.setenv("AI_TRADING_LIQ_FALLBACK_CAP", "0.10")
    assert bot_engine._liquidity_fallback_cap() == pytest.approx(0.10)
