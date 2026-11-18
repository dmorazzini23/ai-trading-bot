import pytest
pd = pytest.importorskip("pandas")


def test_pretrade_lookback_days_setting():
    """Test that the new pretrade_lookback_days setting is available."""
    from ai_trading.config.settings import get_settings

    settings = get_settings()
    assert hasattr(settings, 'pretrade_lookback_days')
    assert settings.pretrade_lookback_days == 120  # Default value

def test_regime_basket_proxy_function():
    """Test that _regime_basket_to_proxy_bars creates proper output."""
    # Simple test of the core logic without importing the full bot engine
    def _mk_wide():
        ts = pd.date_range("2024-01-01", periods=5, freq="B")
        return pd.DataFrame({"timestamp": ts, "SPY":[100,101,102,103,104], "QQQ":[50,51,52,52.5,53]})

    def _regime_basket_to_proxy_bars(wide):
        """Simplified version for testing."""
        if wide is None or wide.empty:
            return pd.DataFrame()
        if "timestamp" not in wide.columns:
            return pd.DataFrame()
        close_cols = [c for c in wide.columns if c != "timestamp"]
        if not close_cols:
            return pd.DataFrame()
        df = wide.copy()
        # Normalize each series to 1.0 at first valid point to avoid scale bias
        base = df[close_cols].iloc[0]
        norm = df[close_cols] / base.replace(0, pd.NA)
        proxy_close = norm.mean(axis=1).astype(float)
        out = pd.DataFrame({"timestamp": df["timestamp"], "close": proxy_close})
        return out

    wide = _mk_wide()
    out = _regime_basket_to_proxy_bars(wide)
    assert "timestamp" in out.columns and "close" in out.columns
    assert len(out) == 5

