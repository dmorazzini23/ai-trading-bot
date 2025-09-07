from ai_trading.meta_learning import validate_trade_data_quality
import pytest


def test_validate_trade_data_quality_filters_non_positive_prices(tmp_path):
    """Rows with non-positive entry or exit prices should be ignored."""
    p = tmp_path / "trades.csv"
    with p.open("w") as f:
        f.write(
            "symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward\n"
        )
        # Valid row
        f.write(
            "GOOD,2025-01-01T00:00:00Z,10,2025-01-01T00:05:00Z,12,1,buy,strat,test,tag,0.5,1\n"
        )
        # Negative entry price
        f.write(
            "NEGENTRY,2025-01-01T00:00:00Z,-10,2025-01-01T00:05:00Z,12,1,buy,strat,test,tag,0.5,1\n"
        )
        # Negative exit price
        f.write(
            "NEGEXIT,2025-01-01T00:00:00Z,10,2025-01-01T00:05:00Z,-12,1,buy,strat,test,tag,0.5,1\n"
        )
    report = validate_trade_data_quality(str(p))
    assert report["row_count"] == 3
    assert report["valid_price_rows"] == 1
    assert report["meta_format_rows"] == 1
    assert report["data_quality_score"] == pytest.approx(1/3)
