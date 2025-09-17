import pytest

from ai_trading.meta_learning import validate_trade_data_quality


def test_validate_trade_data_quality_enforces_decimal_strings(tmp_path):
    """Rows with exponential price strings are rejected by validation."""

    path = tmp_path / "strict_prices.csv"
    content = """symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward
AAA,2025-01-01T00:00:00Z,100.50,2025-01-01T00:05:00Z,105.75,1,buy,strat,test,momentum,0.5,1
BBB,2025-01-01T01:00:00Z,1e2,2025-01-01T01:05:00Z,195.00,1,sell,strat,test,mean_revert,0.4,-1
CCC,2025-01-01T02:00:00Z,invalid,2025-01-01T02:05:00Z,210.00,1,buy,strat,test,momentum,0.6,1
DDD,2025-01-01T03:00:00Z,50.25,2025-01-01T03:05:00Z,55.00,1,sell,strat,test,trend,0.3,1
""".strip()
    path.write_text(content + "\n")

    report = validate_trade_data_quality(str(path))

    assert report["row_count"] == 4
    assert report["valid_price_rows"] == 2
    assert report["data_quality_score"] == pytest.approx(0.5)
