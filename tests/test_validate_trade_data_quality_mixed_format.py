import pytest
from ai_trading.meta_learning import validate_trade_data_quality


@pytest.fixture
def mixed_format_file(tmp_path):
    """Create a trade log containing both meta-learning and audit rows."""
    p = tmp_path / "mixed.csv"
    with p.open("w") as f:
        f.write(
            "symbol,entry_time,entry_price,exit_time,exit_price,qty,side,strategy,classification,signal_tags,confidence,reward\n"
        )
        # Meta-learning formatted row
        f.write(
            "META,2025-01-01T00:00:00Z,10,2025-01-01T01:00:00Z,12,1,buy,strat,test,tag,0.5,1\n"
        )
        # Audit formatted row
        f.write(
            "123e4567-e89b-12d3-a456-426614174000,2025-01-01T02:00:00Z,AAPL,buy,5,150.0,live,filled\n"
        )
    return str(p)


def test_mixed_format_detection(mixed_format_file):
    report = validate_trade_data_quality(mixed_format_file)
    assert report["mixed_format_detected"] is True
    assert report["audit_format_rows"] >= 1
    assert report["meta_format_rows"] >= 1
