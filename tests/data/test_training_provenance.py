from datetime import date
from pathlib import Path

import pandas as pd

from ai_trading.data.training_provenance import validate_training_provenance
from ai_trading.utils.market_calendar import session_info


def test_actual_calendar_gaps_and_session_support_override_manifest_claims(tmp_path: Path) -> None:
    session = session_info(date(2026, 9, 3))
    index = pd.date_range(session.start_utc, session.end_utc, freq="min", inclusive="left")
    pd.DataFrame({"timestamp": index, "open": 100.0, "high": 101.0,
                  "low": 99.0, "close": 100.0, "volume": 1000}).to_csv(tmp_path / "AAPL.csv", index=False)
    identity = {"request_interval": {"start_date": "2026-09-03", "end_date": "2026-09-04"}}
    result = validate_training_provenance(tmp_path, symbols=["AAPL", "MSFT"], dataset_identity=identity)
    assert result["quality_passed"] is False
    assert set(result["rejection_reasons"]) == {"insufficient_sessions:AAPL", "excess_missing_bars:AAPL", "symbol_missing:MSFT"}
    assert result["coverage"]["AAPL"]["missing_ratio"] == 0.5
    assert sum(result["coverage"]["AAPL"]["regime_distribution"].values()) == 390
    identity["request_interval"]["end_date"] = "2026-09-03"
    passed = validate_training_provenance(tmp_path, symbols=["AAPL"], dataset_identity=identity, min_sessions=1)
    assert passed["quality_passed"] is True
