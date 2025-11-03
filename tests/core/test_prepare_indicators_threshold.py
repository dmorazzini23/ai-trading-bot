from __future__ import annotations

import pandas as pd

from ai_trading.core.bot_engine import prepare_indicators


def test_prepare_indicators_handles_twenty_row_window() -> None:
    """Indicator preparation should not discard moderate-length frames."""

    rows = 22
    frame = pd.DataFrame(
        {
            "close": [100.0 + i * 0.1 for i in range(rows)],
            "high": [100.2 + i * 0.1 for i in range(rows)],
            "low": [99.8 + i * 0.1 for i in range(rows)],
        }
    )

    result = prepare_indicators(frame)

    assert not result.empty
    assert len(result) > 0
    assert len(result) <= len(frame)
