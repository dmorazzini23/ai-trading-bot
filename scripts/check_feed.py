"""Small diagnostic to verify market data fetch."""

from types import SimpleNamespace

import pandas as pd  # noqa: F401
import pytz  # noqa: F401

from ai_trading.core.runtime import build_runtime
from ai_trading.data.bars import safe_get_stock_bars

if __name__ == "__main__":  # pragma: no cover - manual use
    rt = build_runtime(SimpleNamespace())
    # AI-AGENT-REF: use timezone-aware UTC now for diagnostics
    now = pd.Timestamp.now(tz="UTC")
    start = now - pd.Timedelta(days=120)
    client = getattr(
        rt,
        "data_client",
        SimpleNamespace(get_stock_bars=lambda req: SimpleNamespace(df=pd.DataFrame())),
    )
    df = safe_get_stock_bars(client, None, "SPY", "1Day")
    print("rows:", len(df), "cols:", list(df.columns))
    print(df.tail(3))
