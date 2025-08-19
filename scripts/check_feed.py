"""Small diagnostic to verify market data fetch."""

import pandas as pd, datetime as dt, pytz  # noqa: F401
from types import SimpleNamespace

from ai_trading.core.bot_engine import safe_get_stock_bars
from ai_trading.core.runtime import build_runtime


if __name__ == "__main__":  # pragma: no cover - manual use
    rt = build_runtime(SimpleNamespace())
    now = pd.Timestamp.utcnow().tz_localize("UTC")
    start = now - pd.Timedelta(days=120)
    client = getattr(rt, "data_client", SimpleNamespace(get_stock_bars=lambda req: SimpleNamespace(df=pd.DataFrame())))
    df = safe_get_stock_bars(client, None, "SPY", "1Day")
    print("rows:", len(df), "cols:", list(df.columns))
    print(df.tail(3))

