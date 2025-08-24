try:
    import pandas as pd
except Exception:  # AI-AGENT-REF: stub without pandas
    pd = None  # type: ignore[assignment]


def download(*a, **kw):
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    if pd is None:
        return {c: [] for c in cols}
    # deterministic empty frame with expected columns
    return pd.DataFrame(columns=cols)
