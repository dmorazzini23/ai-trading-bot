import pandas as pd

def download(*a, **kw):
    # deterministic empty frame with expected columns
    return pd.DataFrame(columns=["Open","High","Low","Close","Adj Close","Volume"])
