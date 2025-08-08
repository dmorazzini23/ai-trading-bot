import os, time
import pandas as pd
from ai_trading.market import cache as mcache

def test_mem_cache_ttl_basic(tmp_path):
    df = pd.DataFrame({"timestamp":[1], "open":[1], "high":[1], "low":[1], "close":[1], "volume":[1]})
    mcache.put_mem("AAPL", "1D", "2024-01-01", "2024-01-31", df)
    got = mcache.get_mem("AAPL","1D","2024-01-01","2024-01-31", ttl=60)
    assert got is not None and list(got.columns)==list(df.columns)
    got2 = mcache.get_mem("AAPL","1D","2024-01-01","2024-01-31", ttl=0)
    assert got2 is None