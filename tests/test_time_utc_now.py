import pytest
pd = pytest.importorskip("pandas")
def test_now_is_aware_utc():
    # AI-AGENT-REF: ensure timezone-aware UTC now
    now_utc = pd.Timestamp.now(tz="UTC")
    assert now_utc.tz is not None
    assert str(now_utc.tz) in {"UTC", "+00:00"}
