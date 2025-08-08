import pytest
from ai_trading.config import Settings

def test_dual_schema_resolution(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "pk_apca")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "sk_apca")
    s = Settings()
    assert s.alpaca_api_key == "pk_apca"
    assert s.alpaca_secret_key == "sk_apca"

def test_shadow_mode_bypasses_validation():
    s = Settings(shadow_mode=True)
    s.require_alpaca_or_raise()  # should not raise

def test_missing_creds_raises():
    s = Settings(shadow_mode=False, alpaca_api_key=None, alpaca_secret_key=None)
    with pytest.raises(RuntimeError):
        s.require_alpaca_or_raise()

def test_worker_defaults_and_overrides(monkeypatch):
    s = Settings()
    assert s.effective_executor_workers(8) == 4
    assert s.effective_prediction_workers(1) == 2
    monkeypatch.setenv("EXECUTOR_WORKERS", "3")
    monkeypatch.setenv("PREDICTION_WORKERS", "5")
    s2 = Settings()
    assert s2.effective_executor_workers(8) == 3
    assert s2.effective_prediction_workers(8) == 5