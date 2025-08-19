# AI-AGENT-REF: ensure settings expose required runtime attributes (including log_market_fetch)
from ai_trading.config.settings import get_settings


def test_cfg_required_fields_exist():
    cfg = get_settings()
    assert hasattr(cfg, "log_market_fetch")
    assert hasattr(cfg, "testing")
    assert hasattr(cfg, "shadow_mode")
    assert hasattr(cfg, "healthcheck_port")
    assert hasattr(cfg, "min_health_rows")
