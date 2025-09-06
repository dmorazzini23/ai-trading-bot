import logging
from types import SimpleNamespace

from ai_trading.startup.config import ensure_max_position_size


def test_startup_autofixes_missing_max_position_size(caplog):
    cfg = SimpleNamespace()
    tcfg = SimpleNamespace(max_position_size=None)
    with caplog.at_level(logging.INFO):
        size = ensure_max_position_size(cfg, tcfg)
    records = [
        r for r in caplog.records
        if r.name == "ai_trading.startup.config" and r.msg == "CONFIG_AUTOFIX"
    ]
    assert records, "CONFIG_AUTOFIX log not emitted"
    assert getattr(records[0], "fallback", 0) == size
    assert tcfg.max_position_size == size
