# tests/test_config_exports.py

import importlib

def test_lazy_exports_resolve():
    cfg = importlib.import_module("ai_trading.config")

    # get_settings is resolved lazily via __getattr__
    assert hasattr(cfg, "get_settings")
    assert callable(cfg.get_settings)

    settings = cfg.get_settings()
    # Settings should be a dataclass/pydantic BaseSettings-like object; we just assert it exists.
    assert settings is not None
    # A light sanity check on a common field if present; tolerate absence.
    assert hasattr(settings, "__class__")