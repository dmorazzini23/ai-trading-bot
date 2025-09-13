"""Ensure runtime package does not expose test mocks."""

import importlib

import pytest


def test_no_mock_exports():
    import ai_trading

    assert not any(name.startswith('Mock') for name in dir(ai_trading))
    try:
        import ai_trading.core.bot_engine as be
    except ModuleNotFoundError:
        pytest.skip('optional deps missing')
    assert not any(name.startswith('Mock') for name in dir(be))
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module('ai_trading.execution.mocks')

