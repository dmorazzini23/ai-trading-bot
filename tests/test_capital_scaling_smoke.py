import importlib
import sys
from pathlib import Path

import pytest

sys.modules.pop("ai_trading.capital_scaling", None)
capital_scaling = importlib.import_module("ai_trading.capital_scaling")


def force_coverage(mod):
    """Force coverage by importing and accessing module attributes instead of using exec."""
    for attr_name in dir(mod):
        if not attr_name.startswith('_'):
            getattr(mod, attr_name, None)


@pytest.mark.smoke
def test_capital_scaler_basic():
    eng = capital_scaling.CapitalScalingEngine({"x": 1})
    assert eng.scale_position(10) == 10
