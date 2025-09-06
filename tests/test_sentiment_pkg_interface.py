import os
import sys

import pytest

sys.modules.pop("tenacity", None)
pytest.importorskip("tenacity")
os.environ.setdefault("PYTEST_RUNNING", "1")

from ai_trading.sentiment import interface


def test_analyze_text_structure():
    try:
        res = interface.analyze_text("markets look good")
    except ModuleNotFoundError:
        pytest.skip("transformers not installed")
    assert isinstance(res, dict)
    assert {"available", "pos", "neg", "neu"} <= set(res)

