import os
import pytest

os.environ.setdefault("PYTEST_RUNNING", "1")
from ai_trading.analysis import sentiment


def test_analyze_text_interface():
    assert not hasattr(sentiment, "predict_text_sentiment")
    try:
        res = sentiment.analyze_text("markets look good")
    except ModuleNotFoundError:
        pytest.skip("transformers not installed")
    assert isinstance(res, dict)
    assert {'available', 'pos', 'neg', 'neu'} <= set(res)

