import pytest
import utils
from datetime import datetime


def test_safe_to_datetime_invalid():
    with pytest.raises(ValueError):
        utils.safe_to_datetime("notadate")


def test_get_datetime_column_variants_empty():
    df = []
    assert utils.get_datetime_column(df) is None


def test_get_symbol_column_variants_empty():
    df = []
    assert utils.get_symbol_column(df) is None


def test_get_order_column_invalid():
    df = []
    assert utils.get_order_column(df, "id") is None


def test_ohlcv_variants_missing():
    df = []
    assert utils.get_ohlcv_columns(df) == []


def test_get_indicator_column_missing():
    df = []
    assert utils.get_indicator_column(df, "foo") is None


def test_model_lock_context():
    with utils.model_lock():
        pass
