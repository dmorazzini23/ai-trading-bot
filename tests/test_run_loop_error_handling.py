from argparse import Namespace

import pytest
import requests

from ai_trading.__main__ import _run_loop


def _args():
    return Namespace(once=True, interval=0)


def test_run_loop_swallow_value_error():
    def fn():
        raise ValueError("bad")

    _run_loop(fn, _args(), "Test")


def test_run_loop_swallow_http_error():
    def fn():
        raise requests.HTTPError("oops")

    _run_loop(fn, _args(), "Test")


def test_run_loop_unexpected_exception_propagates():
    def fn():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _run_loop(fn, _args(), "Test")

