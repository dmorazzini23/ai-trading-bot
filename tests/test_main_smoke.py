import importlib
import pytest

main = importlib.import_module("run")


def test_main_smoke():
    assert main is not None
