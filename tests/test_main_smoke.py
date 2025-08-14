import pytest

main = pytest.importorskip("run")


def test_main_smoke():
    assert main is not None
