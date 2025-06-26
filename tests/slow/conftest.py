import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="Run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")


def pytest_collection_modifyitems(config, items):
    try:
        if config.getoption("--runslow"):
            return
    except ValueError:
        # Option not registered when this conftest is loaded
        pass
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
