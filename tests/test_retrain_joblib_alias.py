import importlib


def test_retrain_joblib_dump_aliases_cli():
    retrain_pkg = importlib.import_module("retrain")
    cli_pkg = importlib.import_module("ai_trading.retrain")

    assert retrain_pkg.joblib.dump is cli_pkg.atomic_joblib_dump
