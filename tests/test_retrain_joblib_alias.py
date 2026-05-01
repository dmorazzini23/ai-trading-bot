import importlib


def test_retrain_joblib_dump_aliases_cli():
    cli_pkg = importlib.import_module("ai_trading.retrain")

    assert cli_pkg.atomic_joblib_dump
    assert cli_pkg.build_parser().prog == "python -m ai_trading.retrain"
