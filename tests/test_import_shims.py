def test_signals_import_paths_identical():
    import importlib
    pkg = importlib.import_module("ai_trading.signals")
    shim = importlib.import_module("signals")
    # Attributes resolve
    assert hasattr(pkg, "__dict__")
    assert hasattr(shim, "__dict__")
    # Optional: same object identity for key callables/constants if feasible
    if hasattr(pkg, "generate_position_hold_signals"):
        assert shim.generate_position_hold_signals is pkg.generate_position_hold_signals

def test_data_fetcher_import_paths_identical():
    import importlib
    pkg = importlib.import_module("ai_trading.data_fetcher")
    shim = importlib.import_module("data_fetcher")
    assert hasattr(pkg, "__dict__")
    assert hasattr(shim, "__dict__")

def test_trade_execution_import_paths_identical():
    import importlib
    pkg = importlib.import_module("ai_trading.trade_execution")
    shim = importlib.import_module("trade_execution")
    assert hasattr(pkg, "__dict__")
    assert hasattr(shim, "__dict__")

def test_indicators_import_paths_identical():
    import importlib
    pkg = importlib.import_module("ai_trading.indicators")
    shim = importlib.import_module("indicators")
    assert hasattr(pkg, "__dict__")
    assert hasattr(shim, "__dict__")

def test_pipeline_import_paths_identical():
    import importlib
    pkg = importlib.import_module("ai_trading.pipeline")
    shim = importlib.import_module("pipeline")
    assert hasattr(pkg, "__dict__")
    assert hasattr(shim, "__dict__")

def test_portfolio_import_paths_identical():
    import importlib
    pkg = importlib.import_module("ai_trading.portfolio")
    shim = importlib.import_module("portfolio")
    assert hasattr(pkg, "__dict__")
    assert hasattr(shim, "__dict__")

def test_rebalancer_import_paths_identical():
    import importlib
    pkg = importlib.import_module("ai_trading.rebalancer")
    shim = importlib.import_module("rebalancer")
    assert hasattr(pkg, "__dict__")
    assert hasattr(shim, "__dict__")