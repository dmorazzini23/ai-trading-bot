def test_package_first_single_tick_smoke(monkeypatch):
    # This test previously tolerated shims; now enforce package-only.
    import importlib
    main_mod = importlib.import_module("ai_trading.main")
    assert hasattr(main_mod, "run_cycle")
    assert callable(main_mod.run_cycle)

def test_ai_trading_module_imports():
    # Test that modules can be imported from the package
    import importlib

    # Test each moved module can be imported from ai_trading
    modules = [
        "signals",
        "data_fetcher",
        "execution",
        "indicators",
        "pipeline",
        "portfolio",
        "rebalancer",
    ]
    for module_name in modules:
        pkg_module = importlib.import_module(f"ai_trading.{module_name}")
        assert hasattr(pkg_module, "__dict__")

def test_ai_trading_init_exports():
    # Test that ai_trading.__init__ properly exports the modules
    import ai_trading

    modules = [
        "signals",
        "data_fetcher",
        "execution",
        "indicators",
        "pipeline",
        "portfolio",
        "rebalancer",
    ]
    for module_name in modules:
        assert hasattr(ai_trading, module_name)
