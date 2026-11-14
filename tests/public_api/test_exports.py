import importlib
from importlib import import_module
from pathlib import Path

import pytest





def test_package_import_does_not_patch_runtime_override_helpers():
    import ai_trading
    from ai_trading.config import management

    original_clear = management.clear_runtime_env_overrides

    importlib.reload(ai_trading)

    assert management.clear_runtime_env_overrides is original_clear
    assert not hasattr(management, "_test_override_guard_installed")




def test_legacy_production_surfaces_are_not_package_exports():
    import ai_trading
    import ai_trading.execution as execution_mod

    importlib.reload(ai_trading)
    importlib.reload(execution_mod)
    vars(ai_trading).pop("production_system", None)
    vars(execution_mod).pop("ProductionExecutionCoordinator", None)

    assert "production_system" not in ai_trading.__all__
    assert "ProductionExecutionCoordinator" not in execution_mod.__all__
    assert not hasattr(ai_trading, "production_system")
    assert not hasattr(execution_mod, "ProductionExecutionCoordinator")
    package_dir = Path(ai_trading.__file__).resolve().parent
    assert not (package_dir / "production_system.py").exists()
    assert not (package_dir / "execution" / "production_engine.py").exists()


def test_root_execution_engine_export_uses_runtime_selector(monkeypatch):
    import ai_trading
    import ai_trading.execution as execution_mod

    class SelectedExecutionEngine:
        pass

    monkeypatch.setattr(
        execution_mod,
        "select_execution_engine",
        lambda: SelectedExecutionEngine,
    )
    vars(ai_trading).pop("ExecutionEngine", None)

    assert ai_trading.ExecutionEngine is SelectedExecutionEngine
    assert "ExecutionEngine" not in vars(ai_trading)
