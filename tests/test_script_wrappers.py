from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_algorithm_optimizer_wrapper():
    script_mod = _load_script("algorithm_optimizer")
    from ai_trading import algorithm_optimizer as pkg_mod

    assert script_mod.AlgorithmOptimizer is pkg_mod.AlgorithmOptimizer
    assert script_mod.get_algorithm_optimizer is pkg_mod.get_algorithm_optimizer


def test_ml_model_wrapper():
    script_mod = _load_script("ml_model")
    from ai_trading import ml_model as pkg_mod

    assert script_mod.MLModel is pkg_mod.MLModel
    assert script_mod.train_model is pkg_mod.train_model
