from types import SimpleNamespace

from ai_trading.core.runtime import NullAlphaModel, build_runtime


def test_runtime_has_model():
    cfg = SimpleNamespace()
    runtime = build_runtime(cfg)
    assert hasattr(runtime, "model")
    assert callable(getattr(runtime.model, "predict", None))
    assert isinstance(runtime.model, NullAlphaModel)

