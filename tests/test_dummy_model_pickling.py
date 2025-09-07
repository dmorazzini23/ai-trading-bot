"""Tests for dummy model pickling functionality."""
import pickle
import sys


def test_dummy_model_function_picklable(tmp_path):
    """The dummy model's factory function should be picklable with stdlib."""
    get_model = sys.modules["dummy_model"].get_model
    path = tmp_path / "factory.pkl"
    with path.open("wb") as fh:
        pickle.dump(get_model, fh)
    with path.open("rb") as fh:
        loaded = pickle.load(fh)
    assert callable(loaded)
    assert loaded().__class__.__name__ == "_DummyModel"


def test_dummy_model_instance_picklable(tmp_path):
    """Instances returned by the factory should also be picklable."""
    model = sys.modules["dummy_model"].get_model()
    path = tmp_path / "model.pkl"
    with path.open("wb") as fh:
        pickle.dump(model, fh)
    with path.open("rb") as fh:
        loaded = pickle.load(fh)
    assert loaded.predict([0]) == [0]
