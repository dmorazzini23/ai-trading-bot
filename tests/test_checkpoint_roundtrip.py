from ai_trading import meta_learning


def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "chk.pkl"
    data = {"foo": 1}
    meta_learning.save_model_checkpoint(data, str(path))
    loaded = meta_learning.load_checkpoint(str(path))
    assert loaded == data
