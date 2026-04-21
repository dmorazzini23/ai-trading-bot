from ai_trading.meta import checkpoint


def test_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "chk.json"
    data = {"foo": 1}
    checkpoint.save_checkpoint(data, str(path))
    loaded = checkpoint.load_checkpoint(str(path))
    assert loaded == data
