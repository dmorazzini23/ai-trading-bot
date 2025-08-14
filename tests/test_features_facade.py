def test_features_facade_resolves_pipeline():
    from features import build_features_pipeline  # must resolve via facade
    assert callable(build_features_pipeline)
