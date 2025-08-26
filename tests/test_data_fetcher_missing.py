import types
import pytest

from ai_trading.core import bot_engine as eng


def test_ensure_data_fetcher_rebuilds(monkeypatch):
    runtime = types.SimpleNamespace(data_fetcher=None, params={'a': 1})
    sentinel = object()

    def fake_build(params):
        assert params == runtime.params
        return sentinel

    eng.data_fetcher = None
    monkeypatch.setattr(eng.data_fetcher_module, 'build_fetcher', fake_build)
    assert eng.ensure_data_fetcher(runtime) is sentinel
    assert runtime.data_fetcher is sentinel
    assert eng.data_fetcher is sentinel


def test_ensure_data_fetcher_raises(monkeypatch):
    runtime = types.SimpleNamespace(data_fetcher=None, params={})

    def fake_build(params):
        raise eng.DataFetchError('boom')

    eng.data_fetcher = None
    monkeypatch.setattr(eng.data_fetcher_module, 'build_fetcher', fake_build)
    with pytest.raises(eng.DataFetchError):
        eng.ensure_data_fetcher(runtime)
    assert runtime.data_fetcher is None
    assert eng.data_fetcher is None
