from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ai_trading.core import bot_engine as bot


@pytest.fixture
def shadow(monkeypatch):
    config = {'AI_TRADING_ML_SHADOW_ENABLED': True}
    monkeypatch.setattr(bot, 'get_env', lambda key, default=None, **k: config.get(key, default))
    monkeypatch.setattr(bot, '_SHADOW_MODEL_CACHE', None)
    monkeypatch.setattr(bot, '_SHADOW_MODEL_CACHE_META', None)
    return config


def test_file_cache_reloads_changed_artifact_through_verified_loader(shadow, tmp_path, monkeypatch):
    path = tmp_path / 'model.pkl'
    path.write_bytes(b'first')
    shadow['AI_TRADING_ML_SHADOW_MODEL_PATH'] = str(path)
    first, second = object(), object()
    loader = Mock(side_effect=[first, second])
    monkeypatch.setattr(bot, 'load_verified_joblib_artifact', loader)
    assert bot._load_shadow_model() is first
    assert bot._load_shadow_model() is first
    loader.assert_called_once_with(str(path))
    path.write_bytes(b'changed content and size')
    assert bot._load_shadow_model() is second
    assert loader.call_count == 2
    path.unlink()
    assert bot._load_shadow_model() is None


def test_invalid_verified_artifact_never_enters_shadow_cache(shadow, tmp_path, monkeypatch):
    path = tmp_path / 'model.pkl'
    path.write_bytes(b'invalid')
    shadow['AI_TRADING_ML_SHADOW_MODEL_PATH'] = str(path)
    monkeypatch.setattr(bot, 'load_verified_joblib_artifact', Mock(side_effect=ValueError('hash mismatch')))
    assert bot._load_shadow_model() is None
    assert bot._SHADOW_MODEL_CACHE is None


@pytest.mark.parametrize('failure', [None, 'missing_factory', 'factory_error', 'import_error'])
def test_module_factory_cache_and_failure_boundaries(shadow, monkeypatch, failure):
    shadow['AI_TRADING_ML_SHADOW_MODEL_MODULE'] = 'test_shadow_module'
    model = object()
    factory = Mock(return_value=model)
    if failure == 'factory_error':
        factory.side_effect = ValueError('bad model')
    module = SimpleNamespace() if failure == 'missing_factory' else SimpleNamespace(get_model=factory)
    original_import = bot.importlib.import_module
    def resolve(name, *args, **kwargs):
        if name != 'test_shadow_module':
            return original_import(name, *args, **kwargs)
        if failure == 'import_error':
            raise ValueError('invalid module')
        return module
    monkeypatch.setattr(bot.importlib, 'import_module', resolve)
    result = bot._load_shadow_model()
    if failure:
        assert result is None and bot._SHADOW_MODEL_CACHE is None
    else:
        assert result is model and bot._load_shadow_model() is model
        factory.assert_called_once()


def test_disabled_or_unconfigured_shadow_does_not_load(shadow):
    assert bot._load_shadow_model() is None
    shadow['AI_TRADING_ML_SHADOW_ENABLED'] = False
    assert bot._load_shadow_model() is None
