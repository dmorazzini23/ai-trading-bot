import inspect
import re
import ai_trading.data_fetcher as df_mod
import ai_trading.main as main_mod


def _source(mod):
    return inspect.getsource(mod)


def test_data_fetcher_has_importerror_for_optional_imports():
    src = _source(df_mod)
    # yfinance optional import
    assert "except ImportError" in src
    # http/requests optional transport import
    assert "HTTP_INIT_FALLBACK" in src and "except ImportError" in src


def test_data_fetcher_json_decode_is_valueerror_only():
    src = _source(df_mod)
    # The JSON decode guard around resp.json() should not be broad
    assert re.search(r"resp\.json\(\)\n\s*except ValueError:\n\s*payload\s*=\s*\{\}", src)


def test_main_get_int_env_narrows_to_valueerror():
    # ensure _get_int_env uses ValueError not Exception
    src = _source(main_mod)
    assert "def _get_int_env" in src
    assert "except ValueError" in src


def test_main_validate_runtime_config_wrapper_is_valueerror():
    src = _source(main_mod)
    assert "RUNTIME_CONFIG_INVALID" in src
    assert "except ValueError" in src


def test_start_api_with_signal_uses_specific_errors():
    src = _source(main_mod)
    assert "def start_api_with_signal" in src
    assert "except (OSError, RuntimeError) as e" in src
