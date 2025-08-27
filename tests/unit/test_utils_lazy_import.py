import types


def test_utils_http_lazy_import_no_recursion():
    import ai_trading.utils as utils

    mod = utils.http  # attribute access triggers __getattr__
    assert isinstance(mod, types.ModuleType)
    # cached on second access
    assert utils.http is mod

