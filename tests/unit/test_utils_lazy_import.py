import types
from ai_trading.utils.lazy_imports import optional_import


def test_utils_http_lazy_import_no_recursion():
    import ai_trading.utils as utils

    mod = utils.http  # attribute access triggers __getattr__
    assert isinstance(mod, types.ModuleType)
    # cached on second access
    assert utils.http is mod


def test_optional_import():
    assert optional_import("math") is not None
    assert optional_import("module_does_not_exist") is None

