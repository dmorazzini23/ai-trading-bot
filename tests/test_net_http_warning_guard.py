import importlib
import sys
import types


def test_disable_warnings_handles_missing_httpwarning():
    import ai_trading.net.http as http

    original_urllib3 = sys.modules.get("urllib3")
    recorded_categories: list[type[Warning]] = []

    stub = types.ModuleType("urllib3")

    class _SystemTimeWarning(Warning):
        pass

    def _disable_warnings(category=None):
        recorded_categories.append(category)

    stub.disable_warnings = _disable_warnings
    stub.exceptions = types.SimpleNamespace(SystemTimeWarning=_SystemTimeWarning)

    sys.modules["urllib3"] = stub
    try:
        importlib.reload(http)
        assert recorded_categories, "disable_warnings was not invoked"
        chosen = recorded_categories[-1]
        assert chosen is not None
        assert issubclass(chosen, Warning)
    finally:
        if original_urllib3 is not None:
            sys.modules["urllib3"] = original_urllib3
        else:
            sys.modules.pop("urllib3", None)
        importlib.reload(http)


def test_disable_warnings_missing_attribute_can_be_monkeypatched():
    import ai_trading.net.http as http

    original_urllib3 = sys.modules.get("urllib3")

    stub = types.ModuleType("urllib3")
    stub.exceptions = types.SimpleNamespace(SystemTimeWarning=Warning)
    # Explicitly omit disable_warnings to simulate an older/broken urllib3.

    sys.modules["urllib3"] = stub
    try:
        # Initial reload should populate the shim without raising.
        importlib.reload(http)
        assert hasattr(stub, "disable_warnings")
        assert callable(stub.disable_warnings)

        recorded_categories: list[type[Warning] | None] = []

        def _patched_disable_warnings(category=None, *args, **kwargs):
            recorded_categories.append(category)

        stub.disable_warnings = _patched_disable_warnings

        importlib.reload(http)
        assert recorded_categories, "patched disable_warnings should be invoked"
        assert recorded_categories[-1] is not None
    finally:
        if original_urllib3 is not None:
            sys.modules["urllib3"] = original_urllib3
        else:
            sys.modules.pop("urllib3", None)
        importlib.reload(http)
