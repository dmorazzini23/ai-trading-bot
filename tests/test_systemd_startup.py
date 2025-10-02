"""
Integration test for startup behavior and systemd compatibility.

Tests the complete startup flow to ensure no import-time crashes occur
even when environment variables are missing, simulating systemd startup.
"""

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest


class TestSystemdStartupCompatibility:
    """Test systemd-compatible startup behavior."""

    def test_import_no_crash_without_credentials(self):
        """Test that imports don't crash without credentials."""
        # Create a test script that imports key modules
        test_script = textwrap.dedent(
            '''
            import os
            import sys
            import types

            if "numpy" not in sys.modules:
                class _Array(list):
                    def mean(self):
                        return sum(self) / len(self) if self else 0.0

                def _to_array(data):
                    if isinstance(data, _Array):
                        return data
                    if data is None:
                        return _Array()
                    return _Array(data)

                def _array(data=None, *args, **kwargs):
                    return _to_array(data)

                def _diff(data, *args, **kwargs):
                    seq = list(data or [])
                    return _Array(seq[i + 1] - seq[i] for i in range(len(seq) - 1))

                def _where(condition, x, y):
                    if isinstance(condition, (list, tuple, _Array)):
                        return _Array(xv if cond else yv for cond, xv, yv in zip(condition, x, y))
                    return x if condition else y

                def _zeros_like(data, *args, **kwargs):
                    return _Array(0 for _ in (data or []))

                numpy_stub = types.ModuleType("numpy")
                numpy_stub.random = types.SimpleNamespace(
                    seed=lambda *args, **kwargs: None,
                )
                numpy_stub.__dict__.update(
                    {
                        "__version__": "0.0-stub",
                        "ndarray": _Array,
                        "nan": float("nan"),
                        "NaN": float("nan"),
                        "float64": float,
                        "int64": int,
                        "array": _array,
                        "asarray": _array,
                        "diff": _diff,
                        "where": _where,
                        "zeros_like": _zeros_like,
                    }
                )
                sys.modules["numpy"] = numpy_stub

            if "portalocker" not in sys.modules:
                portalocker_stub = types.ModuleType("portalocker")
                portalocker_stub.LOCK_EX = 1
                portalocker_stub.lock = lambda *args, **kwargs: None
                portalocker_stub.unlock = lambda *args, **kwargs: None
                sys.modules["portalocker"] = portalocker_stub

            if "bs4" not in sys.modules:
                bs4_stub = types.ModuleType("bs4")

                class BeautifulSoup:  # noqa: D401 - minimal stub
                    """Stub BeautifulSoup that ignores all input."""

                    def __init__(self, *args, **kwargs):
                        self.args = args
                        self.kwargs = kwargs

                    def find(self, *args, **kwargs):  # pragma: no cover - stub
                        return None

                bs4_stub.BeautifulSoup = BeautifulSoup
                sys.modules["bs4"] = bs4_stub

            if "flask" not in sys.modules:
                flask_stub = types.ModuleType("flask")

                class Flask:  # pragma: no cover - minimal server stub
                    def __init__(self, *args, **kwargs):
                        self.args = args
                        self.kwargs = kwargs

                    def route(self, *_args, **_kwargs):
                        def decorator(func):
                            return func

                        return decorator

                    def run(self, *args, **kwargs):
                        return None

                flask_stub.Flask = Flask
                flask_stub.Request = object
                flask_stub.Response = object
                flask_stub.jsonify = lambda *a, **k: {}
                sys.modules["flask"] = flask_stub

            os.environ.setdefault("PYTEST_RUNNING", "1")

            # Clear all credential environment variables
            for key in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]:
                os.environ.pop(key, None)

            # Ensure drawdown thresholds are absent to exercise lenient path
            for key in ["MAX_DRAWDOWN_THRESHOLD", "AI_TRADING_MAX_DRAWDOWN_THRESHOLD"]:
                os.environ.pop(key, None)

            os.environ["CAPITAL_CAP"] = "1"
            os.environ["DOLLAR_RISK_LIMIT"] = "0.1"

            try:
                # Test importing key modules without credentials
                from ai_trading.config.management import _resolve_alpaca_env
                print("✓ Config management imported")

                api_key, secret_key, base_url = _resolve_alpaca_env()
                assert api_key is None
                assert secret_key is None
                assert base_url == "https://paper-api.alpaca.markets"
                print("✓ Credential resolution works with missing creds")

                from ai_trading import main as _main
                print("✓ Main imported")

                _main._fail_fast_env()
                print("✓ Environment validation executed")

                from ai_trading.utils.timefmt import utc_now_iso
                print("✓ Time utilities imported")

                # Test UTC timestamp doesn't have double Z
                timestamp = utc_now_iso()
                assert timestamp.endswith('Z')
                assert timestamp.count('Z') == 1
                print("✓ UTC timestamp has single Z")

                print("SUCCESS: No import-time crashes!")

            except SystemExit as e:
                print(f"FAIL: SystemExit called: {e}")
                sys.exit(1)
            except Exception as e:
                print(f"FAIL: Unexpected error: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
            '''
        )

        # Write test script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name

        try:
            # Run the test script in a clean subprocess
            env = os.environ.copy()
            project_root = Path(__file__).resolve().parents[1]
            stub_path = project_root / "tests" / "stubs"
            python_path_parts = [str(project_root)]
            if stub_path.exists():
                python_path_parts.append(str(stub_path))
            existing_path = env.get("PYTHONPATH", "")
            if existing_path:
                python_path_parts.append(existing_path)
            env["PYTHONPATH"] = os.pathsep.join(python_path_parts)
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
                env=env,
            )

            if result.stderr:
                pass

            # Check that script succeeded
            assert result.returncode == 0, f"Script failed with return code {result.returncode}"
            assert "SUCCESS: No import-time crashes!" in result.stdout
            combined_output = "".join([result.stdout, result.stderr])
            assert "ALPACA_CREDENTIALS_MISSING" in combined_output
            assert "ENV_VALIDATION_FAILED" not in combined_output

        finally:
            os.unlink(script_path)

    def test_import_management_when_settings_config_dict_rejects(self):
        """Simulate lean env where SettingsConfigDict rejects options."""

        test_script = '''
import sys
import types

# Ensure no cached module leaks through
sys.modules.pop("pydantic_settings", None)

stub = types.ModuleType("pydantic_settings")


class BaseSettings:
    model_config = None


class RejectingSettingsConfigDict(dict):
    def __init__(self, *args, **kwargs):
        raise TypeError("options not supported")


stub.BaseSettings = BaseSettings
stub.SettingsConfigDict = RejectingSettingsConfigDict
sys.modules["pydantic_settings"] = stub

from ai_trading.config import management  # noqa: F401 - import ensures no crash

print("✓ management import succeeded with rejecting SettingsConfigDict")
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name

        try:
            env = os.environ.copy()
            project_root = Path(__file__).resolve().parents[1]
            stub_path = project_root / "tests" / "stubs"
            python_path_parts = [str(project_root)]
            if stub_path.exists():
                python_path_parts.append(str(stub_path))
            existing_path = env.get("PYTHONPATH", "")
            if existing_path:
                python_path_parts.append(existing_path)
            env["PYTHONPATH"] = os.pathsep.join(python_path_parts)

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
                env=env,
            )

            assert result.returncode == 0, result.stderr
            assert "management import succeeded" in result.stdout
        finally:
            os.unlink(script_path)

    def test_alpaca_credential_schema_with_env_file(self):
        """Test that the ALPACA credential schema works with .env files."""
        alpaca_env_content = """
ALPACA_API_KEY=test_alpaca_key_from_env
ALPACA_SECRET_KEY=test_alpaca_secret_from_env
ALPACA_BASE_URL=https://paper-api.alpaca.markets
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(alpaca_env_content)
            alpaca_env_path = f.name

        try:
            test_script = f'''
import os
from ai_trading.config.management import _resolve_alpaca_env, reload_env
reload_env("{alpaca_env_path}", override=True)

api_key, secret_key, base_url = _resolve_alpaca_env()

assert api_key == "test_alpaca_key_from_env"
assert secret_key == "test_alpaca_secret_from_env"
assert base_url == "https://paper-api.alpaca.markets"
print("✓ ALPACA schema with .env file works")
'''

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                script_path = f.name

            result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=30, check=True)  # AI-AGENT-REF: Added timeout and check for security
            assert result.returncode == 0, f"ALPACA test failed: {result.stderr}"
            os.unlink(script_path)

        finally:
            os.unlink(alpaca_env_path)

    def test_utc_timestamp_no_double_z(self):
        """Test that UTC timestamps don't have double Z suffix."""
        test_script = '''
from ai_trading.utils.timefmt import utc_now_iso, format_datetime_utc, ensure_utc_format
from datetime import datetime, timezone

# Test utc_now_iso
timestamp = utc_now_iso()
assert timestamp.endswith('Z')
assert timestamp.count('Z') == 1
print(f"✓ utc_now_iso: {timestamp}")

# Test format_datetime_utc
dt = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
formatted = format_datetime_utc(dt)
assert formatted == "2024-01-01T12:00:00Z"
assert formatted.count('Z') == 1
print(f"✓ format_datetime_utc: {formatted}")

# Test ensure_utc_format fixes double Z
fixed = ensure_utc_format("2024-01-01T12:00:00ZZ")
assert fixed == "2024-01-01T12:00:00Z"
assert fixed.count('Z') == 1
print(f"✓ ensure_utc_format: {fixed}")

print("✓ All UTC timestamp functions work correctly")
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name

        try:
            result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=30, check=True)  # AI-AGENT-REF: Added timeout and check for security
            if result.stderr:
                pass
            assert result.returncode == 0, f"UTC test failed: {result.stderr}"

        finally:
            os.unlink(script_path)

    def test_lazy_import_behavior(self):
        """Test that lazy imports work correctly."""
        test_script = '''
import os

# Clear credentials
for key in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY"]:
    os.environ.pop(key, None)

from ai_trading import main as _main

# Verify run_cycle exists
assert hasattr(_main, 'run_cycle')

print("✓ Main module import works correctly")
'''

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name

        try:
            result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=30, check=True)  # AI-AGENT-REF: Added timeout and check for security
            if result.stderr:
                pass
            assert result.returncode == 0, f"Lazy import test failed: {result.stderr}"

        finally:
            os.unlink(script_path)


if __name__ == "__main__":
    pytest.main([__file__])
