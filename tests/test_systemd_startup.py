"""
Integration test for startup behavior and systemd compatibility.

Tests the complete startup flow to ensure no import-time crashes occur
even when environment variables are missing, simulating systemd startup.
"""

import os
import subprocess
import sys
import tempfile

import pytest


class TestSystemdStartupCompatibility:
    """Test systemd-compatible startup behavior."""

    def test_import_no_crash_without_credentials(self):
        """Test that imports don't crash without credentials."""
        # Create a test script that imports key modules
        test_script = '''
import os
import sys

# Clear all credential environment variables
for key in ["ALPACA_API_KEY", "APCA_API_KEY_ID", "ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY"]:
    os.environ.pop(key, None)

try:
    # Test importing key modules without credentials
    from ai_trading.config.management import _resolve_alpaca_env
    print("✓ Config management imported")
    
    from ai_trading import runner
    print("✓ Runner imported")
    
    from ai_trading.utils.timefmt import utc_now_iso
    print("✓ Time utilities imported")
    
    # Test that credential resolution works
    api_key, secret_key, base_url = _resolve_alpaca_env()
    assert api_key is None
    assert secret_key is None
    assert base_url == "https://paper-api.alpaca.markets"
    print("✓ Credential resolution works with missing creds")
    
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

        # Write test script to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            script_path = f.name

        try:
            # Run the test script in a clean subprocess
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=30,
                check=True
            )

            if result.stderr:
                pass

            # Check that script succeeded
            assert result.returncode == 0, f"Script failed with return code {result.returncode}"
            assert "SUCCESS: No import-time crashes!" in result.stdout

        finally:
            os.unlink(script_path)

    def test_dual_credential_schema_with_env_file(self):
        """Test that both credential schemas work with .env files."""
        # Test ALPACA_* schema
        alpaca_env_content = """
ALPACA_API_KEY=test_alpaca_key_from_env
ALPACA_SECRET_KEY=test_alpaca_secret_from_env
ALPACA_BASE_URL=https://paper-api.alpaca.markets
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(alpaca_env_content)
            alpaca_env_path = f.name

        # Test APCA_* schema
        apca_env_content = """
APCA_API_KEY_ID=test_apca_key_from_env
APCA_API_SECRET_KEY=test_apca_secret_from_env
APCA_API_BASE_URL=https://api.alpaca.markets
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(apca_env_content)
            apca_env_path = f.name

        try:
            # Test ALPACA schema
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

            # Test APCA schema
            test_script = f'''
import os
from ai_trading.config.management import _resolve_alpaca_env, reload_env
reload_env("{apca_env_path}", override=True)

api_key, secret_key, base_url = _resolve_alpaca_env()

assert api_key == "test_apca_key_from_env"
assert secret_key == "test_apca_secret_from_env"
assert base_url == "https://api.alpaca.markets"
print("✓ APCA schema with .env file works")
'''

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                script_path = f.name

            result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, timeout=30, check=True)  # AI-AGENT-REF: Added timeout and check for security
            assert result.returncode == 0, f"APCA test failed: {result.stderr}"
            os.unlink(script_path)

        finally:
            os.unlink(alpaca_env_path)
            os.unlink(apca_env_path)

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
for key in ["ALPACA_API_KEY", "APCA_API_KEY_ID", "ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY"]:
    os.environ.pop(key, None)

# Import runner (should work without credentials)
from ai_trading import runner

# Verify lazy loading variables exist
assert hasattr(runner, '_load_engine')
assert hasattr(runner, '_bot_engine') 
assert hasattr(runner, '_bot_state_class')

# Verify initial state is None (not loaded)
assert runner._bot_engine is None
assert runner._bot_state_class is None

print("✓ Lazy import mechanism working correctly")
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
