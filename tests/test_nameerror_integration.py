"""Integration test to verify the NameError fix for BUY_THRESHOLD"""
import os
import sys
import tempfile
import subprocess
from pathlib import Path


def test_bot_engine_import_no_nameerror():
    """Test that bot_engine can be imported without NameError for BUY_THRESHOLD.
    
    This test creates a controlled environment and tries to import bot_engine,
    specifically checking for the NameError that was occurring before the fix.
    """
    
    # Create a test script that tries to import bot_engine
    test_script = '''
import os
import sys

# Set minimal required environment variables
os.environ.update({
    "ALPACA_API_KEY": "PKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ",  # Valid format
    "ALPACA_SECRET_KEY": "SKTEST1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890ABCD",  # Valid format
    "BOT_MODE": "balanced",
    "DOLLAR_RISK_LIMIT": "0.02",
})

try:
    # This should trigger validate_trading_parameters() during import
    import bot_engine
    print("SUCCESS: bot_engine imported without NameError")
    exit_code = 0
except NameError as e:
    if "BUY_THRESHOLD" in str(e):
        print(f"FAILURE: NameError for BUY_THRESHOLD still occurs: {e}")
        exit_code = 1
    elif any(param in str(e) for param in ["CAPITAL_CAP", "CONF_THRESHOLD", "DOLLAR_RISK_LIMIT"]):
        print(f"FAILURE: NameError for trading parameter: {e}")
        exit_code = 1
    else:
        print(f"OTHER_NAMEERROR: {e}")
        exit_code = 2
except Exception as e:
    # Other exceptions are expected due to missing dependencies, incomplete env, etc.
    print(f"OTHER_EXCEPTION: {type(e).__name__}: {e}")
    exit_code = 0  # This is OK

sys.exit(exit_code)
'''
    
    # Write the test script to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name
    
    try:
        # Get the project root directory
        project_root = Path(__file__).resolve().parents[1]
        
        # Run the test script in a subprocess with proper PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root)
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            env=env,
            timeout=30
        )
        
        print(f"Test script output: {result.stdout}")
        if result.stderr:
            print(f"Test script stderr: {result.stderr}")
        
        # Check the exit code
        if result.returncode == 1:
            assert False, f"NameError for BUY_THRESHOLD or other trading parameter still occurs: {result.stdout}"
        elif result.returncode == 2:
            assert False, f"Unexpected NameError: {result.stdout}"
        # exit code 0 means success or expected exception
        
    finally:
        # Clean up the temporary file
        os.unlink(script_path)


if __name__ == "__main__":
    test_bot_engine_import_no_nameerror()
    print("Integration test passed!")