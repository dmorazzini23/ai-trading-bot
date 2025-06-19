from pathlib import Path
import sys
from dotenv import load_dotenv
import os
import pytest
import urllib3

def pytest_configure() -> None:
    """Load environment variables for tests."""
    env_file = Path('.env.test')
    if not env_file.exists():
        env_file = Path('.env')
    if env_file.exists():
        load_dotenv(env_file)
    # Ensure project root is on the import path so modules like
    # ``capital_scaling`` resolve when tests are run from the ``tests``
    # directory by CI tools or developers.
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    os.environ.setdefault("ALPACA_API_KEY", "testkey")
    os.environ.setdefault("ALPACA_SECRET_KEY", "testsecret")
    os.environ.setdefault("FLASK_PORT", "9000")
    os.environ.setdefault("TESTING", "1")

    # Diagnostic debug prints
    print(f"PYTHONPATH: {sys.path}")
    print(f"urllib3.__file__: {urllib3.__file__}")
    print(f"Has urllib3.util? {hasattr(urllib3, 'util')}")
    try:
        from urllib3.util import Retry
        print(f"urllib3.util.Retry imported successfully: {Retry}")
    except Exception as e:
        print(f"Failed to import urllib3.util.Retry: {e}")

@pytest.fixture(autouse=True, scope="session")
def cleanup_test_env_vars():
    """Clean up testing flag after session."""
    yield
    os.environ.pop("TESTING", None)
