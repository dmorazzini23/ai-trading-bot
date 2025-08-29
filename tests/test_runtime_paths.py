"""
Test startup permissions and runtime paths.

Validates that the application has write permissions to required directories.
"""
import pytest
import stat


def test_runtime_paths_writable():
    """Test that runtime paths are writable."""
    from ai_trading import paths

    # Test DATA_DIR is writable
    test_file = paths.DATA_DIR / "test_write.tmp"
    try:
        test_file.write_text("test")
        assert test_file.exists()
        test_file.unlink()
    except PermissionError:
        pytest.fail(f"DATA_DIR {paths.DATA_DIR} is not writable")

    # Test LOG_DIR is writable
    test_file = paths.LOG_DIR / "test_write.tmp"
    try:
        test_file.write_text("test")
        assert test_file.exists()
        test_file.unlink()
    except PermissionError:
        pytest.fail(f"LOG_DIR {paths.LOG_DIR} is not writable")

    # Test CACHE_DIR is writable
    test_file = paths.CACHE_DIR / "test_write.tmp"
    try:
        test_file.write_text("test")
        assert test_file.exists()
        test_file.unlink()
    except PermissionError:
        pytest.fail(f"CACHE_DIR {paths.CACHE_DIR} is not writable")
    assert stat.S_IMODE(paths.CACHE_DIR.stat().st_mode) == 0o700


def test_cache_dir_falls_back(monkeypatch, tmp_path):
    """Cache dir falls back to a temp path if configured location is read-only."""
    import errno
    import importlib
    import tempfile
    from pathlib import Path

    monkeypatch.setenv('AI_TRADING_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('AI_TRADING_LOG_DIR', str(tmp_path / 'log'))
    unwritable = tmp_path / 'readonly' / 'cache'
    monkeypatch.setenv('AI_TRADING_CACHE_DIR', str(unwritable))

    orig_mkdir = Path.mkdir

    def fake_mkdir(self, parents=True, exist_ok=False):
        if self == unwritable:
            raise OSError(errno.EROFS, 'Read-only file system')
        return orig_mkdir(self, parents=parents, exist_ok=exist_ok)

    monkeypatch.setattr(Path, 'mkdir', fake_mkdir)
    import ai_trading.paths as paths
    importlib.reload(paths)

    fallback = Path(tempfile.gettempdir()) / paths.APP_NAME
    assert paths.CACHE_DIR == fallback
    assert fallback.exists()
    assert stat.S_IMODE(fallback.stat().st_mode) == 0o700


def test_paths_module_imports():
    """Test that paths module can be imported."""
    from ai_trading import paths

    assert hasattr(paths, 'DATA_DIR')
    assert hasattr(paths, 'LOG_DIR')
    assert hasattr(paths, 'CACHE_DIR')

    # Ensure directories exist
    assert paths.DATA_DIR.exists()
    assert paths.LOG_DIR.exists()
    assert paths.CACHE_DIR.exists()


def test_http_utilities_available():
    """Test that HTTP utilities are available."""
    from ai_trading.utils import http

    assert hasattr(http, 'get')
    assert hasattr(http, 'post')
    assert hasattr(http, 'put')
    assert hasattr(http, 'delete')

# AI-AGENT-REF: Startup permission tests for runtime path validation
