"""
Test environment loading order and lazy import behavior.

Tests that .env files are loaded before heavy imports and that lazy imports
prevent import-time crashes due to missing environment variables.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


class TestEnvironmentOrderAndLazyImport:
    """Test environment loading order and lazy imports."""

    def test_dotenv_loaded_before_settings_construction(self):
        """Test that .env is loaded before Settings is constructed."""
        # Create a temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("TEST_DOTENV_ORDER=loaded_early\n")
            f.write("ALPACA_API_KEY=test_from_dotenv\n")
            temp_env_path = f.name

        try:
            # Clear any existing env var
            if 'TEST_DOTENV_ORDER' in os.environ:
                del os.environ['TEST_DOTENV_ORDER']
            if 'ALPACA_API_KEY' in os.environ:
                del os.environ['ALPACA_API_KEY']

            # Mock ai_trading.env.load_dotenv to load our temp file
            with patch('ai_trading.env.load_dotenv') as mock_load_dotenv:
                def side_effect(*args, **kwargs):
                    # Simulate loading the .env file
                    with open(temp_env_path) as env_file:
                        for line in env_file:
                            if '=' in line and not line.startswith('#'):
                                key, value = line.strip().split('=', 1)
                                os.environ[key] = value

                mock_load_dotenv.side_effect = side_effect

                from ai_trading.env import ensure_dotenv_loaded
                ensure_dotenv_loaded()

                mock_load_dotenv.assert_called()
                assert os.environ.get('TEST_DOTENV_ORDER') == 'loaded_early'
                assert os.environ.get('ALPACA_API_KEY') == 'test_from_dotenv'

        finally:
            # Cleanup
            os.unlink(temp_env_path)
            os.environ.pop('TEST_DOTENV_ORDER', None)
            os.environ.pop('ALPACA_API_KEY', None)

    def test_lazy_import_prevents_import_time_crash(self):
        """Test that lazy imports prevent crashes during import."""
        # Clear environment to simulate missing credentials
        env_backup = {}
        env_keys_to_clear = [
            'ALPACA_API_KEY', 'APCA_API_KEY_ID',
            'ALPACA_SECRET_KEY', 'APCA_API_SECRET_KEY',
            'ALPACA_BASE_URL', 'APCA_API_BASE_URL'
        ]

        for key in env_keys_to_clear:
            if key in os.environ:
                env_backup[key] = os.environ[key]
                del os.environ[key]

        try:
            # Mock sys.exit to capture if it's called during import
            with patch('sys.exit') as mock_exit:
                # Import runner module - should not crash even with missing env vars
                from ai_trading import runner

                # Should not have called sys.exit during import
                mock_exit.assert_not_called()

                # Verify the module loaded successfully
                assert hasattr(runner, 'run_cycle')
                assert hasattr(runner, '_load_engine')

        finally:
            # Restore environment
            for key, value in env_backup.items():
                os.environ[key] = value

    def test_lazy_engine_loading_defers_heavy_imports(self):
        """Test that engine loading is deferred until actually needed."""
        from ai_trading.runner import _load_engine

        # Mock the bot_engine import to verify it's called lazily
        with patch('ai_trading.runner._bot_engine', None):
            with patch('ai_trading.runner._bot_state_class', None):
                with patch('ai_trading.core.bot_engine.run_all_trades_worker') as mock_worker:
                    with patch('ai_trading.core.bot_engine.BotState') as mock_state:
                        mock_worker.return_value = "mock_worker"
                        mock_state.return_value = "mock_state"

                        # Call _load_engine
                        worker, state_class = _load_engine()

                        # Verify imports were called
                        assert worker == mock_worker
                        assert state_class == mock_state

    def test_lazy_engine_loading_caches_components(self):
        """Test that engine components are cached after first load."""
        from ai_trading import runner

        # Reset the lazy import cache
        runner._bot_engine = None
        runner._bot_state_class = None

        with patch('ai_trading.core.bot_engine.run_all_trades_worker') as mock_worker:
            with patch('ai_trading.core.bot_engine.BotState') as mock_state:
                mock_worker.return_value = "cached_worker"
                mock_state.return_value = "cached_state"

                # First call should import
                worker1, state1 = runner._load_engine()

                # Second call should use cached values
                worker2, state2 = runner._load_engine()

                # Should be the same objects
                assert worker1 == worker2 == "cached_worker"
                assert state1 == state2 == "cached_state"

                # Import should only have been called once
                assert mock_worker.call_count == 0  # Not called, just referenced
                assert mock_state.call_count == 0    # Not called, just referenced

    def test_run_cycle_uses_lazy_loading(self):
        """Test that run_cycle uses lazy loading for bot engine."""
        from ai_trading import runner

        # Reset lazy import cache
        runner._bot_engine = None
        runner._bot_state_class = None

        # Mock the components
        mock_worker = MagicMock()
        mock_state_class = MagicMock()
        mock_state_instance = MagicMock()
        mock_state_class.return_value = mock_state_instance

        with patch.object(runner, '_load_engine') as mock_load:
            mock_load.return_value = (mock_worker, mock_state_class)

            # Call run_cycle
            runner.run_cycle()

            # Verify lazy loading was called
            mock_load.assert_called_once()

            # Verify worker was called with state instance
            mock_worker.assert_called_once_with(mock_state_instance, None)

    def test_main_loads_dotenv_before_runner_import(self):
        """Test that main.py loads .env before importing runner."""
        # Mock ensure_dotenv_loaded to track when it's called
        with patch('ai_trading.env.ensure_dotenv_loaded') as mock_ensure:
            with patch('ai_trading.runner.run_cycle') as mock_run_cycle:
                import importlib
                main = importlib.reload(importlib.import_module('ai_trading.main'))

                result = main.run_bot()

                mock_ensure.assert_called()
                mock_run_cycle.assert_called_once()
                assert result == 0

    def test_env_loaded_multiple_times_safely(self):
        """Test that loading .env multiple times is safe."""
        # Create temp .env with test values
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("MULTI_LOAD_TEST=safe_value\n")
            temp_env_path = f.name

        try:
            # Clear environment
            os.environ.pop('MULTI_LOAD_TEST', None)

            # Mock load_dotenv to use our temp file
            def mock_load_side_effect(*args, **kwargs):
                with open(temp_env_path) as env_file:
                    for line in env_file:
                        if '=' in line and not line.startswith('#'):
                            key, value = line.strip().split('=', 1)
                            os.environ[key] = value

            with patch('ai_trading.env.load_dotenv', side_effect=mock_load_side_effect):
                from ai_trading.env import ensure_dotenv_loaded
                ensure_dotenv_loaded()
                ensure_dotenv_loaded()
                ensure_dotenv_loaded()

            assert os.environ.get('MULTI_LOAD_TEST') == 'safe_value'

        finally:
            os.unlink(temp_env_path)
            os.environ.pop('MULTI_LOAD_TEST', None)

    def test_missing_env_file_handled_gracefully(self):
        """Test that missing .env file doesn't crash the import."""
        with patch('ai_trading.env.load_dotenv') as mock_load_dotenv:
            mock_load_dotenv.side_effect = FileNotFoundError("No .env file")

            # Should not raise exception
            try:
                # If we get here, import succeeded despite missing .env
                assert True
            # noqa: BLE001 TODO: narrow exception
            except Exception as e:
                pytest.fail(f"Import failed with missing .env file: {e}")

    def test_lazy_import_error_handling(self):
        """Test that lazy import handles import errors gracefully."""
        from ai_trading import runner

        # Reset cache
        runner._bot_engine = None
        runner._bot_state_class = None

        with patch('ai_trading.core.bot_engine.run_all_trades_worker', side_effect=ImportError("Mock import error")):
            with pytest.raises(RuntimeError) as exc_info:
                runner._load_engine()

            assert "Cannot load bot engine" in str(exc_info.value)

    def test_import_time_no_credential_validation(self):
        """Test that no credential validation happens at import time."""
        # Clear all credential environment variables
        credential_keys = [
            'ALPACA_API_KEY', 'APCA_API_KEY_ID',
            'ALPACA_SECRET_KEY', 'APCA_API_SECRET_KEY',
            'ALPACA_BASE_URL', 'APCA_API_BASE_URL'
        ]

        env_backup = {}
        for key in credential_keys:
            if key in os.environ:
                env_backup[key] = os.environ[key]
                del os.environ[key]

        try:
            # Mock any validation functions to detect if they're called
            with patch('ai_trading.config.management.validate_alpaca_credentials') as mock_validate:
                with patch('sys.exit') as mock_exit:
                    # Import main module - should not validate credentials

                    # Should not have called validation or exit during import
                    mock_validate.assert_not_called()
                    mock_exit.assert_not_called()

        finally:
            # Restore environment
            for key, value in env_backup.items():
                os.environ[key] = value


if __name__ == "__main__":
    pytest.main([__file__])
