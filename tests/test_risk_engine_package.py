"""
Test that RiskEngine has been properly promoted to package structure.
"""

import os
import unittest


class TestRiskEnginePackage(unittest.TestCase):
    """Test RiskEngine package promotion."""

    def test_risk_engine_import_from_package(self):
        """Test that RiskEngine can be imported from ai_trading.risk."""
        from ai_trading.risk import RiskEngine

        # Verify it's the correct class
        self.assertEqual(RiskEngine.__name__, "RiskEngine")
        self.assertIn("ai_trading.risk.engine", RiskEngine.__module__)

    def test_risk_engine_resolver_uses_package(self):
        """Test that the resolver prefers the package version."""
        import importlib.util

        # Load imports module directly to avoid heavy dependencies
        spec = importlib.util.spec_from_file_location(
            'imports', 'ai_trading/utils/imports.py'
        )
        imports_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imports_module)

        # Test resolution
        cls = imports_module.resolve_risk_engine_cls()
        self.assertIsNotNone(cls)
        self.assertEqual(cls.__name__, "RiskEngine")
        self.assertIn("ai_trading.risk.engine", cls.__module__)
        self.assertNotIn("scripts", cls.__module__)

    def test_scripts_fallback_disabled_by_default(self):
        """Test that scripts fallback is disabled without DEV_ALLOW_SCRIPTS."""
        import importlib.util

        # Ensure DEV_ALLOW_SCRIPTS is not set
        os.environ.pop("DEV_ALLOW_SCRIPTS", None)

        # Load imports module
        spec = importlib.util.spec_from_file_location(
            'imports', 'ai_trading/utils/imports.py'
        )
        imports_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(imports_module)

        # Resolution should use package, not scripts
        cls = imports_module.resolve_risk_engine_cls()
        self.assertIsNotNone(cls)
        self.assertNotIn("scripts", cls.__module__)

    def test_update_exposure_requires_context(self):
        """Test that update_exposure requires context parameter."""
        from ai_trading.risk import RiskEngine

        re = RiskEngine()

        # Should raise RuntimeError without context
        with self.assertRaises(RuntimeError) as cm:
            re.update_exposure()

        self.assertIn("context is required", str(cm.exception))

    def test_update_exposure_works_with_context(self):
        """Test that update_exposure works with context parameter."""
        from ai_trading.risk import RiskEngine

        re = RiskEngine()

        # Mock context with API
        class MockContext:
            class MockAPI:
                def get_all_positions(self):
                    return []

                def get_account(self):
                    class MockAccount:
                        equity = "10000"
                    return MockAccount()

            api = MockAPI()

        # Should not raise with proper context
        try:
            re.update_exposure(context=MockContext())
        except Exception as e:
            self.fail(f"update_exposure failed with context: {e}")


if __name__ == "__main__":
    unittest.main()
