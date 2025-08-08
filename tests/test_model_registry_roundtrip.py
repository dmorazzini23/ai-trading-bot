"""Test model registry register → latest_for → load_model workflow."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestModelRegistryRoundtrip(unittest.TestCase):
    """Test complete model registry workflow."""

    def setUp(self):
        """Set up test with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = self.temp_dir
        
        # Mock a simple model
        self.model = MagicMock()
        self.model.__class__.__name__ = "TestModel"
        self.model.get_params = MagicMock(return_value={"param1": "value1"})
        
        self.metadata = {
            "test_accuracy": 0.95,
            "training_samples": 1000,
            "validation_samples": 200
        }
        
        self.feature_spec = {
            "features": ["feature1", "feature2", "feature3"],
            "scaler": "StandardScaler"
        }

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_register_latest_for_load_workflow(self):
        """Test complete register → latest_for → load_model workflow."""
        from ai_trading.model_registry import ModelRegistry
        
        registry = ModelRegistry(self.registry_path)
        
        # 1. Register model
        model_id = registry.register_model(
            model=self.model,
            strategy="test_strategy",
            model_type="test_model",
            metadata=self.metadata,
            feature_spec=self.feature_spec,
            tags=["test", "validation"]
        )
        
        # Verify model_id is returned
        self.assertIsInstance(model_id, str)
        self.assertTrue(len(model_id) > 0)
        
        # 2. Use latest_for to get model
        loaded_model, loaded_metadata, loaded_id = registry.latest_for("test_strategy")
        
        # 3. Verify round-trip integrity
        self.assertEqual(loaded_id, model_id)
        self.assertEqual(loaded_metadata["strategy"], "test_strategy")
        self.assertEqual(loaded_metadata["model_type"], "test_model")
        self.assertEqual(loaded_metadata["test_accuracy"], 0.95)
        self.assertEqual(loaded_metadata["feature_spec"], self.feature_spec)
        self.assertIn("test", loaded_metadata["tags"])
        self.assertIn("validation", loaded_metadata["tags"])
        
        # 4. Verify model object is properly deserialized
        self.assertIsNotNone(loaded_model)
        # Note: In a real test with actual models, you'd verify model attributes

    def test_index_persistence(self):
        """Test that registry index persists across instances."""
        from ai_trading.model_registry import ModelRegistry
        
        # Register model with first registry instance
        registry1 = ModelRegistry(self.registry_path)
        model_id = registry1.register_model(
            model=self.model,
            strategy="persistence_test",
            model_type="test_model",
            metadata=self.metadata
        )
        
        # Create new registry instance and verify model exists
        registry2 = ModelRegistry(self.registry_path)
        loaded_model, loaded_metadata, loaded_id = registry2.latest_for("persistence_test")
        
        self.assertEqual(loaded_id, model_id)
        self.assertEqual(loaded_metadata["strategy"], "persistence_test")
        
        # Verify index file exists and contains correct data
        index_file = Path(self.registry_path) / "registry_index.json"
        self.assertTrue(index_file.exists())
        
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        self.assertIn(model_id, index_data)
        self.assertEqual(index_data[model_id]["strategy"], "persistence_test")

    def test_metadata_round_trip(self):
        """Test that all metadata is preserved in round-trip."""
        from ai_trading.model_registry import ModelRegistry
        
        registry = ModelRegistry(self.registry_path)
        
        # Enhanced metadata with various data types
        complex_metadata = {
            "string_value": "test",
            "int_value": 42,
            "float_value": 3.14159,
            "bool_value": True,
            "list_value": [1, 2, 3],
            "dict_value": {"nested": "data"},
            "none_value": None
        }
        
        model_id = registry.register_model(
            model=self.model,
            strategy="metadata_test",
            model_type="test_model",
            metadata=complex_metadata,
            feature_spec=self.feature_spec
        )
        
        # Load and verify all metadata
        loaded_model, loaded_metadata, loaded_id = registry.latest_for("metadata_test")
        
        # Check all original metadata is preserved
        for key, value in complex_metadata.items():
            self.assertEqual(loaded_metadata[key], value)
        
        # Check feature_spec is preserved
        self.assertEqual(loaded_metadata["feature_spec"], self.feature_spec)

    def test_dataset_hash_verification_workflow(self):
        """Test dataset hash verification with verify_dataset_hash=True."""
        from ai_trading.model_registry import ModelRegistry
        
        registry = ModelRegistry(self.registry_path)
        
        # Create mock dataset files
        dataset_file1 = Path(self.temp_dir) / "dataset1.csv"
        dataset_file2 = Path(self.temp_dir) / "dataset2.csv"
        
        dataset_file1.write_text("col1,col2\n1,2\n3,4\n")
        dataset_file2.write_text("col1,col2\n5,6\n7,8\n")
        
        dataset_paths = [str(dataset_file1), str(dataset_file2)]
        
        # Register model with dataset paths
        model_id = registry.register_model(
            model=self.model,
            strategy="dataset_test",
            model_type="test_model",
            metadata=self.metadata,
            dataset_paths=dataset_paths
        )
        
        # Load model with same dataset paths - should succeed
        loaded_model, loaded_metadata = registry.load_model(
            model_id, 
            verify_dataset_hash=True,
            current_dataset_paths=dataset_paths
        )
        
        self.assertIsNotNone(loaded_model)
        self.assertIn("dataset_hash", loaded_metadata)
        self.assertEqual(loaded_metadata["dataset_paths"], dataset_paths)
        
        # Modify dataset and try to load - should fail
        dataset_file1.write_text("col1,col2\n9,10\n11,12\n")  # Changed content
        
        with self.assertRaises(ValueError) as context:
            registry.load_model(
                model_id,
                verify_dataset_hash=True,
                current_dataset_paths=dataset_paths
            )
        
        self.assertIn("Dataset hash mismatch", str(context.exception))

    def test_multiple_strategies_workflow(self):
        """Test workflow with multiple strategies and model selection."""
        from ai_trading.model_registry import ModelRegistry
        
        registry = ModelRegistry(self.registry_path)
        
        # Register models for different strategies
        strategies = ["momentum", "mean_reversion", "breakout"]
        model_ids = {}
        
        for strategy in strategies:
            model_ids[strategy] = registry.register_model(
                model=self.model,
                strategy=strategy,
                model_type="test_model",
                metadata={**self.metadata, "strategy_specific": f"{strategy}_data"}
            )
        
        # Verify each strategy returns its own model
        for strategy in strategies:
            loaded_model, loaded_metadata, loaded_id = registry.latest_for(strategy)
            
            self.assertEqual(loaded_id, model_ids[strategy])
            self.assertEqual(loaded_metadata["strategy"], strategy)
            self.assertEqual(loaded_metadata["strategy_specific"], f"{strategy}_data")
        
        # Verify list_models returns all models
        all_models = registry.list_models()
        self.assertEqual(len(all_models), 3)
        
        # Verify filtering by strategy works
        momentum_models = registry.list_models(strategy="momentum")
        self.assertEqual(len(momentum_models), 1)
        self.assertEqual(momentum_models[0]["strategy"], "momentum")


if __name__ == "__main__":
    unittest.main()