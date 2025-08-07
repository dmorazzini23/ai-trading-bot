"""
Model registry for tracking and managing trained models.

Provides centralized model storage, versioning, and retrieval
with metadata tracking for trading models.
"""

import os
import json
import pickle
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import logging

# Use the centralized logger as per AGENTS.md
try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Centralized model registry for trading models.
    
    Manages model storage, versioning, metadata, and retrieval
    across different strategies and time periods.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize model registry.
        
        Args:
            base_path: Base directory for model storage (overridable via MODEL_REGISTRY_DIR env var)
        """
        if base_path is None:
            base_path = os.getenv("MODEL_REGISTRY_DIR", "models")
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Registry index
        self.index_file = self.base_path / "registry_index.json"
        self.model_index = self._load_index()
        
        logger.info(f"ModelRegistry initialized at {self.base_path}")
    
    def register_model(
        self,
        model: Any,
        strategy: str,
        model_type: str,
        metadata: Dict[str, Any],
        feature_spec: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        replace_existing: bool = False
    ) -> str:
        """
        Register a new model in the registry.
        
        Args:
            model: Trained model object
            strategy: Strategy name
            model_type: Type of model (e.g., 'lightgbm', 'xgboost', 'rl')
            metadata: Model metadata
            feature_spec: Feature specification
            metrics: Model performance metrics
            tags: Optional tags for categorization
            replace_existing: Whether to replace existing model with same hash
            
        Returns:
            Model ID (hash-based identifier)
        """
        try:
            # Generate model hash
            model_hash = self._generate_model_hash(model, metadata, feature_spec)
            
            # Check if model already exists
            if model_hash in self.model_index and not replace_existing:
                logger.info(f"Model {model_hash} already exists in registry")
                return model_hash
            
            # Create model directory structure
            timestamp = datetime.now().strftime("%Y-%m-%d")
            model_dir = self.base_path / strategy / timestamp / model_hash
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model_file = model_dir / "model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metadata
            full_metadata = {
                "model_hash": model_hash,
                "strategy": strategy,
                "model_type": model_type,
                "registration_time": datetime.now().isoformat(),
                "model_file": "model.pkl",
                "tags": tags or [],
                **metadata
            }
            
            metadata_file = model_dir / "meta.json"
            with open(metadata_file, 'w') as f:
                json.dump(full_metadata, f, indent=2, default=str)
            
            # Save feature specification
            if feature_spec:
                feature_file = model_dir / "feature_spec.json"
                with open(feature_file, 'w') as f:
                    json.dump(feature_spec, f, indent=2, default=str)
            
            # Save metrics
            if metrics:
                metrics_file = model_dir / "metrics.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2, default=str)
            
            # Update registry index
            self.model_index[model_hash] = {
                "strategy": strategy,
                "model_type": model_type,
                "path": str(model_dir),
                "registration_time": full_metadata["registration_time"],
                "tags": tags or [],
                "active": True
            }
            
            self._save_index()
            
            logger.info(f"Model registered: {model_hash} for strategy {strategy}")
            return model_hash
            
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    def load_model(self, model_id: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Load model and metadata by ID.
        
        Args:
            model_id: Model hash ID
            
        Returns:
            Tuple of (model, metadata)
        """
        try:
            if model_id not in self.model_index:
                raise ValueError(f"Model {model_id} not found in registry")
            
            model_info = self.model_index[model_id]
            model_dir = Path(model_info["path"])
            
            # Load model
            model_file = model_dir / "model.pkl"
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            # Load metadata
            metadata_file = model_dir / "meta.json"
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load additional files if they exist
            feature_file = model_dir / "feature_spec.json"
            if feature_file.exists():
                with open(feature_file, 'r') as f:
                    metadata["feature_spec"] = json.load(f)
            
            metrics_file = model_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metadata["metrics"] = json.load(f)
            
            logger.debug(f"Model loaded: {model_id}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    def load_latest_by_strategy(
        self,
        strategy: str,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Tuple[Any, Dict[str, Any], str]:
        """
        Load latest model for a strategy.
        
        Args:
            strategy: Strategy name
            model_type: Optional model type filter
            tags: Optional tags filter
            
        Returns:
            Tuple of (model, metadata, model_id)
        """
        try:
            # Find matching models
            candidates = []
            for model_id, info in self.model_index.items():
                if not info.get("active", True):
                    continue
                    
                if info["strategy"] != strategy:
                    continue
                    
                if model_type and info.get("model_type") != model_type:
                    continue
                    
                if tags:
                    model_tags = set(info.get("tags", []))
                    if not set(tags).issubset(model_tags):
                        continue
                
                candidates.append((model_id, info))
            
            if not candidates:
                raise ValueError(f"No models found for strategy {strategy}")
            
            # Sort by registration time (latest first)
            candidates.sort(key=lambda x: x[1]["registration_time"], reverse=True)
            latest_id, _ = candidates[0]
            
            model, metadata = self.load_model(latest_id)
            return model, metadata, latest_id
            
        except Exception as e:
            logger.error(f"Error loading latest model for strategy {strategy}: {e}")
            raise
    
    def list_models(
        self,
        strategy: Optional[str] = None,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        List models matching criteria.
        
        Args:
            strategy: Optional strategy filter
            model_type: Optional model type filter
            tags: Optional tags filter
            active_only: Whether to include only active models
            
        Returns:
            List of model information dictionaries
        """
        try:
            results = []
            
            for model_id, info in self.model_index.items():
                if active_only and not info.get("active", True):
                    continue
                    
                if strategy and info.get("strategy") != strategy:
                    continue
                    
                if model_type and info.get("model_type") != model_type:
                    continue
                    
                if tags:
                    model_tags = set(info.get("tags", []))
                    if not set(tags).issubset(model_tags):
                        continue
                
                # Load basic metadata
                try:
                    metadata_file = Path(info["path"]) / "meta.json"
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    result = {
                        "model_id": model_id,
                        "strategy": info["strategy"],
                        "model_type": info.get("model_type"),
                        "registration_time": info["registration_time"],
                        "tags": info.get("tags", []),
                        "path": info["path"],
                        "metadata": metadata
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error loading metadata for model {model_id}: {e}")
                    continue
            
            # Sort by registration time (latest first)
            results.sort(key=lambda x: x["registration_time"], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def deactivate_model(self, model_id: str) -> None:
        """
        Deactivate a model (soft delete).
        
        Args:
            model_id: Model hash ID
        """
        try:
            if model_id not in self.model_index:
                raise ValueError(f"Model {model_id} not found in registry")
            
            self.model_index[model_id]["active"] = False
            self._save_index()
            
            logger.info(f"Model deactivated: {model_id}")
            
        except Exception as e:
            logger.error(f"Error deactivating model {model_id}: {e}")
            raise
    
    def delete_model(self, model_id: str, permanent: bool = False) -> None:
        """
        Delete a model from registry.
        
        Args:
            model_id: Model hash ID
            permanent: Whether to permanently delete files
        """
        try:
            if model_id not in self.model_index:
                raise ValueError(f"Model {model_id} not found in registry")
            
            model_info = self.model_index[model_id]
            
            if permanent:
                # Delete model files
                model_dir = Path(model_info["path"])
                if model_dir.exists():
                    import shutil
                    shutil.rmtree(model_dir)
                    logger.info(f"Model files deleted: {model_dir}")
            
            # Remove from index
            del self.model_index[model_id]
            self._save_index()
            
            logger.info(f"Model removed from registry: {model_id}")
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            raise
    
    def add_model_tags(self, model_id: str, tags: List[str]) -> None:
        """
        Add tags to a model.
        
        Args:
            model_id: Model hash ID
            tags: Tags to add
        """
        try:
            if model_id not in self.model_index:
                raise ValueError(f"Model {model_id} not found in registry")
            
            current_tags = set(self.model_index[model_id].get("tags", []))
            current_tags.update(tags)
            self.model_index[model_id]["tags"] = list(current_tags)
            
            self._save_index()
            logger.debug(f"Tags added to model {model_id}: {tags}")
            
        except Exception as e:
            logger.error(f"Error adding tags to model {model_id}: {e}")
            raise
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary across all models.
        
        Returns:
            Performance summary statistics
        """
        try:
            summary = {
                "total_models": len(self.model_index),
                "active_models": sum(1 for info in self.model_index.values() if info.get("active", True)),
                "strategies": {},
                "model_types": {},
                "best_performers": {}
            }
            
            # Collect statistics
            for model_id, info in self.model_index.items():
                if not info.get("active", True):
                    continue
                
                strategy = info["strategy"]
                model_type = info.get("model_type", "unknown")
                
                # Count by strategy
                summary["strategies"][strategy] = summary["strategies"].get(strategy, 0) + 1
                
                # Count by model type
                summary["model_types"][model_type] = summary["model_types"].get(model_type, 0) + 1
                
                # Load metrics for performance analysis
                try:
                    metrics_file = Path(info["path"]) / "metrics.json"
                    if metrics_file.exists():
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        
                        # Track best performers by strategy
                        if strategy not in summary["best_performers"]:
                            summary["best_performers"][strategy] = {
                                "model_id": model_id,
                                "metrics": metrics
                            }
                        else:
                            # Simple comparison - use mean_reward if available
                            current_score = metrics.get("mean_reward", 0)
                            best_score = summary["best_performers"][strategy]["metrics"].get("mean_reward", 0)
                            
                            if current_score > best_score:
                                summary["best_performers"][strategy] = {
                                    "model_id": model_id,
                                    "metrics": metrics
                                }
                                
                except Exception as e:
                    logger.debug(f"Error loading metrics for model {model_id}: {e}")
                    continue
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {}
    
    def _generate_model_hash(
        self,
        model: Any,
        metadata: Dict[str, Any],
        feature_spec: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate unique hash for model."""
        try:
            # Create a string representation for hashing
            hash_components = []
            
            # Model type and key parameters
            model_repr = str(type(model).__name__)
            if hasattr(model, 'get_params'):
                # Scikit-learn style
                params = model.get_params()
                hash_components.append(str(sorted(params.items())))
            elif hasattr(model, '__dict__'):
                # General object attributes
                attrs = {k: v for k, v in model.__dict__.items() if not k.startswith('_')}
                hash_components.append(str(sorted(attrs.items())))
            
            hash_components.append(model_repr)
            
            # Metadata
            metadata_clean = {k: v for k, v in metadata.items() if k not in ['timestamp', 'training_time']}
            hash_components.append(str(sorted(metadata_clean.items())))
            
            # Feature specification
            if feature_spec:
                hash_components.append(str(sorted(feature_spec.items())))
            
            # Generate hash
            hash_string = '|'.join(hash_components)
            model_hash = hashlib.md5(hash_string.encode()).hexdigest()[:16]
            
            return model_hash
            
        except Exception as e:
            logger.error(f"Error generating model hash: {e}")
            # Fallback to timestamp-based hash
            return hashlib.md5(str(datetime.now()).encode()).hexdigest()[:16]
    
    def _load_index(self) -> Dict[str, Any]:
        """Load registry index."""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Error loading registry index: {e}")
            return {}
    
    def _save_index(self) -> None:
        """Save registry index."""
        try:
            with open(self.index_file, 'w') as f:
                json.dump(self.model_index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving registry index: {e}")


# Convenience functions
def register_model(
    model: Any,
    strategy: str,
    model_type: str,
    metadata: Dict[str, Any],
    registry_path: str = "models",
    **kwargs
) -> str:
    """
    Convenience function to register a model.
    
    Args:
        model: Trained model
        strategy: Strategy name
        model_type: Model type
        metadata: Model metadata
        registry_path: Registry base path
        **kwargs: Additional arguments for registry.register_model()
        
    Returns:
        Model ID
    """
    registry = ModelRegistry(registry_path)
    return registry.register_model(model, strategy, model_type, metadata, **kwargs)


def load_latest_model(
    strategy: str,
    registry_path: str = "models",
    **kwargs
) -> Tuple[Any, Dict[str, Any], str]:
    """
    Convenience function to load latest model for strategy.
    
    Args:
        strategy: Strategy name
        registry_path: Registry base path
        **kwargs: Additional arguments for registry.load_latest_by_strategy()
        
    Returns:
        Tuple of (model, metadata, model_id)
    """
    registry = ModelRegistry(registry_path)
    return registry.load_latest_by_strategy(strategy, **kwargs)