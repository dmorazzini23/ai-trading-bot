"""
Determinism and model specification locking module.

Ensures reproducible training and inference by managing random seeds,
model hashes, and data specifications.
"""

import hashlib
import json
import logging
import os
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np
logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducible results.

    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)

    # NumPy (if available)
    if HAS_NUMPY:
        np.random.seed(seed)

    # TensorFlow (if available)
    import tensorflow as tf
    # PyTorch (if available)
    import torch
    # LightGBM (if available)
    # AI-AGENT-REF: optional lightgbm import with shim
    try:
        import importlib
        lgb = importlib.import_module("lightgbm")  # noqa: F401
    except Exception:  # pragma: no cover - optional dep
        from ai_trading.thirdparty import lightgbm_compat as lgb  # noqa: F401
    # Set environment variables for additional determinism
    os.environ["PYTHONHASHSEED"] = str(seed)

    logger.info(f"Set random seeds to {seed} for reproducible results")


def hash_data(data: Any) -> str:
    """
    Generate hash for data (features, labels, etc.).

    Args:
        data: Data to hash (DataFrame, array, etc.)

    Returns:
        SHA256 hash string
    """
    try:
        if hasattr(data, "values"):
            # pandas DataFrame/Series
            content = data.values.tobytes()
        elif hasattr(data, "tobytes"):
            # numpy array
            content = data.tobytes()
        elif isinstance(data, list | tuple):
            # Convert to string and encode
            content = str(sorted(data)).encode("utf-8")
        elif isinstance(data, dict):
            # Sort keys for consistent hashing
            content = json.dumps(data, sort_keys=True).encode("utf-8")
        else:
            # Fallback to string representation
            content = str(data).encode("utf-8")

        return hashlib.sha256(content).hexdigest()[:16]  # First 16 chars

    except Exception as e:
        logger.warning(f"Failed to hash data: {e}")
        return "unknown"


def hash_features(feature_data) -> str:
    """
    Generate hash for feature data.

    Args:
        feature_data: Feature dataset

    Returns:
        Feature hash string
    """
    if feature_data is None:
        return "no_features"

    try:
        # Include feature names and sample of data
        if hasattr(feature_data, "columns"):
            # DataFrame - include column names and shape
            feature_names = sorted(feature_data.columns.tolist())
            shape_info = feature_data.shape
            sample_data = (
                feature_data.head(10) if len(feature_data) > 10 else feature_data
            )

            hash_content = {
                "columns": feature_names,
                "shape": shape_info,
                "sample_hash": hash_data(sample_data),
            }
        else:
            # Array - include shape and sample
            shape_info = getattr(feature_data, "shape", None)
            hash_content = {"shape": shape_info, "data_hash": hash_data(feature_data)}

        return hash_data(hash_content)

    except Exception as e:
        logger.warning(f"Failed to hash features: {e}")
        return "feature_hash_error"


def hash_labels(label_data) -> str:
    """
    Generate hash for label data.

    Args:
        label_data: Label dataset

    Returns:
        Label hash string
    """
    if label_data is None:
        return "no_labels"

    return hash_data(label_data)


def generate_spec_hash(
    feature_hash: str,
    label_hash: str,
    data_window: dict[str, Any],
    cost_model_version: str = "1.0",
    additional_params: dict[str, Any] | None = None,
) -> str:
    """
    Generate specification hash for model training/inference.

    Args:
        feature_hash: Hash of feature data
        label_hash: Hash of label data
        data_window: Data window specification
        cost_model_version: Cost model version
        additional_params: Additional parameters to include

    Returns:
        Combined specification hash
    """
    spec_dict = {
        "feature_hash": feature_hash,
        "label_hash": label_hash,
        "data_window": data_window,
        "cost_model_version": cost_model_version,
    }

    if additional_params:
        spec_dict.update(additional_params)

    return hash_data(spec_dict)


class ModelSpecification:
    """
    Model specification with hash checking and version control.
    """

    def __init__(self, spec_file: str = "meta.json"):
        """
        Initialize model specification.

        Args:
            spec_file: Path to specification file
        """
        self.spec_file = Path(spec_file)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Current specification
        self._spec: dict[str, Any] = {}
        self._is_locked = False

        # Load existing spec if available
        self._load_spec()

    def _load_spec(self) -> None:
        """Load specification from file."""
        if self.spec_file.exists():
            try:
                with open(self.spec_file) as f:
                    self._spec = json.load(f)

                self.logger.info(f"Loaded model specification from {self.spec_file}")

                # Check if spec is locked
                self._is_locked = self._spec.get("locked", False)

            except Exception as e:
                self.logger.error(f"Failed to load specification: {e}")
                self._spec = {}

    def _save_spec(self) -> None:
        """Save specification to file."""
        try:
            # Add metadata
            self._spec["created_at"] = datetime.now(UTC).isoformat()
            self._spec["version"] = self._spec.get("version", "1.0.0")

            with open(self.spec_file, "w") as f:
                json.dump(self._spec, f, indent=2)

            self.logger.info(f"Saved model specification to {self.spec_file}")

        except Exception as e:
            self.logger.error(f"Failed to save specification: {e}")

    def update_spec(
        self,
        feature_data=None,
        label_data=None,
        data_window: dict[str, Any] | None = None,
        cost_model_version: str = "1.0",
        training_params: dict[str, Any] | None = None,
        force: bool = False,
    ) -> str:
        """
        Update model specification with new training data.

        Args:
            feature_data: Training features
            label_data: Training labels
            data_window: Data window specification
            cost_model_version: Cost model version
            training_params: Training parameters
            force: Force update even if locked

        Returns:
            Specification hash
        """
        if self._is_locked and not force:
            self.logger.warning("Specification is locked, use force=True to override")
            return self._spec.get("spec_hash", "")

        # Generate hashes
        feature_hash = (
            hash_features(feature_data)
            if feature_data is not None
            else self._spec.get("feature_hash", "")
        )
        label_hash = (
            hash_labels(label_data)
            if label_data is not None
            else self._spec.get("label_hash", "")
        )

        # Use existing data_window if not provided
        if data_window is None:
            data_window = self._spec.get("data_window", {})

        # Generate specification hash
        spec_hash = generate_spec_hash(
            feature_hash=feature_hash,
            label_hash=label_hash,
            data_window=data_window,
            cost_model_version=cost_model_version,
            additional_params=training_params,
        )

        # Update specification
        self._spec.update(
            {
                "feature_hash": feature_hash,
                "label_hash": label_hash,
                "data_window": data_window,
                "cost_model_version": cost_model_version,
                "spec_hash": spec_hash,
                "training_params": training_params or {},
                "updated_at": datetime.now(UTC).isoformat(),
            }
        )

        # Save to file
        self._save_spec()

        self.logger.info(f"Updated model specification with hash: {spec_hash}")

        return spec_hash

    def lock_spec(self) -> None:
        """Lock specification to prevent changes."""
        self._spec["locked"] = True
        self._spec["locked_at"] = datetime.now(UTC).isoformat()
        self._is_locked = True
        self._save_spec()

        self.logger.info("Locked model specification")

    def unlock_spec(self) -> None:
        """Unlock specification to allow changes."""
        self._spec["locked"] = False
        self._spec["unlocked_at"] = datetime.now(UTC).isoformat()
        self._is_locked = False
        self._save_spec()

        self.logger.info("Unlocked model specification")

    def validate_compatibility(
        self,
        feature_data=None,
        label_data=None,
        data_window: dict[str, Any] | None = None,
        cost_model_version: str = "1.0",
        allow_override: bool = None,
    ) -> tuple[bool, str]:
        """
        Validate if current data is compatible with locked specification.

        Args:
            feature_data: Current features
            label_data: Current labels
            data_window: Current data window
            cost_model_version: Current cost model version
            allow_override: Override via environment variable

        Returns:
            Tuple of (is_compatible, reason)
        """
        if not self._spec:
            return True, "No existing specification"

        # Check environment override
        if allow_override is None:
            allow_override = (
                os.getenv("AI_TRADING_SPEC_OVERRIDE", "false").lower() == "true"
            )

        if allow_override:
            self.logger.warning(
                "Specification validation overridden by environment variable"
            )
            return True, "Override enabled"

        # Generate current hashes
        current_feature_hash = (
            hash_features(feature_data) if feature_data is not None else ""
        )
        current_label_hash = hash_labels(label_data) if label_data is not None else ""

        # Use existing data_window if not provided
        if data_window is None:
            data_window = {}

        current_spec_hash = generate_spec_hash(
            feature_hash=current_feature_hash,
            label_hash=current_label_hash,
            data_window=data_window,
            cost_model_version=cost_model_version,
        )

        # Compare with stored specification
        stored_spec_hash = self._spec.get("spec_hash", "")

        if current_spec_hash == stored_spec_hash:
            return True, "Specification matches"

        # Detailed mismatch analysis
        mismatches = []

        if current_feature_hash != self._spec.get("feature_hash", ""):
            mismatches.append("feature_hash")

        if current_label_hash != self._spec.get("label_hash", ""):
            mismatches.append("label_hash")

        if data_window != self._spec.get("data_window", {}):
            mismatches.append("data_window")

        if cost_model_version != self._spec.get("cost_model_version", ""):
            mismatches.append("cost_model_version")

        reason = f"Specification mismatch: {', '.join(mismatches)}"

        return False, reason

    def get_spec(self) -> dict[str, Any]:
        """Get current specification."""
        return self._spec.copy()

    def is_locked(self) -> bool:
        """Check if specification is locked."""
        return self._is_locked


# Global specification instance
_global_spec: ModelSpecification | None = None


def get_model_spec() -> ModelSpecification:
    """Get or create global model specification instance."""
    global _global_spec
    if _global_spec is None:
        _global_spec = ModelSpecification()
    return _global_spec


def ensure_deterministic_training(
    seed: int = 42,
    feature_data=None,
    label_data=None,
    data_window: dict[str, Any] | None = None,
    cost_model_version: str = "1.0",
    force_update: bool = False,
) -> tuple[bool, str]:
    """
    Ensure deterministic training setup.

    Args:
        seed: Random seed
        feature_data: Training features
        label_data: Training labels
        data_window: Data window specification
        cost_model_version: Cost model version
        force_update: Force specification update

    Returns:
        Tuple of (is_valid, message)
    """
    # Set random seeds
    set_random_seeds(seed)

    # Validate specification
    spec = get_model_spec()

    if force_update:
        spec.update_spec(
            feature_data=feature_data,
            label_data=label_data,
            data_window=data_window,
            cost_model_version=cost_model_version,
            force=True,
        )
        return True, "Specification updated"

    # Check compatibility
    is_compatible, reason = spec.validate_compatibility(
        feature_data=feature_data,
        label_data=label_data,
        data_window=data_window,
        cost_model_version=cost_model_version,
    )

    if not is_compatible:
        logger.error(f"Training validation failed: {reason}")
        return False, reason

    return True, "Deterministic training setup complete"


def lock_model_spec() -> None:
    """Lock model specification for production."""
    spec = get_model_spec()
    spec.lock_spec()


def unlock_model_spec() -> None:
    """Unlock model specification for development."""
    spec = get_model_spec()
    spec.unlock_spec()
