"""
Hyperparameters schema validation with versioning support.

Provides pydantic model for hyperparams.json validation and ensures
schema compatibility across different versions of the trading system.
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

# AI-AGENT-REF: graceful fallback for missing pydantic
try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for testing environments
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    def Field(*args, **kwargs):
        return kwargs.get('default')
    
    def validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    PYDANTIC_AVAILABLE = False

logger = logging.getLogger(__name__)

# Current schema version - increment when making breaking changes
HYPERPARAMS_SCHEMA_VERSION = "1.0.0"

# Rate limited warning tracker (once per hour)
_last_missing_warning = 0
_warning_interval = 3600  # 1 hour


class HyperparametersSchema(BaseModel):
    """
    Hyperparameters schema with version validation.
    
    Defines the expected structure and validation rules for
    hyperparams.json configuration file.
    """
    
    # Schema metadata
    schema_version: str = Field(
        default=HYPERPARAMS_SCHEMA_VERSION,
        description="Schema version for compatibility checking"
    )
    
    # Trading strategy parameters
    buy_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Signal strength threshold for buy decisions"
    )
    sell_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Signal strength threshold for sell decisions"
    )
    
    # Risk management parameters
    max_position_size: float = Field(
        default=0.1,
        ge=0.001,
        le=1.0,
        description="Maximum position size as fraction of portfolio"
    )
    stop_loss_pct: float = Field(
        default=0.02,
        ge=0.001,
        le=0.5,
        description="Stop loss percentage"
    )
    take_profit_pct: float = Field(
        default=0.04,
        ge=0.001,
        le=1.0,
        description="Take profit percentage"
    )
    
    # Signal parameters
    signal_lookback_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Lookback period for signal calculation"
    )
    signal_confirmation_bars: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of bars for signal confirmation"
    )
    
    # Portfolio management
    max_positions: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of concurrent positions"
    )
    rebalance_frequency: str = Field(
        default="daily",
        description="Portfolio rebalancing frequency"
    )
    
    # Model parameters
    model_retrain_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Days between model retraining"
    )
    model_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum model confidence for trading"
    )
    
    # Execution parameters
    order_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Order timeout in seconds"
    )
    slippage_tolerance_bps: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum slippage tolerance in basis points"
    )
    
    # Advanced features
    use_sentiment_analysis: bool = Field(
        default=True,
        description="Enable sentiment analysis features"
    )
    use_technical_indicators: bool = Field(
        default=True,
        description="Enable technical indicator features"
    )
    use_fundamental_data: bool = Field(
        default=False,
        description="Enable fundamental data features"
    )
    
    # Metadata
    created_at: Optional[str] = Field(
        default=None,
        description="Timestamp when hyperparameters were created"
    )
    updated_at: Optional[str] = Field(
        default=None,
        description="Timestamp when hyperparameters were last updated"
    )

    if PYDANTIC_AVAILABLE:
        @validator('rebalance_frequency')
        def validate_rebalance_frequency(cls, v):
            """Validate rebalancing frequency."""
            valid_frequencies = ['hourly', 'daily', 'weekly', 'monthly']
            if v not in valid_frequencies:
                raise ValueError(f"rebalance_frequency must be one of: {valid_frequencies}")
            return v
        
        @validator('schema_version')
        def validate_schema_version(cls, v):
            """Validate schema version format."""
            if not v or len(v.split('.')) != 3:
                raise ValueError("schema_version must be in format 'X.Y.Z'")
            return v


def load_hyperparams(file_path: str = "hyperparams.json") -> HyperparametersSchema:
    """
    Load and validate hyperparameters from JSON file.
    
    Args:
        file_path: Path to hyperparams.json file
        
    Returns:
        Validated hyperparameters schema
    """
    global _last_missing_warning
    
    # Check if file exists
    if not os.path.exists(file_path):
        # Rate-limited warning (once per hour)
        current_time = datetime.now(timezone.utc).timestamp()
        if current_time - _last_missing_warning > _warning_interval:
            logger.warning(f"Hyperparams file not found: {file_path}. Using defaults.")
            _last_missing_warning = current_time
        
        # Return default hyperparameters
        return HyperparametersSchema()
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate schema version compatibility
        file_version = data.get('schema_version', '0.0.0')
        if not _is_compatible_version(file_version, HYPERPARAMS_SCHEMA_VERSION):
            logger.warning(
                f"Hyperparams schema version mismatch: file={file_version}, "
                f"expected={HYPERPARAMS_SCHEMA_VERSION}. Proceeding with caution."
            )
        
        # Load and validate
        hyperparams = HyperparametersSchema(**data)
        logger.info(f"Loaded hyperparams from {file_path} (version {file_version})")
        return hyperparams
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in hyperparams file {file_path}: {e}")
        return HyperparametersSchema()
    except Exception as e:
        logger.error(f"Error loading hyperparams from {file_path}: {e}")
        return HyperparametersSchema()


def save_hyperparams(
    hyperparams: HyperparametersSchema, 
    file_path: str = "hyperparams.json"
) -> bool:
    """
    Save hyperparameters to JSON file.
    
    Args:
        hyperparams: Hyperparameters to save
        file_path: Path to save file
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Update metadata
        current_time = datetime.now(timezone.utc).isoformat()
        if hyperparams.created_at is None:
            hyperparams.created_at = current_time
        hyperparams.updated_at = current_time
        
        # Convert to dict
        if PYDANTIC_AVAILABLE:
            data = hyperparams.dict()
        else:
            data = vars(hyperparams)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
        
        logger.info(f"Saved hyperparams to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving hyperparams to {file_path}: {e}")
        return False


def _is_compatible_version(file_version: str, current_version: str) -> bool:
    """
    Check if file version is compatible with current schema version.
    
    Args:
        file_version: Version from file
        current_version: Current schema version
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        file_parts = [int(x) for x in file_version.split('.')]
        current_parts = [int(x) for x in current_version.split('.')]
        
        # Major version must match
        if file_parts[0] != current_parts[0]:
            return False
        
        # Minor version can be older or same
        if file_parts[1] > current_parts[1]:
            return False
        
        return True
        
    except (ValueError, IndexError):
        return False


def get_default_hyperparams() -> HyperparametersSchema:
    """Get default hyperparameters configuration."""
    return HyperparametersSchema()


def validate_hyperparams_file(file_path: str = "hyperparams.json") -> Dict[str, Any]:
    """
    Validate hyperparams file and return validation report.
    
    Args:
        file_path: Path to hyperparams.json file
        
    Returns:
        Validation report dictionary
    """
    report = {
        'file_exists': False,
        'valid_json': False,
        'valid_schema': False,
        'schema_version': None,
        'version_compatible': False,
        'errors': [],
        'warnings': []
    }
    
    # Check file existence
    if os.path.exists(file_path):
        report['file_exists'] = True
        
        try:
            # Check JSON validity
            with open(file_path, 'r') as f:
                data = json.load(f)
            report['valid_json'] = True
            
            # Check schema version
            file_version = data.get('schema_version', '0.0.0')
            report['schema_version'] = file_version
            report['version_compatible'] = _is_compatible_version(file_version, HYPERPARAMS_SCHEMA_VERSION)
            
            # Validate against schema
            try:
                HyperparametersSchema(**data)
                report['valid_schema'] = True
            except Exception as e:
                report['errors'].append(f"Schema validation failed: {e}")
            
        except json.JSONDecodeError as e:
            report['errors'].append(f"Invalid JSON: {e}")
        except Exception as e:
            report['errors'].append(f"File read error: {e}")
    else:
        report['warnings'].append(f"File not found: {file_path}")
    
    return report


if __name__ == "__main__":
    # Test hyperparams schema
    print("Testing Hyperparameters Schema...")
    
    # Test default hyperparams
    default_params = get_default_hyperparams()
    print(f"Default schema version: {default_params.schema_version}")
    
    # Test loading (will use defaults if file doesn't exist)
    loaded_params = load_hyperparams()
    print(f"Loaded params - buy_threshold: {loaded_params.buy_threshold}")
    
    # Test validation
    validation_report = validate_hyperparams_file()
    print(f"Validation report: {validation_report}")
    
    print("Hyperparams schema tests completed!")