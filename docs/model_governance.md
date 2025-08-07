# Model Governance Documentation

## Overview

This document describes the model governance framework for managing the promotion of trading models from development through shadow testing to production deployment.

## Shadow-to-Production Workflow

### Model Lifecycle States

1. **Registered** - Newly trained model stored in registry
2. **Shadow** - Model running in shadow mode alongside production
3. **Production** - Model actively making trading decisions

### Governance Pipeline

```
Development → Registry → Shadow Testing → Production
     ↓           ↓            ↓              ↓
  Training   Metadata    Performance    Live Trading
   & Test    Storage     Validation      Decisions
```

## Dataset Hash Verification

### Purpose
Ensure model-data compatibility by tracking dataset fingerprints and preventing deployment with incompatible data.

### Implementation
- Computes hash from dataset paths, sizes, and modification times
- Includes content samples for small files (< 10MB)
- Stored in model metadata during registration
- Verified during model loading

### Usage Examples

```python
from ai_trading.model_registry import ModelRegistry
from ai_trading.governance.promotion import ModelPromotion

# Register model with dataset governance
registry = ModelRegistry()
model_id = registry.register_model(
    model=trained_model,
    strategy='momentum',
    model_type='lightgbm',
    metadata={'version': '1.2.3'},
    dataset_paths=[
        '/data/features/train.parquet',
        '/data/labels/targets.parquet'
    ]
)

# Load with dataset verification (default)
model, metadata = registry.load_model(
    model_id=model_id,
    verify_dataset_hash=True,
    current_dataset_paths=[
        '/data/features/live.parquet',
        '/data/labels/live_targets.parquet'  
    ]
)
```

### Hash Mismatch Handling
```python
# Allow dataset mismatch with environment variable
import os
os.environ['ALLOW_DATASET_MISMATCH'] = '1'

# Or catch and handle mismatch
try:
    model, metadata = registry.load_model(model_id, verify_dataset_hash=True)
except ValueError as e:
    if "Dataset hash mismatch" in str(e):
        # Handle mismatch appropriately
        print(f"Dataset changed: {e}")
        # Retrain model or accept risk
```

## Shadow Testing

### Purpose
Validate model performance in live market conditions without impacting production trading.

### Shadow Testing Process

1. **Initialization**
   ```python
   from ai_trading.governance.promotion import get_promotion_manager
   
   promotion = get_promotion_manager()
   
   # Start shadow testing
   success = promotion.start_shadow_testing(
       model_id='abc123def456',
       benchmark_model_id='prod_model_id'  # Optional
   )
   ```

2. **Metrics Collection**
   ```python
   # Update shadow metrics after each session
   session_stats = {
       'trade_count': 15,
       'turnover_ratio': 1.2,
       'sharpe_ratio': 0.8,
       'max_drawdown': 0.02,
       'drift_psi': 0.15,
       'avg_latency_ms': 45.0,
       'error_rate': 0.001
   }
   
   promotion.update_shadow_metrics(model_id, session_stats)
   ```

3. **Promotion Eligibility**
   ```python
   # Check if model meets promotion criteria
   eligible, details = promotion.check_promotion_eligibility(model_id)
   
   if eligible:
       print("Model ready for promotion!")
   else:
       print(f"Not eligible: {details['checks']}")
   ```

### Promotion Criteria

Default criteria for production promotion:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Shadow Sessions | ≥ 5 | Minimum trading sessions |
| Shadow Days | ≥ 3 | Minimum days in shadow |
| Trade Count | ≥ 10 | Minimum number of trades |
| Turnover Ratio | ≤ 1.5 | Max turnover vs benchmark |
| Live Sharpe | ≥ 0.5 | Minimum Sharpe ratio |
| Max Drawdown | ≤ 0.05 | Maximum drawdown (5%) |
| Drift PSI | ≤ 0.25 | Maximum population stability index |

### Custom Criteria Configuration

```python
from ai_trading.governance.promotion import PromotionCriteria

custom_criteria = PromotionCriteria(
    min_shadow_sessions=10,
    min_shadow_days=7,
    max_turnover_ratio=1.2,
    min_live_sharpe=0.6,
    max_drift_psi=0.20,
    max_drawdown_threshold=0.03
)

promotion = ModelPromotion(criteria=custom_criteria)
```

## Production Promotion

### Automatic Promotion
```python
# Promote model if criteria are met
success = promotion.promote_to_production(model_id)

if success:
    print(f"Model {model_id} promoted to production")
else:
    print("Promotion failed - check eligibility")
```

### Force Promotion
```python
# Force promotion bypassing criteria (emergency use)
success = promotion.promote_to_production(model_id, force=True)
```

### Active Model Management
```python
# Get current production model
current_prod = registry.get_production_model('momentum')
if current_prod:
    model_id, metadata = current_prod
    print(f"Current production model: {model_id}")

# Get active model path (symlink)
active_path = promotion.get_active_model_path('momentum')
print(f"Active model path: {active_path}")
```

## Model Registry Features

### Model Storage Structure
```
models/
├── strategy_name/
│   ├── 2024-12-07/
│   │   ├── model_hash_abc123/
│   │   │   ├── model.pkl
│   │   │   ├── meta.json
│   │   │   ├── feature_spec.json
│   │   │   └── metrics.json
│   │   └── model_hash_def456/
│   └── 2024-12-08/
└── registry_index.json
```

### Metadata Schema
```json
{
  "model_hash": "abc123def456",
  "strategy": "momentum",
  "model_type": "lightgbm",
  "registration_time": "2024-12-07T10:30:00Z",
  "dataset_hash": "a1b2c3d4",
  "dataset_paths": ["/data/train.parquet"],
  "governance": {
    "status": "shadow",
    "shadow_start_time": "2024-12-07T11:00:00Z",
    "shadow_sessions": 3,
    "promotion_eligible": false,
    "promotion_metrics": {
      "turnover_ratio": 1.1,
      "live_sharpe_ratio": 0.7
    }
  }
}
```

### Registry Operations
```python
# List models by criteria
models = registry.list_models(
    strategy='momentum',
    model_type='lightgbm',
    active_only=True
)

# Get performance summary
summary = registry.get_model_performance_summary()
print(f"Total models: {summary['total_models']}")
print(f"Active models: {summary['active_models']}")

# Deactivate old model
registry.deactivate_model('old_model_id')
```

## Shadow Model Monitoring

### List Shadow Models
```python
# Get all models in shadow testing
shadow_models = promotion.list_shadow_models()

for model_info in shadow_models:
    print(f"Model: {model_info['model_id']}")
    print(f"Strategy: {model_info['strategy']}")
    print(f"Eligible: {model_info['promotion_eligible']}")
    print(f"Sessions: {model_info['metrics'].sessions_completed}")
```

### Shadow Metrics Tracking
- **Sessions Completed** - Number of trading sessions
- **Total Trades** - Cumulative trade count
- **Turnover Ratio** - Rolling average vs benchmark
- **Live Sharpe Ratio** - Risk-adjusted performance
- **Max Drawdown** - Maximum peak-to-trough decline
- **Drift PSI** - Population Stability Index for drift detection
- **Latency** - Average execution latency
- **Error Rate** - System error percentage

## Active Model Symlinks

### Purpose
Provide stable filesystem paths to production models independent of model IDs.

### Structure
```
artifacts/governance/active/
├── momentum_active -> /models/momentum/2024-12-07/abc123def456/
├── mean_reversion_active -> /models/mean_reversion/2024-12-06/xyz789/
└── ml_strategy_active -> /models/ml_strategy/2024-12-08/def456ghi/
```

### Usage in Production
```python
# Live trading systems can use stable paths
active_path = promotion.get_active_model_path('momentum')
if active_path:
    # Load model from active symlink
    with open(f"{active_path}/model.pkl", 'rb') as f:
        model = pickle.load(f)
```

## Best Practices

### Model Development
1. **Tag models appropriately** for easy filtering and management
2. **Include comprehensive metadata** about training process and data
3. **Document feature specifications** for reproducibility
4. **Store performance metrics** from validation testing

### Shadow Testing
1. **Run sufficient shadow sessions** before promotion consideration
2. **Monitor all key metrics** not just profitability
3. **Compare against benchmark models** when available
4. **Check for data drift** using PSI or similar metrics

### Production Deployment
1. **Use gradual rollouts** when possible
2. **Monitor production performance closely** after promotion
3. **Keep fallback models ready** for quick reversion
4. **Document promotion decisions** and rationale

### Data Governance
1. **Version control datasets** used for training
2. **Track data lineage** from source to features
3. **Validate data quality** before model training
4. **Monitor for distribution drift** in production

## Troubleshooting

### Common Issues

1. **Dataset Hash Mismatch**
   - Check if training data has changed
   - Verify file paths are correct
   - Consider if retrain is needed

2. **Shadow Model Not Promoting**
   - Review promotion criteria
   - Check shadow metrics collection
   - Verify sufficient testing time

3. **Model Loading Failures**
   - Check file permissions
   - Verify model registry integrity
   - Validate pickle compatibility

### Debug Commands
```python
# Check model governance status
model_data = registry.load_model(model_id, verify_dataset_hash=False)[1]
governance = model_data.get('governance', {})
print(f"Status: {governance.get('status')}")

# Get promotion eligibility details
eligible, details = promotion.check_promotion_eligibility(model_id)
print(f"Eligibility details: {details}")

# Review shadow metrics
shadow_models = promotion.list_shadow_models()
for model in shadow_models:
    if model['model_id'] == model_id:
        print(f"Shadow metrics: {model['metrics']}")
```