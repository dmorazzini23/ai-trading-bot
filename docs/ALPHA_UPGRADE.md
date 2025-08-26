# Alpha Quality Overhaul - Migration Guide

This document outlines the major changes introduced in the alpha quality overhaul and provides migration guidance for existing trading strategies.

## Overview

The alpha quality overhaul introduces significant improvements to model training, backtesting realism, signal processing, and RL reward shaping. These changes ensure leak-proof training, realistic execution modeling, and enhanced signal ensemble capabilities.

## Key Changes

### 1. Data Labeling & Splits

#### New Modules
- `ai_trading/data/labels.py` - Explicit labeling functions
- `ai_trading/data/splits.py` - Purged time series cross-validation

#### Usage Examples

```python
from ai_trading.data.labels import fixed_horizon_return, triple_barrier_labels
from ai_trading.data.splits import PurgedGroupTimeSeriesSplit, walkforward_splits

# Label generation
returns = fixed_horizon_return(prices, horizon_bars=5, fee_bps=5.0)
barrier_labels = triple_barrier_labels(prices, pt_sl=(0.02, -0.02))

# Leak-proof cross-validation
cv = PurgedGroupTimeSeriesSplit(n_splits=5, embargo_pct=0.01)
for train_idx, test_idx in cv.split(X, y, t1=barrier_labels['t1']):
    # Train with leak-proof splits
    pass
```

### 2. Feature Engineering Pipeline

#### New Module
- `ai_trading/features/pipeline.py` - Leak-proof feature pipeline

#### Migration
Replace ad-hoc feature engineering with standardized pipeline:

```python
# OLD - Manual feature engineering
X_scaled = StandardScaler().fit_transform(X)

# NEW - Leak-proof pipeline
from ai_trading.features.pipeline import create_feature_pipeline

pipeline = create_feature_pipeline(
    scaler_type="standard",
    build_features_params={
        'include_regime': True,
        'include_volatility': True
    }
)

# Fit only on training data
X_train_processed = pipeline.fit_transform(X_train, y_train)
X_test_processed = pipeline.transform(X_test)  # No fit on test data
```

### 3. Model Training with Purged CV

#### New Module
- `ai_trading/training/train_ml.py` - Enhanced model training

#### Usage
```python
from ai_trading.training.train_ml import MLTrainer

trainer = MLTrainer(
    model_type="lightgbm",
    cv_splits=5,
    embargo_pct=0.01,
    purge_pct=0.02
)

results = trainer.train(
    X=features,
    y=labels,
    optimize_hyperparams=True,
    optimization_trials=100
)

# Save with metadata
trainer.save_model("models/strategy_20240101", metadata={
    "strategy": "momentum",
    "universe": ["AAPL", "MSFT"],
    "train_period": "2020-2024"
})
```

### 4. Walk-Forward Evaluation

#### New Module
- `ai_trading/evaluation/walkforward.py` - Comprehensive walk-forward analysis

#### Usage
```python
from ai_trading.evaluation.walkforward import WalkForwardEvaluator

evaluator = WalkForwardEvaluator(
    mode="rolling",  # or "anchored"
    train_span=252,  # 1 year
    test_span=21,    # 1 month
    embargo_pct=0.01
)

results = evaluator.run_walkforward(
    data=full_dataset,
    target_col='returns',
    model_type='lightgbm',
    save_results=True
)

print(f"Net Sharpe: {results['aggregate_metrics']['net_sharpe']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### 5. Enhanced Backtesting

#### Updated Module
- `ai_trading/strategies/backtest.py` - Now includes realistic execution

#### Migration
Update backtest initialization:

```python
# OLD
backtest_engine = BacktestEngine()

# NEW - With realistic execution
from ai_trading.strategies.backtest import BacktestEngine

backtest_engine = BacktestEngine(
    initial_capital=100000,
    commission_bps=5.0,      # 5 bps commission
    commission_flat=1.0,     # $1 flat fee
    latency_ms=50.0,         # 50ms latency
    enable_slippage=True,    # Model slippage
    enable_partial_fills=False,  # Partial fills (optional)
    slippage_model="impact"  # Advanced slippage model
)

# Results now include net metrics
results = backtest_engine.run_backtest(strategy, data, start_date, end_date)
print(f"Gross Return: {results['gross_return']:.2%}")
print(f"Net Return: {results['net_return']:.2%}")  # After all costs
print(f"Average Slippage: {results['avg_slippage_bps']:.1f} bps")
```

### 6. Signal Ensemble Upgrades

#### Enhanced Module
- `ai_trading/strategies/signals.py` - Meta-learning and stacking

#### Migration
Update signal aggregation:

```python
# OLD
aggregator = SignalAggregator()
signal = aggregator.aggregate_signals(signals, method="weighted_average")

# NEW - With meta-learning
from ai_trading.strategies.signals import SignalAggregator
from datetime import UTC, datetime

aggregator = SignalAggregator(
    enable_stacking=True,
    decay_window=30,
    turnover_penalty=0.1,
    conflict_resolution="majority"
)

signal = aggregator.aggregate_signals(
    signals,
    method="stacking",  # Meta-learning approach
    market_data=current_market_data,
    timestamp=datetime.now(UTC)
)

# Update performance for meta-learning
aggregator.update_signal_performance("signal_id", actual_return=0.02)
```

### 7. Portfolio Sizing

#### New Module
- `ai_trading/portfolio/sizing.py` - Advanced position sizing

#### Usage
```python
from ai_trading.portfolio.sizing import VolatilityTargetingSizer, RiskParitySizer

# Volatility targeting
vol_sizer = VolatilityTargetingSizer(target_vol=0.15)
positions = vol_sizer.calculate_position_sizes(
    signals={'AAPL': 0.8, 'MSFT': 0.6},
    current_prices={'AAPL': 150, 'MSFT': 300},
    portfolio_value=100000,
    price_history=price_data
)

# Risk parity
rp_sizer = RiskParitySizer()
weights = rp_sizer.calculate_risk_parity_weights(signals, price_history)
```

### 8. Enhanced RL Environment

#### Updated Modules
- `ai_trading/rl_trading/env.py` - Enhanced reward function
- `ai_trading/rl_trading/train.py` - Advanced training with callbacks

#### Migration
```python
# OLD
env = TradingEnv(data)
model = PPO("MlpPolicy", env)

# NEW - Enhanced environment and training
from ai_trading.rl_trading.env import TradingEnv
from ai_trading.rl_trading.train import RLTrainer

trainer = RLTrainer(
    algorithm="PPO",
    total_timesteps=100000,
    eval_freq=10000,
    early_stopping_patience=10
)

results = trainer.train(
    data=training_data,
    env_params={
        'transaction_cost': 0.001,
        'slippage': 0.0005,
        'half_spread': 0.0002
    },
    save_path='models/rl_strategy'
)

# New reward includes turnover/drawdown/variance penalties
print(f"Final Reward: {results['final_evaluation']['mean_reward']:.4f}")
print(f"Avg Turnover Penalty: {results['final_evaluation']['avg_turnover_penalty']:.4f}")
```

### 9. Model Registry

#### New Module
- `ai_trading/model_registry.py` - Centralized model management

#### Usage
```python
from ai_trading.model_registry import ModelRegistry, register_model

# Register a model
model_id = register_model(
    model=trained_model,
    strategy="momentum",
    model_type="lightgbm",
    metadata={
        "training_date": "2024-01-01",
        "cv_score": 0.85,
        "feature_hash": "abc123"
    },
    feature_spec=feature_specification,
    metrics=performance_metrics,
    tags=["production", "v2"]
)

# Load latest model
registry = ModelRegistry()
model, metadata, model_id = registry.load_latest_by_strategy(
    "momentum", 
    tags=["production"]
)
```

## CLI Usage

### Training Models
```bash
# Train ML model with walk-forward smoke test
python -m ai_trading.training.train_ml --symbol-list AAPL MSFT --wf-smoke

# Walk-forward evaluation
python -m ai_trading.evaluation.walkforward --smoke
```

### Validation Commands
```bash
# Run alpha quality tests
python tests/test_alpha_quality.py

# Validate backtest realism
python -c "
from ai_trading.strategies.backtest import BacktestEngine
# Verify net metrics < gross metrics due to costs
"
```

## Configuration Changes

### New Parameters

Add to your strategy configuration:

```json
{
  "training": {
    "purged_cv": {
      "n_splits": 5,
      "embargo_pct": 0.01,
      "purge_pct": 0.02
    },
    "walk_forward": {
      "mode": "rolling",
      "train_span_days": 252,
      "test_span_days": 21
    }
  },
  "execution": {
    "commission_bps": 5.0,
    "commission_flat": 1.0,
    "latency_ms": 50.0,
    "enable_slippage": true,
    "slippage_model": "impact"
  },
  "signals": {
    "enable_stacking": true,
    "decay_window": 30,
    "turnover_penalty": 0.1
  },
  "portfolio": {
    "target_volatility": 0.15,
    "max_position_weight": 0.25,
    "correlation_threshold": 0.7
  }
}
```

## Breaking Changes

### 1. Signal Aggregation
- `SignalAggregator.aggregate_signals()` now requires additional parameters for meta-learning
- Return format includes additional metadata

### 2. Backtest Results
- Results dictionary now includes separate `gross_return` and `net_return`
- New fields: `avg_slippage_bps`, `avg_fill_ratio`, `cost_drag`

### 3. RL Environment
- `TradingEnv` constructor requires additional cost parameters
- Step function returns enhanced info dictionary with penalty details

## Performance Impact

### Expected Improvements
- **Reduced overfitting**: Purged CV and embargo prevent data leakage
- **More realistic returns**: Net metrics after transaction costs and slippage
- **Better signal quality**: Meta-learning improves signal combination
- **Lower turnover**: Turnover penalties reduce excessive trading

### Computational Overhead
- **Training time**: +20-30% due to purged CV and hyperparameter optimization
- **Memory usage**: +15% for extended tracking and metadata
- **Evaluation time**: +50% for walk-forward analysis (one-time cost)

## Migration Checklist

- [ ] Update data splits to use `PurgedGroupTimeSeriesSplit`
- [ ] Replace manual feature engineering with `create_feature_pipeline()`
- [ ] Enhance model training with `MLTrainer` class
- [ ] Add walk-forward evaluation to model validation
- [ ] Update backtest engine with realistic execution parameters
- [ ] Upgrade signal aggregation to use stacking meta-learner
- [ ] Implement portfolio sizing with volatility targeting
- [ ] Enhance RL training with new reward shaping
- [ ] Set up model registry for centralized model management
- [ ] Run alpha quality tests to validate implementation
- [ ] Update configuration files with new parameters
- [ ] Document strategy-specific migration notes

## Support

For questions or issues with the migration:

1. Review the test files in `tests/test_alpha_quality.py` for examples
2. Check module docstrings for detailed API documentation
3. Run smoke tests to validate installation: `python -m ai_trading.evaluation.walkforward --smoke`

## Next Steps

After migration:

1. **Retrain Models**: Use new leak-proof training pipeline
2. **Validate Performance**: Run walk-forward analysis on historical data
3. **Update Monitoring**: Include new net metrics in performance tracking
4. **Gradual Rollout**: Deploy enhanced strategies incrementally
5. **Performance Review**: Compare old vs new strategy performance after 30 days