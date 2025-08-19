# Risk Management Parameter Audit Report
==================================================

## Summary
- ✅ Correct values found: 12
- ❌ Incorrect values found: 6
- 🔍 Hardcoded old values: 131
- ⚙️ Configuration files: 2
- 🚨 **Total issues requiring attention: 137**

## Expected Values (PR #864)
- CAPITAL_CAP: 0.25 (25.0%)
- MAX_POSITION_SIZE: 8,000 shares
- DOLLAR_RISK_LIMIT: 0.05 (5.0%)
- MAX_DRAWDOWN_THRESHOLD: 0.15 (15.0%)

## ✅ Correct Parameter Values
- bot_engine.py:1616 - CAPITAL_CAP = 0.25
- bot_engine.py:1626 - CAPITAL_CAP = 0.25
- bot_engine.py:1636 - CAPITAL_CAP = 0.25
- bot_engine.py:1763 - CAPITAL_CAP = 0.25
- bot_engine.py:1775 - CAPITAL_CAP = 0.25
- bot_engine.py:7584 - CAPITAL_CAP = 0.25
- bot_engine.py:1739 - MAX_POSITION_SIZE = 8000.0
- bot_engine.py:1785 - MAX_POSITION_SIZE = 8000.0
- bot_engine.py:1780 - DOLLAR_RISK_LIMIT = 0.05
- ai_trading/config/management.py:446 - DOLLAR_RISK_LIMIT = 0.05
- hyperparams.json:N/A - CAPITAL_CAP = 0.25
- backup/test_backup/hyperparams.json:N/A - CAPITAL_CAP = 0.25

## ❌ Incorrect Parameter Values (REQUIRE FIXING)
- demonstrate_optimization.py:63 - MAX_POSITION_SIZE
  Found: 10.0 | Expected: 8000
  Line: print(f"  • MAX_POSITION_SIZE: 10.0% → {RISK_PARAMETERS['MAX_POSITION_SIZE']*100:.1f}% (+150% increase)")
- demo_drawdown_protection.py:20 - MAX_DRAWDOWN_THRESHOLD
  Found: 0.1 | Expected: 0.15
  Line: print(f"Configuration: Max Drawdown = {config.MAX_DRAWDOWN_THRESHOLD:.1%}")
- demo_drawdown_protection.py:63 - MAX_DRAWDOWN_THRESHOLD
  Found: 0.1 | Expected: 0.15
  Line: print(f"      💥 CIRCUIT BREAKER TRIGGERED: {status['current_drawdown']:.1%} > {config.MAX_DRAWDOWN_THRESHOLD:.1%}")
- demonstrate_optimization_simple.py:43 - MAX_POSITION_SIZE
  Found: 10.0 | Expected: 8000
  Line: print(f"  • MAX_POSITION_SIZE: 10.0% → 25.0% (+150% increase)")
- ai_trading/core/parameter_validator.py:315 - MAX_POSITION_SIZE
  Found: 0.1 | Expected: 8000
  Line: logger.info(f"  MAX_POSITION_SIZE: 0.10 → {RISK_PARAMETERS['MAX_POSITION_SIZE']} (increased for larger positions)")
- ai_trading/core/constants.py:26 - MAX_POSITION_SIZE
  Found: 0.25 | Expected: 8000
  Line: "MAX_POSITION_SIZE": 0.25,              # 25% max position size

## 🔍 Hardcoded Old Values (REQUIRE REVIEW)
- security_manager.py:267 - 1000 (old MAX_POSITION_SIZE)
  Line: self.pnl_history: deque = deque(maxlen=1000)
- security_manager.py:268 - 1000 (old MAX_POSITION_SIZE)
  Line: self.position_history: deque = deque(maxlen=1000)
- security_manager.py:398 - 1000 (old MAX_POSITION_SIZE)
  Line: self.security_events: deque = deque(maxlen=1000)
- production_validator.py:89 - 1000 (old MAX_POSITION_SIZE)
  Line: 'stress_test': {'concurrent_users': 200, 'duration': 300, 'rps_target': 1000},
- production_validator.py:134 - 1000 (old MAX_POSITION_SIZE)
  Line: response_time = (time.perf_counter() - submit_time) * 1000  # ms
- production_validator.py:147 - 1000 (old MAX_POSITION_SIZE)
  Line: response_time = (time.perf_counter() - submit_time) * 1000
- production_validator.py:487 - 1000 (old MAX_POSITION_SIZE)
  Line: sum(i * i for i in range(1000))
- predict.py:62 - 1000 (old MAX_POSITION_SIZE)
  Line: _sentiment_cache = TTLCache(maxsize=1000, ttl=300)
- predict.py:145 - 1000 (old MAX_POSITION_SIZE)
  Line: if len(_sentiment_cache) >= 1000:
- meta_learning.py:401 - 1000 (old MAX_POSITION_SIZE)
  Line: extreme_moves = price_change_pct > 10.0  # 1000% change
- meta_learning.py:405 - 1000 (old MAX_POSITION_SIZE)
  Line: logger.warning(f"META_LEARNING_EXTREME_MOVES: Found {extreme_count} trades with >1000% price moves")
- logger.py:307 - 1000 (old MAX_POSITION_SIZE)
  Line: ...         'requested': 1000,
- logger.py:405 - 1000 (old MAX_POSITION_SIZE)
  Line: elif isinstance(value, str) and len(value) > 1000:
- logger.py:406 - 1000 (old MAX_POSITION_SIZE)
  Line: sanitized[key] = value[:1000] + "...[TRUNCATED]"
- bot_engine.py:2928 - 1000 (old MAX_POSITION_SIZE)
  Line: max_order_size = int(os.getenv("MAX_ORDER_SIZE", "1000"))
- bot_engine.py:4898 - 0.02 (old 2% value)
  Line: cash: float, price: float, returns: np.ndarray, target_vol: float = 0.02
- bot_engine.py:5522 - 0.02 (old 2% value)
  Line: vol_sz = vol_target_position_size(cash, price, rets, target_vol=0.02)
- bot_engine.py:5532 - 1000 (old MAX_POSITION_SIZE)
  Line: return max(1, int(1000 / price)) if price > 0 else 1
- bot_engine.py:5992 - 1000 (old MAX_POSITION_SIZE)
  Line: if balance > 1000 and target_weight > 0.001 and current_price > 0:
- bot_engine.py:5993 - 1000 (old MAX_POSITION_SIZE)
  Line: raw_qty = max(1, int(1000 / current_price))  # Minimum $1000 position
- bot_engine.py:6517 - 0.02 (old 2% value)
  Line: if len(state.rolling_losses) >= 20 and sum(state.rolling_losses[-20:]) > 0.02:
- bot_engine.py:7174 - 1000 (old MAX_POSITION_SIZE)
  Line: limit=1000,
- bot_engine.py:7186 - 1000 (old MAX_POSITION_SIZE)
  Line: 'volume': [1000] * 100,
- bot_engine.py:7197 - 1000 (old MAX_POSITION_SIZE)
  Line: 'volume': [1000] * 100,
- bot_engine.py:7212 - 1000 (old MAX_POSITION_SIZE)
  Line: 'volume': [1000] * 100,
- bot_engine.py:7223 - 1000 (old MAX_POSITION_SIZE)
  Line: 'volume': [1000] * 100,
- bot_engine.py:7578 - 0.02 (old 2% value)
  Line: if avg_r < -0.02:
- bot_engine.py:7584 - 0.02 (old 2% value)
  Line: max(0.02, min(0.1, params.get("CAPITAL_CAP", 0.25) * (1 - dd))), 3
- bot_engine.py:8230 - 0.02 (old 2% value)
  Line: if side == "long" and price > vwap and pnl > 0.02:
- config.py:370 - 0.02 (old 2% value)
  Line: LIQUIDITY_VOL_THRESHOLD = float(os.getenv("LIQUIDITY_VOL_THRESHOLD", "0.02"))
- config.py:545 - 0.02 (old 2% value)
  Line: delta_threshold: float = 0.02
- config.py:569 - 0.02 (old 2% value)
  Line: delta_threshold=float(os.getenv("DELTA_THRESHOLD", "0.02")),
- production_monitoring.py:138 - 1000 (old MAX_POSITION_SIZE)
  Line: self.latency_tracker: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
- production_monitoring.py:267 - 1000 (old MAX_POSITION_SIZE)
  Line: elapsed = (time.perf_counter() - start_time) * 1000  # Convert to ms
- process_manager.py:347 - 1000 (old MAX_POSITION_SIZE)
  Line: if total_memory > 1000:
- profile_indicators.py:28 - 1000 (old MAX_POSITION_SIZE)
  Line: 'volume': np.random.randint(1000, 10000, size=100_000)
- demo_centralized_imports.py:62 - 1000 (old MAX_POSITION_SIZE)
  Line: 'volume': [1000, 1200, 800, 1500, 900, 1800, 1100, 2000]
- signals.py:446 - 1000 (old MAX_POSITION_SIZE)
  Line: window_size: int = 1000,
- strategy_allocator.py:16 - 0.02 (old 2% value)
  Line: delta_threshold: float = 0.02
- algorithm_optimizer.py:108 - 1000 (old MAX_POSITION_SIZE)
  Line: self.parameter_history: deque = deque(maxlen=1000)
- algorithm_optimizer.py:113 - 1000 (old MAX_POSITION_SIZE)
  Line: regime: deque(maxlen=1000) for regime in MarketRegime  # Bounded growth
- algorithm_optimizer.py:165 - 0.02 (old 2% value)
  Line: volatility=0.02,
- algorithm_optimizer.py:234 - 0.02 (old 2% value)
  Line: volatility=0.02,
- algorithm_optimizer.py:436 - 0.02 (old 2% value)
  Line: if avg_return < -0.02:  # Losing streak
- algorithm_optimizer.py:468 - 0.02 (old 2% value)
  Line: base_position_pct = 0.02  # 2% base position
- algorithm_optimizer.py:529 - 0.02 (old 2% value)
  Line: return 0.02  # Default conservative fraction
- algorithm_optimizer.py:538 - 0.02 (old 2% value)
  Line: return 0.02
- algorithm_optimizer.py:564 - 0.02 (old 2% value)
  Line: return 0.02  # Conservative default
- algorithm_optimizer.py:596 - 0.02 (old 2% value)
  Line: base_stop_pct = 0.02  # 2% base stop
- algorithm_optimizer.py:676 - 0.02 (old 2% value)
  Line: stop = self.calculate_stop_loss(100.0, 'BUY', 0.02, 2.0)
- algorithm_optimizer.py:699 - 0.02 (old 2% value)
  Line: optimized = self.optimize_parameters(conditions, [0.01, -0.02, 0.03], True)
- memory_optimizer.py:119 - 1000 (old MAX_POSITION_SIZE)
  Line: 'collection_time_ms': gc_time * 1000,
- memory_optimizer.py:339 - 1000 (old MAX_POSITION_SIZE)
  Line: 'execution_time_ms': execution_time * 1000,
- scalability_manager.py:81 - 1000 (old MAX_POSITION_SIZE)
  Line: self.workload_history: deque = deque(maxlen=1000)
- verify_critical_fixes.py:32 - 1000 (old MAX_POSITION_SIZE)
  Line: assert 'balance > 1000 and target_weight > 0.001' in content, "Minimum position logic not found"
- verify_critical_fixes.py:33 - 1000 (old MAX_POSITION_SIZE)
  Line: assert 'max(1, int(1000 / current_price))' in content, "Minimum $1000 position logic not found"
- system_diagnostic.py:131 - 1000 (old MAX_POSITION_SIZE)
  Line: 'collection_time_ms': gc_time * 1000
- system_diagnostic.py:317 - 1000 (old MAX_POSITION_SIZE)
  Line: diagnostic_results[f'{check_name}_time_ms'] = check_time * 1000
- ai_trading.tools.env_validate:84 - 1000 (old MAX_POSITION_SIZE)
  Line: VOLUME_THRESHOLD: int = Field(default=50000, ge=1000, description="Minimum daily volume requirement")
- performance_monitor.py:27 - 1000 (old MAX_POSITION_SIZE)
  Line: self.metrics_history = deque(maxlen=1000)  # Store last 1000 measurements
- performance_monitor.py:93 - 1000 (old MAX_POSITION_SIZE)
  Line: metrics['collection_time_ms'] = (time.time() - start_time) * 1000
- performance_monitor.py:582 - 1000 (old MAX_POSITION_SIZE)
  Line: self.trade_metrics = deque(maxlen=1000)
- health_check.py:514 - 1000 (old MAX_POSITION_SIZE)
  Line: latency_ms = (time.perf_counter() - start_time) * 1000
- production_integration.py:278 - 1000 (old MAX_POSITION_SIZE)
  Line: execution_time = (time.perf_counter() - start_time) * 1000
- production_integration.py:287 - 1000 (old MAX_POSITION_SIZE)
  Line: execution_time = (time.perf_counter() - start_time) * 1000
- performance_optimizer.py:73 - 1000 (old MAX_POSITION_SIZE)
  Line: self.execution_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
- performance_optimizer.py:74 - 1000 (old MAX_POSITION_SIZE)
  Line: self.memory_snapshots: deque = deque(maxlen=1000)
- performance_optimizer.py:80 - 1000 (old MAX_POSITION_SIZE)
  Line: self.cache_max_size: int = 1000
- performance_optimizer.py:140 - 1000 (old MAX_POSITION_SIZE)
  Line: execution_time_ms = execution_time * 1000
- performance_optimizer.py:225 - 1000 (old MAX_POSITION_SIZE)
  Line: execution_time_ms = execution_time * 1000
- performance_optimizer.py:305 - 1000 (old MAX_POSITION_SIZE)
  Line: self.execution_times[func_name] = deque(recent_times, maxlen=1000)
- monitoring_dashboard.py:504 - 1000 (old MAX_POSITION_SIZE)
  Line: if latest_risk['var_95'] > 1000:  # $1000 daily VaR
- monitoring_dashboard.py:508 - 0.02 (old 2% value)
  Line: if latest_risk['volatility'] > 0.02:  # 2% daily volatility
- risk_engine.py:466 - 0.02 (old 2% value)
  Line: scale *= max(0.5, min(1.0, 0.02 / volatility))
- risk_engine.py:1036 - 0.08 (old 8% value)
  Line: ...     'current_drawdown': 0.08,  # 8% drawdown
- trade_execution.py:74 - 1000 (old MAX_POSITION_SIZE)
  Line: _RECENT_BUYS_MAX_SIZE = 1000  # Prevent memory growth
- trade_execution.py:737 - 0.02 (old 2% value)
  Line: elif aggressive and spread < 0.02:
- trade_execution.py:1244 - 1000 (old MAX_POSITION_SIZE)
  Line: latency *= 1000.0
- trade_execution.py:1402 - 1000 (old MAX_POSITION_SIZE)
  Line: latency *= 1000.0
- rebalancer.py:296 - 1000 (old MAX_POSITION_SIZE)
  Line: base_score = min(1000, abs(tax_benefit))  # Cap at $1000 benefit
- ai_trading/data_validation.py:50 - 1000 (old MAX_POSITION_SIZE)
  Line: self.min_volume_threshold = 1000  # Minimum volume
- ai_trading/data_validation.py:288 - 1000 (old MAX_POSITION_SIZE)
  Line: volume_spikes = (volume_changes > 10.0).sum()  # 1000% volume spikes
- ai_trading/capital_scaling.py:137 - 0.02 (old 2% value)
  Line: base_allocation = equity * 0.02
- ai_trading/health_monitor.py:117 - 1000 (old MAX_POSITION_SIZE)
  Line: response_time = (time.time() - start_time) * 1000
- ai_trading/health_monitor.py:136 - 1000 (old MAX_POSITION_SIZE)
  Line: response_time = self.timeout_seconds * 1000
- ai_trading/health_monitor.py:143 - 1000 (old MAX_POSITION_SIZE)
  Line: response_time = (time.time() - start_time) * 1000
- ai_trading/trade_logic.py:45 - 0.02 (old 2% value)
  Line: max_risk = risk_params.get("max_risk", 0.02)
- strategies/mean_reversion.py:60 - 0.02 (old 2% value)
  Line: weight = min(0.05, max(0.01, abs(z) * 0.02))  # 1-5% allocation based on z-score
- strategies/mean_reversion.py:73 - 0.02 (old 2% value)
  Line: weight = min(0.05, max(0.01, abs(z) * 0.02))  # 1-5% allocation based on z-score
- ai_trading/safety/monitoring.py:63 - 0.02 (old 2% value)
  Line: "max_position_risk": 0.02,    # 2% max per position
- ai_trading/safety/monitoring.py:65 - 1000 (old MAX_POSITION_SIZE)
  Line: "min_available_cash": 1000,   # $1000 minimum cash
- ai_trading/safety/monitoring.py:438 - 1000 (old MAX_POSITION_SIZE)
  Line: if len(self.metrics["order_latency"]) > 1000:
- ai_trading/rl_trading/train.py:16 - 1000 (old MAX_POSITION_SIZE)
  Line: def train(data: np.ndarray, model_path: str | Path, timesteps: int = 1000) -> str:
- ai_trading/core/parameter_validator.py:314 - 0.02 (old 2% value)
  Line: logger.info(f"  MAX_PORTFOLIO_RISK: 0.02 → {RISK_PARAMETERS['MAX_PORTFOLIO_RISK']} (higher profit potential)")
- ai_trading/core/enums.py:67 - 0.02 (old 2% value)
  Line: RiskLevel.CONSERVATIVE: 0.02,  # 2%
- ai_trading/core/constants.py:81 - 1000 (old MAX_POSITION_SIZE)
  Line: "MAX_DAILY_TRADES": 1000,           # Maximum trades per day
- ai_trading/execution/production_engine.py:194 - 1000 (old MAX_POSITION_SIZE)
  Line: if self.max_slippage_bps < 0 or self.max_slippage_bps > 1000:
- ai_trading/execution/production_engine.py:195 - 1000 (old MAX_POSITION_SIZE)
  Line: self._validation_errors.append("Max slippage must be between 0 and 1000 basis points")
- ai_trading/execution/production_engine.py:293 - 1000 (old MAX_POSITION_SIZE)
  Line: kwargs['client_order_id'] = f"req_{int(time.time() * 1000)}"  # Use milliseconds for uniqueness
- ai_trading/execution/production_engine.py:467 - 1000 (old MAX_POSITION_SIZE)
  Line: execution_time_ms = (time.time() - start_time) * 1000
- ai_trading/execution/microstructure.py:160 - 1000 (old MAX_POSITION_SIZE)
  Line: elif spread_bps > 50 or market_depth < 1000:
- ai_trading/execution/microstructure.py:294 - 1000 (old MAX_POSITION_SIZE)
  Line: features["large_trade_ratio"] = sum(1 for size in trade_sizes if size > 1000) / len(trade_sizes)
- ai_trading/execution/microstructure.py:452 - 1000 (old MAX_POSITION_SIZE)
  Line: return min(1.0, information_content * 1000)  # Scaling factor
- ai_trading/execution/liquidity.py:475 - 0.02 (old 2% value)
  Line: "max_participation_rate": 0.02,
- ai_trading/execution/liquidity.py:635 - 0.02 (old 2% value)
  Line: participation_rate = 0.02
- ai_trading/execution/simulator.py:224 - 1000 (old MAX_POSITION_SIZE)
  Line: size_factor = min(1.0, quantity / 1000)  # Normalize to 1000 shares
- ai_trading/config/management.py:88 - 1000 (old MAX_POSITION_SIZE)
  Line: "min_value": 1000,
- ai_trading/strategies/regime_detection.py:632 - 0.02 (old 2% value)
  Line: elif abs(returns_1m) > 0.02:
- ai_trading/strategies/regime_detection.py:648 - 0.08 (old 8% value)
  Line: elif abs(returns_6m) > 0.08:
- ai_trading/strategies/regime_detection.py:717 - 0.02 (old 2% value)
  Line: elif abs(macd) > 0.02:
- ai_trading/strategies/regime_detection.py:723 - 0.02 (old 2% value)
  Line: elif abs(roc) > 0.02:
- ai_trading/strategies/metalearning.py:790 - 0.02 (old 2% value)
  Line: target_pct = max(0.02, volatility * 2)
- ai_trading/strategies/metalearning.py:799 - 0.02 (old 2% value)
  Line: target_pct = max(0.02, volatility * 2)
- ai_trading/monitoring/alerting.py:52 - 1000 (old MAX_POSITION_SIZE)
  Line: self.id = f"alert_{int(time.time() * 1000)}"
- ai_trading/monitoring/alerting.py:303 - 1000 (old MAX_POSITION_SIZE)
  Line: self.max_history_size = 1000
- ai_trading/monitoring/performance_dashboard.py:43 - 1000 (old MAX_POSITION_SIZE)
  Line: self.trades = deque(maxlen=1000)  # Keep last 1000 trades
- ai_trading/monitoring/performance_dashboard.py:50 - 0.02 (old 2% value)
  Line: self.risk_free_rate = 0.02  # 2% risk-free rate
- ai_trading/monitoring/performance_dashboard.py:266 - 1000 (old MAX_POSITION_SIZE)
  Line: self.pnl_history = deque(maxlen=1000)
- ai_trading/monitoring/alerts.py:54 - 1000 (old MAX_POSITION_SIZE)
  Line: self.id = f"alert_{int(time.time() * 1000)}"
- ai_trading/monitoring/alerts.py:122 - 1000 (old MAX_POSITION_SIZE)
  Line: self.max_alerts = 1000
- ai_trading/monitoring/dashboard.py:367 - 1000 (old MAX_POSITION_SIZE)
  Line: if cpu_usage > 80 or memory_usage > 80 or latency > 1000:
- ai_trading/monitoring/metrics.py:309 - 1000 (old MAX_POSITION_SIZE)
  Line: if len(histogram) > 1000:
- ai_trading/monitoring/metrics.py:310 - 1000 (old MAX_POSITION_SIZE)
  Line: histogram[:] = histogram[-1000:]
- ai_trading/monitoring/metrics.py:342 - 0.02 (old 2% value)
  Line: def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
- ai_trading/risk/position_sizing.py:257 - 0.02 (old 2% value)
  Line: atr_value = entry_price * 0.02  # 2% fallback
- ai_trading/risk/position_sizing.py:539 - 0.02 (old 2% value)
  Line: pos["notional_value"] * 0.02  # Assume 2% risk per position
- ai_trading/risk/adaptive_sizing.py:551 - 0.02 (old 2% value)
  Line: atr_value = market_data.get("atr", entry_price * 0.02)  # 2% fallback
- ai_trading/risk/metrics.py:92 - 0.02 (old 2% value)
  Line: def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
- ai_trading/risk/metrics.py:115 - 0.02 (old 2% value)
  Line: def calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
- ai_trading/risk/circuit_breakers.py:302 - 1000 (old MAX_POSITION_SIZE)
  Line: self.max_daily_trades = 1000  # From SYSTEM_LIMITS
- ai_trading/database/models.py:180 - 1000 (old MAX_POSITION_SIZE)
  Line: var_score = min(abs(self.var_95) * 1000, 50)  # Scale VaR

## ⚙️ Configuration Files with Parameters
- hyperparams.json - CAPITAL_CAP
  Snippet: "CAPITAL_CAP": 0.25,
- backup/test_backup/hyperparams.json - CAPITAL_CAP
  Snippet: "CAPITAL_CAP": 0.25,
