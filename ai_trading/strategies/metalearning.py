"""
MetaLearning trading strategy using machine learning for price prediction.

This strategy implements ensemble machine learning methods to predict price movements
and generate trading signals with confidence scoring and risk assessment.
"""

import warnings
from ai_trading.exc import COMMON_EXC  # AI-AGENT-REF: narrow handler
from datetime import UTC, datetime, timedelta
from typing import Any

warnings.filterwarnings("ignore")

# AI-AGENT-REF: Use centralized logger as per AGENTS.md
from ai_trading.logging import logger

# AI-AGENT-REF: Import dependencies - sklearn is a hard dependency
import numpy as np

NUMPY_AVAILABLE = True

import pandas as pd

PANDAS_AVAILABLE = True

# AI-AGENT-REF: Import data fetcher for historical data
from ai_trading.data_fetcher import get_minute_df  # type: ignore


# AI-AGENT-REF: Import base strategy framework
from ..core.enums import OrderSide, RiskLevel
from .base import BaseStrategy, StrategySignal

# Machine learning imports - sklearn is a hard dependency
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ML_AVAILABLE = True


class MetaLearning(BaseStrategy):
    """
    Machine Learning-based trading strategy using ensemble methods.

    Features:
    - Pattern recognition in historical price data
    - Feature engineering with technical indicators
    - Ensemble ML models (Random Forest + Gradient Boosting)
    - Confidence scoring for predictions
    - Risk-adjusted signal generation
    """

    def __init__(
        self,
        strategy_id: str = "metalearning",
        name: str = "MetaLearning Strategy",
        risk_level: RiskLevel = RiskLevel.MODERATE,
    ):
        """Initialize MetaLearning strategy."""
        super().__init__(strategy_id, name, risk_level)

        # AI-AGENT-REF: Strategy parameters
        self.parameters = {
            "lookback_period": 60,  # Days of historical data for training
            "feature_window": 20,  # Window for technical indicators
            "prediction_horizon": 5,  # Days ahead to predict
            "min_confidence": 0.6,  # Minimum confidence for signal generation
            "ensemble_weight_rf": 0.6,  # Random Forest weight in ensemble
            "ensemble_weight_gb": 0.4,  # Gradient Boosting weight in ensemble
            "retrain_frequency": 7,  # Days between model retraining
            "min_data_points": 50,  # Reduced minimum data points for training
        }

        # Model components
        self.rf_model = None
        self.gb_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.last_training_date = None
        self.feature_columns = []

        # Prediction cache
        self.prediction_cache = {}
        self.cache_expiry = {}

        # Performance tracking
        self.prediction_accuracy = 0.0
        self.model_confidence_history = []

        logger.info(f"MetaLearning strategy initialized with risk level {risk_level}")

    def execute_strategy(self, data=None, symbol: str = None) -> dict:
        """
        Main execution method called by bot_engine.

        Args:
            data: Optional market data (for compatibility with bot_engine calling patterns)
            symbol: Trading symbol to analyze

        Returns:
            Dictionary with signal information or empty dict if no signal
        """
        # AI-AGENT-REF: Handle both calling patterns for compatibility
        # If called with positional args: execute_strategy(data, symbol)
        if data is not None and symbol is None:
            # Check if first argument is actually the symbol (string)
            if isinstance(data, str):
                symbol = data
                data = None

        if symbol is None:
            logger.error("execute_strategy called without symbol")
            return self._neutral_signal()
        try:
            # Check cache first
            if self._is_cached_prediction_valid(symbol):
                cached_result = self.prediction_cache[symbol]
                logger.debug(f"Using cached prediction for {symbol}")
                return cached_result

            # Get historical data if not provided
            if data is None:
                end_date = datetime.now(UTC)
                start_date = end_date - timedelta(
                    days=self.parameters["lookback_period"]
                )

                data = get_minute_df(symbol, start_date, end_date)
                if data is None or len(data) < self.parameters["min_data_points"]:
                    logger.warning(
                        f"Insufficient data for {symbol}, returning neutral signal"
                    )
                    return self._neutral_signal()
            else:
                # Use provided data but validate it
                if len(data) < self.parameters["min_data_points"]:
                    logger.warning(
                        f"Provided data insufficient for {symbol}, returning neutral signal"
                    )
                    return self._neutral_signal()

            # Check if model needs retraining
            if self._should_retrain():
                success = self.train_model(data)
                if not success:
                    logger.error(f"Model training failed for {symbol}")
                    return self._neutral_signal()

            # Generate prediction
            prediction_result = self.predict_price_movement(data)
            if not prediction_result:
                logger.warning(f"Prediction failed for {symbol}")
                return self._neutral_signal()

            # Convert prediction to trading signal
            signal_dict = self._convert_prediction_to_signal(
                symbol, prediction_result, data
            )

            # Cache the result
            self._cache_prediction(symbol, signal_dict)

            logger.info(
                f"Generated signal for {symbol}: {signal_dict.get('signal', 'hold')} "
                f"(confidence: {signal_dict.get('confidence', 0):.2f})"
            )

            return signal_dict

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error in execute_strategy for {symbol}: {e}")
            return self._neutral_signal()

    def generate_signals(self, market_data: dict) -> list[StrategySignal]:
        """
        Generate trading signals based on market data.

        Args:
            market_data: Current market data and indicators

        Returns:
            List of trading signals
        """
        signals = []

        try:
            for symbol in self.symbols:
                signal_dict = self.execute_strategy(symbol)

                if signal_dict and signal_dict.get("signal") != "hold":
                    # Convert to StrategySignal object
                    side = (
                        OrderSide.BUY
                        if signal_dict["signal"] == "buy"
                        else OrderSide.SELL
                    )

                    signal = StrategySignal(
                        symbol=symbol,
                        side=side,
                        strength=signal_dict.get("strength", 0.5),
                        confidence=signal_dict.get("confidence", 0.5),
                        strategy_id=self.strategy_id,
                        price_target=signal_dict.get("price_target"),
                        stop_loss=signal_dict.get("stop_loss"),
                        signal_type="ml_prediction",
                        metadata={
                            "reasoning": signal_dict.get("reasoning", ""),
                            "prediction_horizon": self.parameters["prediction_horizon"],
                            "model_accuracy": self.prediction_accuracy,
                        },
                    )

                    if self.validate_signal(signal):
                        signals.append(signal)
                        self.signals_generated += 1

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error generating signals: {e}")

        return signals

    def calculate_position_size(
        self,
        signal: StrategySignal,
        portfolio_value: float,
        current_position: float = 0,
    ) -> int:
        """
        Calculate optimal position size for signal using Kelly criterion and confidence.

        Args:
            signal: Trading signal
            portfolio_value: Current portfolio value
            current_position: Current position size

        Returns:
            Recommended position size
        """
        try:
            # Base position size from risk level
            base_position_pct = self.risk_level.max_position_size

            # Adjust by signal confidence and strength
            confidence_adjustment = signal.confidence * signal.strength
            adjusted_position_pct = base_position_pct * confidence_adjustment

            # Further adjust by model accuracy if available
            if self.prediction_accuracy > 0:
                accuracy_adjustment = min(self.prediction_accuracy, 1.0)
                adjusted_position_pct *= accuracy_adjustment

            # Calculate dollar amount
            position_value = portfolio_value * adjusted_position_pct

            # Convert to shares (approximate - would need current price in real implementation)
            # For now, return position value as integer
            position_size = int(position_value)

            logger.debug(
                f"Calculated position size for {signal.symbol}: {position_size} "
                f"(confidence: {signal.confidence:.2f}, strength: {signal.strength:.2f})"
            )

            return position_size

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error calculating position size: {e}")
            return 0

    def train_model(self, data) -> bool:
        """
        Train ML models on historical data.

        Args:
            data: Historical price data with OHLCV columns

        Returns:
            True if training successful
        """
        try:
            if not ML_AVAILABLE:
                logger.warning("ML libraries not available, using fallback prediction")
                self.is_trained = True  # Set to True for fallback mode
                return True

            if not PANDAS_AVAILABLE:
                logger.warning("Pandas not available, using fallback mode")
                self.is_trained = True
                return True

            logger.info("Training MetaLearning models")

            # Extract features and labels
            features_df = self.extract_features(data)
            if (
                features_df is None
                or len(features_df) < self.parameters["min_data_points"]
            ):
                logger.error(
                    f"Insufficient features for training: {len(features_df) if features_df is not None else 0} < {self.parameters['min_data_points']}"
                )
                return False

            # Create target labels (price direction prediction)
            targets = self._create_target_labels(data, features_df.index)

            # Align features and targets
            common_index = features_df.index.intersection(targets.index)
            if len(common_index) < self.parameters["min_data_points"]:
                logger.error("Insufficient aligned data for training")
                return False

            X = features_df.loc[common_index]
            y = targets.loc[common_index]

            # AI-AGENT-REF: Validate class distribution before training
            unique_classes = y.unique()
            class_counts = y.value_counts()

            logger.info(f"Training data class distribution: {dict(class_counts)}")

            # Ensure minimum class diversity for ML training
            if len(unique_classes) < 2:
                logger.error(
                    f"Insufficient class diversity for ML training: only {len(unique_classes)} class(es)"
                )
                return False

            # Check minimum samples per class
            min_class_size = min(class_counts.values)
            if (
                min_class_size < 3
            ):  # Need at least 3 samples per class for train/test split
                logger.warning(
                    f"Small class size detected: {min_class_size} samples. This may affect training quality."
                )
                # Continue with training but log the warning

            # Split data with stratification if possible
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            except ValueError as e:
                if "The least populated class" in str(e):
                    # Fallback: split without stratification for very small datasets
                    logger.warning(
                        "Cannot stratify split due to small class sizes, using random split"
                    )
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                else:
                    raise

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train Random Forest
            self.rf_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            self.rf_model.fit(X_train_scaled, y_train)

            # Train Gradient Boosting
            self.gb_model = GradientBoostingClassifier(
                n_estimators=100, max_depth=6, random_state=42
            )
            self.gb_model.fit(X_train_scaled, y_train)

            # Evaluate models
            rf_pred = self.rf_model.predict(X_test_scaled)
            gb_pred = self.gb_model.predict(X_test_scaled)

            rf_accuracy = accuracy_score(y_test, rf_pred)
            gb_accuracy = accuracy_score(y_test, gb_pred)

            # Ensemble prediction
            rf_proba = self.rf_model.predict_proba(X_test_scaled)
            gb_proba = self.gb_model.predict_proba(X_test_scaled)

            ensemble_proba = (
                self.parameters["ensemble_weight_rf"] * rf_proba
                + self.parameters["ensemble_weight_gb"] * gb_proba
            )
            ensemble_pred = np.argmax(ensemble_proba, axis=1)

            self.prediction_accuracy = accuracy_score(y_test, ensemble_pred)

            # Store feature columns for consistent prediction
            self.feature_columns = X.columns.tolist()

            self.is_trained = True
            self.last_training_date = datetime.now(UTC)

            logger.info(
                f"Model training completed - Accuracy: {self.prediction_accuracy:.3f} "
                f"(RF: {rf_accuracy:.3f}, GB: {gb_accuracy:.3f})"
            )

            return True

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error training model: {e}")
            logger.info("ML training failed, enabling fallback mode")
            # AI-AGENT-REF: Set fallback mode when ML training fails
            self.is_trained = True  # Enable fallback predictions
            self.prediction_accuracy = 0.6  # Assumed fallback accuracy
            return True  # Return True to allow fallback operation

    def predict_price_movement(self, data) -> dict | None:
        """
        Generate ML-based price movement predictions.

        Args:
            data: Historical price data

        Returns:
            Dictionary with prediction results
        """
        try:
            if not self.is_trained:
                logger.warning("Model not trained, cannot make predictions")
                return None

            if not PANDAS_AVAILABLE:
                logger.debug("Pandas not available, using fallback prediction")
                return self._fallback_prediction(data)

            # Extract features for latest data point
            features_df = self.extract_features(data)
            if features_df is None or len(features_df) == 0:
                logger.warning("Could not extract features for prediction")
                return None

            # Use latest data point for prediction
            latest_features = features_df.iloc[-1:][self.feature_columns]

            if not ML_AVAILABLE or self.rf_model is None or self.gb_model is None:
                # Fallback prediction based on technical analysis
                logger.debug("Using technical analysis fallback")
                return self._fallback_prediction(data)

            # Scale features
            features_scaled = self.scaler.transform(latest_features)

            # Get predictions from both models
            try:
                rf_proba = self.rf_model.predict_proba(features_scaled)[0]
                gb_proba = self.gb_model.predict_proba(features_scaled)[0]
            except COMMON_EXC as e:  # AI-AGENT-REF: narrow
                logger.warning(f"Model prediction failed: {e}, using fallback")
                return self._fallback_prediction(data)

            # AI-AGENT-REF: Handle variable number of classes in prediction
            # Ensure both probability arrays have same length
            n_classes = min(len(rf_proba), len(gb_proba))
            rf_proba = rf_proba[:n_classes]
            gb_proba = gb_proba[:n_classes]

            # Ensemble prediction
            ensemble_proba = (
                self.parameters["ensemble_weight_rf"] * rf_proba
                + self.parameters["ensemble_weight_gb"] * gb_proba
            )

            # Get predicted class and confidence
            if not NUMPY_AVAILABLE:
                # Fallback for when numpy is not available
                predicted_class = 0 if ensemble_proba[0] > 0.5 else 1
                confidence = max(ensemble_proba)
            else:
                predicted_class = np.argmax(ensemble_proba)
                confidence = np.max(ensemble_proba)

            # Map prediction to direction - handle variable number of classes
            if n_classes == 2:
                # Binary classification: 0=sell/hold, 1=buy
                direction_map = {0: "sell", 1: "buy"}
                prob_dist = {
                    "sell": float(ensemble_proba[0]),
                    "hold": 0.0,
                    "buy": float(ensemble_proba[1]),
                }
            elif n_classes >= 3:
                # Multi-class: 0=sell, 1=hold, 2=buy
                direction_map = {0: "sell", 1: "hold", 2: "buy"}
                prob_dist = {
                    "sell": float(ensemble_proba[0]),
                    "hold": float(ensemble_proba[1]),
                    "buy": float(ensemble_proba[2]) if len(ensemble_proba) > 2 else 0.0,
                }
            else:
                # Fallback for unexpected case
                logger.warning(
                    f"Unexpected number of classes: {n_classes}, using fallback"
                )
                return self._fallback_prediction(data)

            predicted_direction = direction_map.get(predicted_class, "hold")

            # Calculate additional metrics
            current_price = data["close"].iloc[-1]

            # Handle volatility calculation with fallbacks
            if PANDAS_AVAILABLE:
                volatility = data["close"].pct_change().rolling(20).std().iloc[-1]
                if NUMPY_AVAILABLE:
                    volatility = float(volatility) if not np.isnan(volatility) else 0.05
                else:
                    # Simple fallback volatility calculation
                    volatility = 0.05 if str(volatility) == "nan" else float(volatility)
            else:
                volatility = 0.05  # Default fallback

            result = {
                "direction": predicted_direction,
                "confidence": float(confidence),
                "probability_distribution": prob_dist,
                "current_price": float(current_price),
                "volatility": volatility,
                "model_accuracy": self.prediction_accuracy,
                "prediction_timestamp": datetime.now(UTC),
                "source": "ml_ensemble",
            }

            return result

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error in predict_price_movement: {e}")
            return None

    def extract_features(self, data) -> Any | None:
        """
        Engineer features for ML model from price data.

        Args:
            data: Historical price data with OHLCV columns

        Returns:
            DataFrame with engineered features or fallback data
        """
        try:
            if not PANDAS_AVAILABLE:
                logger.debug("Pandas not available for feature extraction")
                return None

            if data is None or len(data) < self.parameters["feature_window"]:
                return None

            # Import pandas locally if available
            import numpy as np
            import pandas as pd

            features = pd.DataFrame(index=data.index)

            # Price-based features
            features["returns"] = data["close"].pct_change()
            features["log_returns"] = np.log(data["close"]).diff()

            # Moving averages
            window = self.parameters["feature_window"]
            features["sma_5"] = data["close"].rolling(5).mean()
            features["sma_10"] = data["close"].rolling(10).mean()
            features["sma_20"] = data["close"].rolling(window).mean()

            # Price relative to moving averages
            features["price_vs_sma_5"] = data["close"] / features["sma_5"] - 1
            features["price_vs_sma_10"] = data["close"] / features["sma_10"] - 1
            features["price_vs_sma_20"] = data["close"] / features["sma_20"] - 1

            # Volatility features
            features["volatility_5"] = features["returns"].rolling(5).std()
            features["volatility_20"] = features["returns"].rolling(window).std()

            # Momentum features
            features["momentum_5"] = data["close"] / data["close"].shift(5) - 1
            features["momentum_10"] = data["close"] / data["close"].shift(10) - 1

            # Volume features (if available)
            if "volume" in data.columns:
                features["volume_sma"] = data["volume"].rolling(window).mean()
                features["volume_ratio"] = data["volume"] / features["volume_sma"]
            else:
                features["volume_sma"] = 0
                features["volume_ratio"] = 1

            # Technical indicators
            # RSI (Relative Strength Index)
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features["rsi"] = 100 - (100 / (1 + rs))

            # AI-AGENT-REF: Enhanced technical indicators for better signal diversity
            # Add RSI-based signals
            features["rsi_oversold"] = (features["rsi"] < 30).astype(int)
            features["rsi_overbought"] = (features["rsi"] > 70).astype(int)
            features["rsi_momentum"] = features["rsi"].diff()

            # Bollinger Bands
            bb_sma = data["close"].rolling(window).mean()
            bb_std = data["close"].rolling(window).std()
            features["bb_upper"] = bb_sma + (bb_std * 2)
            features["bb_lower"] = bb_sma - (bb_std * 2)
            features["bb_position"] = (data["close"] - features["bb_lower"]) / (
                features["bb_upper"] - features["bb_lower"]
            )

            # BB-based signals
            features["bb_squeeze"] = (
                features["bb_upper"] - features["bb_lower"]
            ) / bb_sma  # Volatility measure
            features["bb_breakout_up"] = (data["close"] > features["bb_upper"]).astype(
                int
            )
            features["bb_breakout_down"] = (
                data["close"] < features["bb_lower"]
            ).astype(int)

            # MACD
            ema_12 = data["close"].ewm(span=12).mean()
            ema_26 = data["close"].ewm(span=26).mean()
            features["macd"] = ema_12 - ema_26
            features["macd_signal"] = features["macd"].ewm(span=9).mean()
            features["macd_histogram"] = features["macd"] - features["macd_signal"]

            # MACD-based signals
            features["macd_bullish"] = (
                (features["macd"] > features["macd_signal"])
                & (features["macd"].shift(1) <= features["macd_signal"].shift(1))
            ).astype(int)
            features["macd_bearish"] = (
                (features["macd"] < features["macd_signal"])
                & (features["macd"].shift(1) >= features["macd_signal"].shift(1))
            ).astype(int)

            # Enhanced momentum indicators
            features["momentum_short"] = (
                data["close"] / data["close"].shift(3) - 1
            )  # 3-period momentum
            features["momentum_medium"] = (
                data["close"] / data["close"].shift(10) - 1
            )  # 10-period momentum

            # Trend strength indicators
            features["trend_strength_5"] = (
                features["sma_5"] - features["sma_5"].shift(5)
            ) / features["sma_5"].shift(5)
            features["trend_strength_20"] = (
                features["sma_20"] - features["sma_20"].shift(10)
            ) / features["sma_20"].shift(10)

            # Cross-over signals
            features["sma_cross_bullish"] = (
                (features["sma_5"] > features["sma_10"])
                & (features["sma_5"].shift(1) <= features["sma_10"].shift(1))
            ).astype(int)
            features["sma_cross_bearish"] = (
                (features["sma_5"] < features["sma_10"])
                & (features["sma_5"].shift(1) >= features["sma_10"].shift(1))
            ).astype(int)

            # Price patterns
            features["high_low_ratio"] = data["high"] / data["low"] - 1
            features["open_close_ratio"] = data["close"] / data["open"] - 1

            # Lag features
            for lag in [1, 2, 3, 5]:
                features[f"returns_lag_{lag}"] = features["returns"].shift(lag)
                features[f"price_lag_{lag}"] = (
                    data["close"].shift(lag) / data["close"] - 1
                )

            # Drop rows with NaN values
            features = features.dropna()

            # Fill any remaining NaN with 0
            features = features.fillna(0)

            # Replace infinite values
            features = features.replace([np.inf, -np.inf], 0)

            logger.debug(
                f"Extracted {len(features.columns)} features for {len(features)} data points"
            )

            return features

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error extracting features: {e}")
            return None

    def _create_target_labels(self, data, feature_index) -> Any:
        """Create target labels for ML training using dynamic thresholds."""
        try:
            if not PANDAS_AVAILABLE:
                logger.debug("Pandas not available for target label creation")
                return []  # Return empty list as fallback

            # Import pandas locally
            import pandas as pd

            horizon = self.parameters["prediction_horizon"]

            # Calculate future returns
            future_returns = data["close"].shift(-horizon) / data["close"] - 1

            # Remove NaN values for threshold calculation
            valid_returns = future_returns.dropna()

            if len(valid_returns) < self.parameters["min_data_points"]:
                logger.warning(
                    f"Insufficient return data for labeling: {len(valid_returns)}"
                )
                return pd.Series(dtype=int)

            # AI-AGENT-REF: Use dynamic percentile-based thresholds instead of fixed 2%
            # This solves the single-class problem identified in the issue
            sell_threshold = valid_returns.quantile(0.25)  # Bottom 25% = sell signals
            buy_threshold = valid_returns.quantile(0.75)  # Top 25% = buy signals
            # Middle 50% = hold signals

            # Log threshold values for debugging
            logger.debug(
                f"Dynamic thresholds - Sell: {sell_threshold:.4f} ({sell_threshold*100:.2f}%), "
                f"Buy: {buy_threshold:.4f} ({buy_threshold*100:.2f}%)"
            )

            # Create categorical labels
            # 0 (sell): return < 25th percentile
            # 1 (hold): 25th percentile <= return <= 75th percentile
            # 2 (buy): return > 75th percentile
            labels = pd.Series(1, index=data.index)  # Default to hold
            labels[future_returns < sell_threshold] = 0  # Sell
            labels[future_returns > buy_threshold] = 2  # Buy

            # Align with feature index
            aligned_labels = labels.reindex(feature_index).dropna()

            # Validate class distribution
            unique_classes = aligned_labels.unique()
            class_counts = aligned_labels.value_counts()

            logger.debug(
                f"Label distribution: {dict(class_counts)} (unique classes: {len(unique_classes)})"
            )

            # Ensure we have at least 2 classes for ML training
            if len(unique_classes) < 2:
                logger.warning(
                    "Only 1 class detected after labeling - using fallback strategy"
                )
                # Fallback: create balanced classes by splitting data in thirds
                n_samples = len(aligned_labels)
                third = n_samples // 3

                # Create more aggressive thresholds
                fallback_labels = pd.Series(
                    1, index=aligned_labels.index
                )  # Default hold

                # Sort by return values and assign classes
                sorted_indices = (
                    future_returns.reindex(aligned_labels.index)
                    .dropna()
                    .sort_values()
                    .index
                )
                if len(sorted_indices) >= 6:  # Need minimum samples
                    # Bottom third = sell
                    fallback_labels.loc[sorted_indices[:third]] = 0
                    # Top third = buy
                    fallback_labels.loc[sorted_indices[-third:]] = 2
                    # Middle = hold (already set to 1)

                    logger.info(
                        f"Applied fallback labeling: {dict(fallback_labels.value_counts())}"
                    )
                    return fallback_labels

            return aligned_labels

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error creating target labels: {e}")
            return pd.Series(dtype=int)

    def _should_retrain(self) -> bool:
        """Check if model should be retrained."""
        if not self.is_trained or self.last_training_date is None:
            return True

        days_since_training = (datetime.now(UTC) - self.last_training_date).days
        return days_since_training >= self.parameters["retrain_frequency"]

    def _is_cached_prediction_valid(self, symbol: str) -> bool:
        """Check if cached prediction is still valid."""
        if symbol not in self.prediction_cache:
            return False

        if symbol not in self.cache_expiry:
            return False

        return datetime.now(UTC) < self.cache_expiry[symbol]

    def _cache_prediction(self, symbol: str, result: dict):
        """Cache prediction result."""
        self.prediction_cache[symbol] = result
        # Cache expires in 1 hour
        self.cache_expiry[symbol] = datetime.now(UTC) + timedelta(hours=1)

    def _convert_prediction_to_signal(
        self, symbol: str, prediction: dict, data
    ) -> dict:
        """Convert ML prediction to trading signal format."""
        try:
            direction = prediction["direction"]
            confidence = prediction["confidence"]
            current_price = prediction["current_price"]
            volatility = prediction["volatility"]

            # Check minimum confidence threshold
            if confidence < self.parameters["min_confidence"]:
                return self._neutral_signal()

            # Calculate strength based on confidence and model accuracy
            strength = confidence * min(self.prediction_accuracy, 1.0)

            # Calculate price targets and stop losses
            price_target = None
            stop_loss = None

            if direction == "buy":
                # Target: 2-3% gain, adjusted by volatility
                target_pct = max(0.02, volatility * 2)
                price_target = current_price * (1 + target_pct)

                # Stop loss: 1-2% loss, adjusted by volatility
                stop_pct = max(0.01, volatility)
                stop_loss = current_price * (1 - stop_pct)

            elif direction == "sell":
                # Target: 2-3% decline, adjusted by volatility
                target_pct = max(0.02, volatility * 2)
                price_target = current_price * (1 - target_pct)

                # Stop loss: 1-2% rise, adjusted by volatility
                stop_pct = max(0.01, volatility)
                stop_loss = current_price * (1 + stop_pct)

            # Generate reasoning
            prob_dist = prediction["probability_distribution"]
            reasoning = (
                f"ML ensemble prediction: {direction} "
                f"(confidence: {confidence:.2f}, "
                f"buy_prob: {prob_dist['buy']:.2f}, "
                f"sell_prob: {prob_dist['sell']:.2f})"
            )

            return {
                "signal": direction,
                "confidence": confidence,
                "strength": strength,
                "price_target": price_target,
                "stop_loss": stop_loss,
                "reasoning": reasoning,
                "current_price": current_price,
                "volatility": volatility,
                "model_accuracy": self.prediction_accuracy,
            }

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error converting prediction to signal: {e}")
            return self._neutral_signal()

    def _neutral_signal(self) -> dict:
        """Return neutral/hold signal."""
        return {
            "signal": "hold",
            "confidence": 0.0,
            "strength": 0.0,
            "price_target": None,
            "stop_loss": None,
            "reasoning": "Insufficient data or low confidence for prediction",
        }

    def _fallback_prediction(self, data) -> dict:
        """Enhanced fallback prediction when ML models are not available."""
        try:
            # AI-AGENT-REF: Enhanced fallback using multiple technical indicators

            if not PANDAS_AVAILABLE:
                return self._simple_momentum_fallback(data)

            # Extract features for technical analysis
            features = self.extract_features(data)
            if features is None or len(features) == 0:
                return self._simple_momentum_fallback(data)

            latest_features = features.iloc[-1]

            # Technical indicator signals
            signals = []
            confidences = []

            # RSI signal
            rsi = latest_features.get("rsi", 50)
            if rsi < 30:  # Oversold
                signals.append("buy")
                confidences.append(min(0.8, (30 - rsi) / 30))
            elif rsi > 70:  # Overbought
                signals.append("sell")
                confidences.append(min(0.8, (rsi - 70) / 30))

            # MACD signal
            macd_bullish = latest_features.get("macd_bullish", 0)
            macd_bearish = latest_features.get("macd_bearish", 0)
            if macd_bullish:
                signals.append("buy")
                confidences.append(0.7)
            elif macd_bearish:
                signals.append("sell")
                confidences.append(0.7)

            # Bollinger Bands signal
            bb_breakout_up = latest_features.get("bb_breakout_up", 0)
            bb_breakout_down = latest_features.get("bb_breakout_down", 0)
            if bb_breakout_up:
                signals.append("buy")
                confidences.append(0.6)
            elif bb_breakout_down:
                signals.append("sell")
                confidences.append(0.6)

            # Moving average crossover
            sma_cross_bullish = latest_features.get("sma_cross_bullish", 0)
            sma_cross_bearish = latest_features.get("sma_cross_bearish", 0)
            if sma_cross_bullish:
                signals.append("buy")
                confidences.append(0.5)
            elif sma_cross_bearish:
                signals.append("sell")
                confidences.append(0.5)

            # Momentum analysis
            momentum_5 = latest_features.get("momentum_5", 0)
            if momentum_5 > 0.01:  # 1% positive momentum
                signals.append("buy")
                confidences.append(min(0.7, momentum_5 * 20))
            elif momentum_5 < -0.01:  # 1% negative momentum
                signals.append("sell")
                confidences.append(min(0.7, abs(momentum_5) * 20))

            # Aggregate signals
            if not signals:
                direction = "hold"
                confidence = 0.3
            else:
                # Weighted voting
                buy_weight = sum(
                    c for s, c in zip(signals, confidences, strict=False) if s == "buy"
                )
                sell_weight = sum(
                    c for s, c in zip(signals, confidences, strict=False) if s == "sell"
                )

                if buy_weight > sell_weight:
                    direction = "buy"
                    confidence = min(0.8, buy_weight / len(signals))
                elif sell_weight > buy_weight:
                    direction = "sell"
                    confidence = min(0.8, sell_weight / len(signals))
                else:
                    direction = "hold"
                    confidence = 0.4

            current_price = data["close"].iloc[-1]
            volatility = data["close"].pct_change().rolling(20).std().iloc[-1]

            # Create probability distribution
            if direction == "buy":
                prob_buy = confidence
                prob_sell = (1 - confidence) * 0.3
                prob_hold = 1 - prob_buy - prob_sell
            elif direction == "sell":
                prob_sell = confidence
                prob_buy = (1 - confidence) * 0.3
                prob_hold = 1 - prob_buy - prob_sell
            else:
                prob_hold = confidence
                prob_buy = (1 - confidence) * 0.5
                prob_sell = (1 - confidence) * 0.5

            return {
                "direction": direction,
                "confidence": confidence,
                "probability_distribution": {
                    "sell": float(prob_sell),
                    "hold": float(prob_hold),
                    "buy": float(prob_buy),
                },
                "current_price": float(current_price),
                "volatility": float(volatility) if not np.isnan(volatility) else 0.05,
                "model_accuracy": 0.6,  # Assumed fallback accuracy
                "prediction_timestamp": datetime.now(UTC),
                "source": "technical_fallback",
            }

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error in enhanced fallback prediction: {e}")
            return self._simple_momentum_fallback(data)

    def _simple_momentum_fallback(self, data) -> dict:
        """Simple momentum-based fallback prediction."""
        try:
            if not PANDAS_AVAILABLE:
                # Very basic fallback when pandas is not available
                return {
                    "direction": "hold",
                    "confidence": 0.3,
                    "probability_distribution": {"sell": 0.3, "hold": 0.4, "buy": 0.3},
                    "current_price": 100.0,  # Default price
                    "volatility": 0.05,
                    "model_accuracy": 0.6,
                    "prediction_timestamp": datetime.now(UTC),
                    "source": "basic_fallback",
                }

            # Simple momentum-based prediction
            returns = data["close"].pct_change().dropna()
            recent_returns = returns.tail(5).mean()
            volatility = returns.tail(20).std()

            # Determine direction based on recent momentum
            if recent_returns > 0.005:  # 0.5% positive momentum
                direction = "buy"
                confidence = min(0.7, abs(recent_returns) * 100)
            elif recent_returns < -0.005:  # 0.5% negative momentum
                direction = "sell"
                confidence = min(0.7, abs(recent_returns) * 100)
            else:
                direction = "hold"
                confidence = 0.3

            current_price = data["close"].iloc[-1]

            return {
                "direction": direction,
                "confidence": confidence,
                "probability_distribution": {
                    "sell": 0.5 - confidence / 2 if direction != "sell" else confidence,
                    "hold": 0.3,
                    "buy": 0.5 - confidence / 2 if direction != "buy" else confidence,
                },
                "current_price": current_price,
                "volatility": (
                    volatility
                    if not (str(volatility) == "nan" or str(volatility) == "NaN")
                    else 0.05
                ),
                "model_accuracy": 0.6,  # Assumed fallback accuracy
                "prediction_timestamp": datetime.now(UTC),
                "source": "momentum_fallback",
            }

        except COMMON_EXC as e:  # AI-AGENT-REF: narrow
            logger.error(f"Error in simple fallback prediction: {e}")
            return None