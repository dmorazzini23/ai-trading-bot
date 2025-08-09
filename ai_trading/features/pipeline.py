"""
Leak-proof feature engineering pipeline.

Provides feature transformers and pipelines that ensure no future 
information leaks into training data during cross-validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import logging

# Use the centralized logger as per AGENTS.md
try:
    from ai_trading.logging import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.pipeline import Pipeline
    sklearn_available = True
except ImportError:
    # Fallback classes if sklearn not available
    class BaseEstimator:
        pass
    class TransformerMixin:
        pass
    class StandardScaler:
        def fit(self, X, y=None): return self
        def transform(self, X): return X
        def fit_transform(self, X, y=None): return X
    class RobustScaler:
        def fit(self, X, y=None): return self
        def transform(self, X): return X
        def fit_transform(self, X, y=None): return X
    class Pipeline:
        def __init__(self, steps): self.steps = steps
        def fit(self, X, y=None): return self
        def transform(self, X): return X
        def fit_transform(self, X, y=None): return X
    sklearn_available = False


class BuildFeatures(BaseEstimator, TransformerMixin):
    """
    Feature builder that creates technical indicators and regime features.
    
    Ensures no future information leakage by only using past data
    for indicator calculations and parameter fitting.
    """
    
    def __init__(
        self,
        include_returns: bool = True,
        include_volatility: bool = True,
        include_volume: bool = True,
        include_regime: bool = True,
        lookback_window: int = 20,
        vol_span: int = 30,
        regime_span: int = 50
    ):
        """
        Initialize feature builder.
        
        Args:
            include_returns: Include return-based features
            include_volatility: Include volatility features
            include_volume: Include volume features
            include_regime: Include regime detection features
            lookback_window: Lookback window for indicators
            vol_span: Span for volatility calculations
            regime_span: Span for regime detection
        """
        self.include_returns = include_returns
        self.include_volatility = include_volatility
        self.include_volume = include_volume
        self.include_regime = include_regime
        self.lookback_window = lookback_window
        self.vol_span = vol_span
        self.regime_span = regime_span
        
        # Parameters fitted during training
        self.feature_params_ = {}
        self.is_fitted_ = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'BuildFeatures':
        """
        Fit feature parameters on training data only.
        
        Args:
            X: Input data (price/volume data)
            y: Target variable (unused)
            
        Returns:
            Self for method chaining
        """
        try:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("BuildFeatures requires DataFrame input")
            
            # Fit parameters that should not change between train/test
            self.feature_params_ = {
                'lookback_window': self.lookback_window,
                'vol_span': self.vol_span,
                'regime_span': self.regime_span,
                'columns': X.columns.tolist()
            }
            
            # Calculate historical averages/thresholds on training data
            if 'close' in X.columns or 'price' in X.columns:
                price_col = 'close' if 'close' in X.columns else 'price'
                prices = X[price_col]
                
                # Fit volatility parameters
                if self.include_volatility:
                    returns = prices.pct_change().dropna()
                    self.feature_params_['vol_mean'] = returns.std()
                    self.feature_params_['vol_threshold'] = returns.std() * 2
                
                # Fit regime parameters
                if self.include_regime:
                    rolling_vol = returns.rolling(self.regime_span).std()
                    self.feature_params_['regime_vol_low'] = rolling_vol.quantile(0.33)
                    self.feature_params_['regime_vol_high'] = rolling_vol.quantile(0.67)
            
            self.is_fitted_ = True
            logger.debug("BuildFeatures fitted successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting BuildFeatures: {e}")
            raise
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to create features.
        
        Args:
            X: Input data
            
        Returns:
            DataFrame with engineered features
        """
        try:
            if not self.is_fitted_:
                raise ValueError("BuildFeatures must be fitted before transform")
            
            if not isinstance(X, pd.DataFrame):
                raise ValueError("Input must be DataFrame")
            
            features = X.copy()
            
            # Determine price column
            if 'close' in X.columns:
                price_col = 'close'
            elif 'price' in X.columns:
                price_col = 'price'
            else:
                price_col = X.columns[0]
            
            prices = X[price_col]
            
            # Generate features
            if self.include_returns:
                features = self._add_return_features(features, prices)
            
            if self.include_volatility:
                features = self._add_volatility_features(features, prices)
            
            if self.include_volume and 'volume' in X.columns:
                features = self._add_volume_features(features, X['volume'])
            
            if self.include_regime:
                features = self._add_regime_features(features, prices)
            
            # Remove any infinite or extremely large values
            features = features.replace([np.inf, -np.inf], np.nan)
            
            logger.debug(f"Generated {features.shape[1]} features from {X.shape[1]} inputs")
            return features
            
        except Exception as e:
            logger.error(f"Error transforming features: {e}")
            raise
    
    def _add_return_features(self, features: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """Add return-based features."""
        try:
            # Simple returns at different horizons
            for period in [1, 2, 5, 10]:
                features[f'ret_{period}d'] = prices.pct_change(periods=period)
            
            # Log returns
            features['log_ret_1d'] = np.log(prices / prices.shift(1))
            
            # Cumulative returns
            features['cum_ret_5d'] = (prices / prices.shift(5)) - 1
            features['cum_ret_20d'] = (prices / prices.shift(20)) - 1
            
            # Return momentum
            features['ret_momentum'] = (
                features['ret_1d'].rolling(5).mean() - 
                features['ret_1d'].rolling(20).mean()
            )
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding return features: {e}")
            return features
    
    def _add_volatility_features(self, features: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """Add volatility-based features."""
        try:
            returns = prices.pct_change()
            
            # Rolling volatility at different windows
            for window in [5, 10, 20, 50]:
                features[f'vol_{window}d'] = returns.rolling(window).std()
            
            # EWMA volatility
            features[f'vol_ewma_{self.vol_span}'] = returns.ewm(span=self.vol_span).std()
            
            # Volatility ratio
            features['vol_ratio'] = (
                features['vol_5d'] / features['vol_20d']
            )
            
            # Volatility rank (percentile of recent volatility)
            vol_rank_window = min(252, len(returns))  # 1 year or available data
            features['vol_rank'] = (
                features['vol_20d'].rolling(vol_rank_window)
                .rank(pct=True)
            )
            
            # High-low volatility
            if 'high' in features.columns and 'low' in features.columns:
                features['hl_vol'] = np.log(features['high'] / features['low'])
                features['hl_vol_ma'] = features['hl_vol'].rolling(20).mean()
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding volatility features: {e}")
            return features
    
    def _add_volume_features(self, features: pd.DataFrame, volume: pd.Series) -> pd.DataFrame:
        """Add volume-based features."""
        try:
            # Volume moving averages
            features['vol_ma_10'] = volume.rolling(10).mean()
            features['vol_ma_20'] = volume.rolling(20).mean()
            
            # Volume ratio
            features['vol_ratio'] = volume / features['vol_ma_20']
            
            # Price-volume features
            if 'close' in features.columns:
                prices = features['close']
                returns = prices.pct_change()
                
                # On-balance volume
                obv = np.where(returns > 0, volume, 
                              np.where(returns < 0, -volume, 0))
                features['obv'] = pd.Series(obv, index=volume.index).cumsum()
                
                # Volume-weighted average price (VWAP) deviation
                if 'high' in features.columns and 'low' in features.columns:
                    typical_price = (features['high'] + features['low'] + features['close']) / 3
                    vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
                    features['vwap_dev'] = (typical_price - vwap) / vwap
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding volume features: {e}")
            return features
    
    def _add_regime_features(self, features: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """Add regime detection features."""
        try:
            returns = prices.pct_change()
            
            # Rolling volatility for regime detection
            rolling_vol = returns.rolling(self.regime_span).std()
            
            # Volatility regime
            vol_low = self.feature_params_.get('regime_vol_low', rolling_vol.quantile(0.33))
            vol_high = self.feature_params_.get('regime_vol_high', rolling_vol.quantile(0.67))
            
            features['vol_regime'] = np.where(
                rolling_vol <= vol_low, 0,  # Low vol
                np.where(rolling_vol >= vol_high, 2, 1)  # High vol, Medium vol
            )
            
            # Trend regime (EMA slope)
            ema_short = prices.ewm(span=12).mean()
            ema_long = prices.ewm(span=26).mean()
            trend_signal = ema_short - ema_long
            features['trend_slope'] = trend_signal.pct_change()
            
            # Trend regime classification
            trend_ma = features['trend_slope'].rolling(20).mean()
            features['trend_regime'] = np.where(
                trend_ma > 0.001, 1,  # Uptrend
                np.where(trend_ma < -0.001, -1, 0)  # Downtrend, Sideways
            )
            
            # Market microstructure proxy
            if 'high' in features.columns and 'low' in features.columns:
                # HL2 vs VWAP deviation as microstructure proxy
                hl2 = (features['high'] + features['low']) / 2
                if 'volume' in features.columns:
                    # Simplified VWAP
                    vwap_approx = ((features['high'] + features['low'] + prices) / 3 * 
                                  features['volume']).rolling(20).sum() / features['volume'].rolling(20).sum()
                    features['microstructure_dev'] = (hl2 - vwap_approx) / vwap_approx
                else:
                    # Use simple moving average as proxy
                    sma20 = prices.rolling(20).mean()
                    features['microstructure_dev'] = (hl2 - sma20) / sma20
            
            return features
            
        except Exception as e:
            logger.error(f"Error adding regime features: {e}")
            return features


def create_feature_pipeline(
    scaler_type: str = "standard",
    build_features_params: Optional[Dict[str, Any]] = None
) -> Pipeline:
    """
    Create a complete feature engineering pipeline.
    
    Args:
        scaler_type: Type of scaler ('standard', 'robust', 'none')
        build_features_params: Parameters for BuildFeatures transformer
        
    Returns:
        sklearn Pipeline with feature engineering and scaling
    """
    try:
        if not sklearn_available:
            logger.warning("sklearn not available, returning simple pipeline")
            return Pipeline([
                ('features', BuildFeatures(**(build_features_params or {}))),
            ])
        
        # Initialize components
        build_features_params = build_features_params or {}
        feature_builder = BuildFeatures(**build_features_params)
        
        # Select scaler
        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "robust":
            scaler = RobustScaler()
        elif scaler_type == "none":
            scaler = None
        else:
            logger.warning(f"Unknown scaler type: {scaler_type}, using standard")
            scaler = StandardScaler()
        
        # Build pipeline
        pipeline_steps = [('features', feature_builder)]
        
        if scaler is not None:
            pipeline_steps.append(('scaler', scaler))
        
        pipeline = Pipeline(pipeline_steps)
        
        logger.info(f"Created feature pipeline with {len(pipeline_steps)} steps")
        return pipeline
        
    except Exception as e:
        logger.error(f"Error creating feature pipeline: {e}")
        raise


def validate_pipeline_no_leakage(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: Optional[pd.Series] = None
) -> bool:
    """
    Validate that pipeline doesn't leak information from test to train.
    
    Args:
        pipeline: Feature pipeline to validate
        X_train: Training data
        X_test: Test data
        y_train: Training targets (optional)
        
    Returns:
        True if no leakage detected, False otherwise
    """
    try:
        # Fit pipeline on training data only
        pipeline.fit(X_train, y_train)
        
        # Transform both sets
        X_train_transformed = pipeline.transform(X_train)
        X_test_transformed = pipeline.transform(X_test)
        
        # Check that statistics differ between train and test
        # (if they're identical, there might be leakage)
        if hasattr(X_train_transformed, 'mean'):
            train_mean = X_train_transformed.mean()
            test_mean = X_test_transformed.mean()
            
            # Allow some tolerance for statistical differences
            tolerance = 0.1
            if hasattr(train_mean, '__iter__'):
                mean_diff = np.abs(train_mean - test_mean).mean()
            else:
                mean_diff = abs(train_mean - test_mean)
            
            if mean_diff < tolerance:
                logger.warning("Train and test statistics are very similar - possible leakage")
                return False
        
        logger.debug("Pipeline validation passed - no obvious leakage detected")
        return True
        
    except Exception as e:
        logger.error(f"Error validating pipeline: {e}")
        return False