"""
Leak-proof feature engineering pipeline.

Provides feature transformers and pipelines that ensure no future
information leaks into training data during cross-validation.
"""
from __future__ import annotations

from datetime import date
from typing import Any, TYPE_CHECKING
from zoneinfo import ZoneInfo
import numpy as np
from ai_trading.utils.lazy_imports import (
    load_pandas,
    load_sklearn_pipeline,
    load_sklearn_preprocessing,
)
from importlib.util import find_spec

try:
    sklearn_available = bool(find_spec("sklearn"))
except (ImportError, ValueError):
    sklearn_available = False
from ai_trading.logging import logger
from ai_trading.market.calendar_wrapper import previous_trading_session
_ET = ZoneInfo("America/New_York")

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

class BuildFeatures:
    """
    Feature builder that creates technical indicators and regime features.

    Ensures no future information leakage by only using past data
    for indicator calculations and parameter fitting.
    """

    def __init__(self, include_returns: bool=True, include_volatility: bool=True, include_volume: bool=True, include_regime: bool=True, lookback_window: int=20, vol_span: int=30, regime_span: int=50):
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
        self.feature_params_: dict[str, Any] = {}
        self.is_fitted_ = False
        self._lookback_tail_: Any | None = None
        self._lookback_tail_symbol_: str | None = None
        self._lookback_tail_last_session_: date | None = None
        self._lookback_tail_last_ts_: Any | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None=None) -> 'BuildFeatures':
        """
        Fit feature parameters on training data only.

        Args:
            X: Input data (price/volume data)
            y: Target variable (unused)

        Returns:
            Self for method chaining
        """
        pd = load_pandas()
        try:
            if not isinstance(X, pd.DataFrame):
                raise ValueError('BuildFeatures requires DataFrame input')
            lookback_rows = self._max_lookback_rows()
            self.feature_params_ = {'lookback_window': self.lookback_window, 'vol_span': self.vol_span, 'regime_span': self.regime_span, 'columns': X.columns.tolist(), 'lookback_tail_rows': lookback_rows}
            self._lookback_tail_ = X.tail(lookback_rows).copy()
            self._lookback_tail_symbol_ = self._single_symbol_value(self._lookback_tail_)
            tail_sessions = self._session_values(self._lookback_tail_)
            self._lookback_tail_last_session_ = tail_sessions[-1] if tail_sessions else None
            self._lookback_tail_last_ts_ = self._timestamp_values(self._lookback_tail_)[-1] if self._timestamp_values(self._lookback_tail_) else None
            if 'close' in X.columns or 'price' in X.columns:
                price_col = 'close' if 'close' in X.columns else 'price'
                prices = X[price_col]
                returns = prices.pct_change().dropna()
                if self.include_volatility:
                    self.feature_params_['vol_mean'] = returns.std()
                    self.feature_params_['vol_threshold'] = returns.std() * 2
                if self.include_regime:
                    rolling_vol = returns.rolling(self.regime_span).std()
                    vol_low = rolling_vol.quantile(0.33)
                    vol_high = rolling_vol.quantile(0.67)
                    fallback_vol = returns.std()
                    if not np.isfinite(vol_low):
                        vol_low = fallback_vol if np.isfinite(fallback_vol) else 0.0
                    if not np.isfinite(vol_high):
                        vol_high = fallback_vol if np.isfinite(fallback_vol) else 0.0
                    self.feature_params_['regime_vol_low'] = float(vol_low)
                    self.feature_params_['regime_vol_high'] = float(vol_high)
            self.is_fitted_ = True
            logger.debug('BuildFeatures fitted successfully')
            return self
        except (KeyError, ValueError, TypeError, pd.errors.EmptyDataError) as e:
            logger.error(f'Error fitting BuildFeatures: {e}')
            raise

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data to create features.

        Args:
            X: Input data

        Returns:
            DataFrame with engineered features
        """
        pd = load_pandas()
        try:
            if not self.is_fitted_:
                raise ValueError('BuildFeatures must be fitted before transform')
            if not isinstance(X, pd.DataFrame):
                raise ValueError('Input must be DataFrame')
            self._validate_raw_price_columns(X)
            source = self._with_lookback_tail(X)
            raw_columns = set(source.columns)
            features = source.copy()
            if 'close' in source.columns:
                price_col = 'close'
            elif 'price' in source.columns:
                price_col = 'price'
            else:
                price_col = source.columns[0]
            prices = source[price_col]
            if self.include_returns:
                features = self._add_return_features(features, prices)
            if self.include_volatility:
                features = self._add_volatility_features(features, prices)
            if self.include_volume and 'volume' in source.columns:
                features = self._add_volume_features(features, source['volume'])
            if self.include_regime:
                features = self._add_regime_features(features, prices)
            derived_columns = [col for col in features.columns if col not in raw_columns]
            if derived_columns:
                features[derived_columns] = (
                    features[derived_columns]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )
            if len(source) != len(X):
                features = features.iloc[-len(X):].copy()
            logger.debug(f'Generated {features.shape[1]} features from {X.shape[1]} inputs')
            return features
        except (KeyError, ValueError, TypeError, pd.errors.EmptyDataError) as e:
            logger.error(f'Error transforming features: {e}')
            raise

    def _max_lookback_rows(self) -> int:
        """Return the raw row tail needed to keep rolling features stable."""
        return max(
            int(self.lookback_window),
            int(self.vol_span),
            int(self.regime_span),
            50,
        )

    def _with_lookback_tail(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepend fitted training tail only for the adjacent same-symbol session."""
        pd = load_pandas()
        tail = self._lookback_tail_
        if tail is None or len(tail) == 0 or len(X) == 0:
            return X
        if list(tail.columns) != list(X.columns):
            return X
        if not self._same_tail_symbol(X):
            return X
        if not self._is_adjacent_tail_window(X):
            return X
        try:
            if len(tail.index.intersection(X.index)) > 0:
                return X
            if not (tail.index.is_monotonic_increasing and X.index.is_monotonic_increasing):
                return X
            if not bool(X.index[0] > tail.index[-1]):
                return X
        except (TypeError, ValueError, AttributeError):
            return X
        return pd.concat([tail, X], axis=0)

    def _single_symbol_value(self, frame: pd.DataFrame) -> str | None:
        """Return a single frame symbol if one is explicitly available."""
        pd = load_pandas()
        values = None
        if "symbol" in frame.columns:
            values = frame["symbol"]
        elif "ticker" in frame.columns:
            values = frame["ticker"]
        elif getattr(frame.index, "names", None) and "symbol" in frame.index.names:
            values = pd.Series(frame.index.get_level_values("symbol"), index=frame.index)
        if values is None:
            return None
        unique = pd.Series(values).dropna().astype(str).str.strip().unique()
        if len(unique) != 1 or not unique[0]:
            return None
        return str(unique[0])

    def _same_tail_symbol(self, X: pd.DataFrame) -> bool:
        tail_symbol = self._lookback_tail_symbol_
        current_symbol = self._single_symbol_value(X)
        if tail_symbol is None and current_symbol is None:
            return True
        return tail_symbol is not None and current_symbol == tail_symbol

    def _timestamp_values(self, frame: pd.DataFrame) -> list[Any]:
        pd = load_pandas()
        try:
            if "timestamp" in frame.columns:
                values = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
            elif isinstance(frame.index, pd.DatetimeIndex):
                index = frame.index
                if getattr(index, "tz", None) is None:
                    values = pd.Series(index.tz_localize("UTC"), index=frame.index)
                else:
                    values = pd.Series(index.tz_convert("UTC"), index=frame.index)
            else:
                return []
            values = values.dropna()
            return list(values)
        except (AttributeError, TypeError, ValueError):
            return []

    def _session_values(self, frame: pd.DataFrame) -> list[date]:
        timestamps = self._timestamp_values(frame)
        sessions: list[date] = []
        for ts in timestamps:
            try:
                sessions.append(ts.tz_convert("America/New_York").date())
            except AttributeError:
                sessions.append(ts.astimezone(_ET).date())
            except (TypeError, ValueError):
                return []
        return sessions

    def _is_adjacent_tail_window(self, X: pd.DataFrame) -> bool:
        tail_session = self._lookback_tail_last_session_
        current_sessions = self._session_values(X)
        if tail_session is None or not current_sessions:
            return False
        first_session = current_sessions[0]
        if first_session == tail_session:
            tail_ts = self._lookback_tail_last_ts_
            current_ts_values = self._timestamp_values(X)
            if tail_ts is None or not current_ts_values:
                return False
            try:
                return bool(current_ts_values[0] > tail_ts)
            except (TypeError, ValueError):
                return False
        try:
            return previous_trading_session(first_session) == tail_session
        except (RecursionError, ValueError, TypeError, AttributeError):
            return False

    def _validate_raw_price_columns(self, X: pd.DataFrame) -> None:
        """Reject non-finite or non-positive raw price inputs before feature fill."""
        pd = load_pandas()
        price_columns = [col for col in ("open", "high", "low", "close", "price") if col in X.columns]
        if not price_columns:
            price_columns = [X.columns[0]]
        for column in price_columns:
            values = pd.to_numeric(X[column], errors="coerce")
            finite_positive = np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)) & (values > 0).to_numpy()
            if not bool(finite_positive.all()):
                raise ValueError(f"Raw price column {column!r} must be finite and positive")

    def _add_return_features(self, features: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """Add return-based features."""
        pd = load_pandas()
        try:
            for period in [1, 2, 5, 10]:
                features[f'ret_{period}d'] = prices.pct_change(periods=period)
            features['log_ret_1d'] = np.log(prices / prices.shift(1))
            features['cum_ret_5d'] = prices / prices.shift(5) - 1
            features['cum_ret_20d'] = prices / prices.shift(20) - 1
            features['ret_momentum'] = features['ret_1d'].rolling(5).mean() - features['ret_1d'].rolling(20).mean()
            return features
        except (KeyError, ValueError, TypeError, pd.errors.EmptyDataError) as e:
            logger.error(f'Error adding return features: {e}')
            return features

    def _add_volatility_features(self, features: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """Add volatility-based features."""
        pd = load_pandas()
        try:
            returns = prices.pct_change()
            for window in [5, 10, 20, 50]:
                features[f'vol_{window}d'] = returns.rolling(window).std()
            features[f'vol_ewma_{self.vol_span}'] = returns.ewm(span=self.vol_span).std()
            features['vol_ratio'] = features['vol_5d'] / features['vol_20d']
            vol_rank_window = min(252, len(returns))
            features['vol_rank'] = features['vol_20d'].rolling(vol_rank_window).rank(pct=True)
            if 'high' in features.columns and 'low' in features.columns:
                features['hl_vol'] = np.log(features['high'] / features['low'])
                features['hl_vol_ma'] = features['hl_vol'].rolling(20).mean()
            return features
        except (KeyError, ValueError, TypeError, pd.errors.EmptyDataError) as e:
            logger.error(f'Error adding volatility features: {e}')
            return features

    def _add_volume_features(self, features: pd.DataFrame, volume: pd.Series) -> pd.DataFrame:
        """Add volume-based features."""
        pd = load_pandas()
        try:
            features['vol_ma_10'] = volume.rolling(10).mean()
            features['vol_ma_20'] = volume.rolling(20).mean()
            features['vol_ratio'] = volume / features['vol_ma_20']
            if 'close' in features.columns:
                prices = features['close']
                returns = prices.pct_change()
                obv = np.where(returns > 0, volume, np.where(returns < 0, -volume, 0))
                features['obv'] = pd.Series(obv, index=volume.index).cumsum()
                if 'high' in features.columns and 'low' in features.columns:
                    typical_price = (features['high'] + features['low'] + features['close']) / 3
                    vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
                    features['vwap_dev'] = (typical_price - vwap) / vwap
            return features
        except (KeyError, ValueError, TypeError, pd.errors.EmptyDataError) as e:
            logger.error(f'Error adding volume features: {e}')
            return features

    def _add_regime_features(self, features: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """Add regime detection features."""
        pd = load_pandas()
        try:
            returns = prices.pct_change()
            rolling_vol = returns.rolling(self.regime_span).std()
            vol_low = self.feature_params_.get('regime_vol_low', rolling_vol.quantile(0.33))
            vol_high = self.feature_params_.get('regime_vol_high', rolling_vol.quantile(0.67))
            fallback_vol = returns.std()
            if not np.isfinite(vol_low):
                vol_low = fallback_vol if np.isfinite(fallback_vol) else 0.0
            if not np.isfinite(vol_high):
                vol_high = fallback_vol if np.isfinite(fallback_vol) else 0.0
            features['vol_regime'] = np.where(rolling_vol <= vol_low, 0, np.where(rolling_vol >= vol_high, 2, 1))
            ema_short = prices.ewm(span=12).mean()
            ema_long = prices.ewm(span=26).mean()
            trend_signal = ema_short - ema_long
            features['trend_slope'] = trend_signal.pct_change()
            trend_ma = features['trend_slope'].rolling(20).mean()
            features['trend_regime'] = np.where(trend_ma > 0.001, 1, np.where(trend_ma < -0.001, -1, 0))
            if 'high' in features.columns and 'low' in features.columns:
                hl2 = (features['high'] + features['low']) / 2
                if 'volume' in features.columns:
                    vwap_approx = ((features['high'] + features['low'] + prices) / 3 * features['volume']).rolling(20).sum() / features['volume'].rolling(20).sum()
                    features['microstructure_dev'] = (hl2 - vwap_approx) / vwap_approx
                else:
                    sma20 = prices.rolling(20).mean()
                    features['microstructure_dev'] = (hl2 - sma20) / sma20
            return features
        except (KeyError, ValueError, TypeError, pd.errors.EmptyDataError) as e:
            logger.error(f'Error adding regime features: {e}')
            return features

def create_feature_pipeline(
    scaler_type: str = 'standard',
    build_features_params: dict[str, Any] | None = None,
):
    """
    Create a complete feature engineering pipeline.

    Args:
        scaler_type: Type of scaler ('standard', 'robust', 'none')
        build_features_params: Parameters for BuildFeatures transformer

    Returns:
        sklearn Pipeline with feature engineering and scaling
    """
    pd = load_pandas()
    try:
        if not sklearn_available:
            logger.warning('sklearn not available, returning simple pipeline')
            skl_pipe = load_sklearn_pipeline()
            if skl_pipe is None:
                raise RuntimeError('sklearn.pipeline not available')
            return skl_pipe.Pipeline(
                [('features', BuildFeatures(**(build_features_params or {})))]
            )
        build_features_params = build_features_params or {}
        preproc = load_sklearn_preprocessing()
        skl_pipe = load_sklearn_pipeline()
        if not all([preproc, skl_pipe]):
            raise RuntimeError('Required sklearn modules not available')
        feature_builder = BuildFeatures(**build_features_params)
        if scaler_type == 'standard':
            scaler = preproc.StandardScaler()
        elif scaler_type == 'robust':
            scaler = preproc.RobustScaler()
        elif scaler_type == 'none':
            scaler = None
        else:
            logger.warning(f'Unknown scaler type: {scaler_type}, using standard')
            scaler = preproc.StandardScaler()
        pipeline_steps = [('features', feature_builder)]
        if scaler is not None:
            pipeline_steps.append(('scaler', scaler))
        pipeline = skl_pipe.Pipeline(pipeline_steps)
        logger.info(f'Created feature pipeline with {len(pipeline_steps)} steps')
        return pipeline
    except (KeyError, ValueError, TypeError, pd.errors.EmptyDataError) as e:
        logger.error(f'Error creating feature pipeline: {e}')
        raise

def validate_pipeline_no_leakage(
    pipeline: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series | None = None,
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
    pd = load_pandas()
    try:
        pipeline.fit(X_train, y_train)
        X_train_transformed = pipeline.transform(X_train)
        X_test_transformed = pipeline.transform(X_test)
        if hasattr(X_train_transformed, 'mean'):
            train_mean = X_train_transformed.mean()
            test_mean = X_test_transformed.mean()
            tolerance = 0.1
            if hasattr(train_mean, '__iter__'):
                mean_diff = np.abs(train_mean - test_mean).mean()
            else:
                mean_diff = abs(train_mean - test_mean)
            if mean_diff < tolerance:
                logger.warning('Train and test statistics are very similar - possible leakage')
                return False
        logger.debug('Pipeline validation passed - no obvious leakage detected')
        return True
    except (KeyError, ValueError, TypeError, pd.errors.EmptyDataError) as e:
        logger.error(f'Error validating pipeline: {e}')
        return False
