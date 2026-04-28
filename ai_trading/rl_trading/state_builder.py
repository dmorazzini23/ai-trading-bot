"""State-builder utilities for RL training datasets.

The builder converts raw matrix inputs into feature states and applies optional
train-split normalization so train/eval transformations stay leakage-safe.
"""
from __future__ import annotations
from ai_trading.exception_family import AI_TRADING_FALLBACK_EXCEPTIONS

from dataclasses import asdict, dataclass
from typing import Any, Mapping, cast

try:  # optional dependency
    import numpy as np
except AI_TRADING_FALLBACK_EXCEPTIONS:  # noqa: BLE001 - numpy is optional until state builder is used
    np = None

from ai_trading.logging import get_logger
from ai_trading.utils.lazy_imports import load_pandas

from .features import atr, bollinger_position, obv, rsi, vwap_bias

logger = get_logger(__name__)

STATE_BUILDER_METADATA_VERSION = 1


@dataclass(frozen=True)
class StateBuilderConfig:
    """Configuration for RL state feature construction."""

    use_ohlcv_features: bool = True
    normalize: bool = True
    clip_zscore: float = 6.0
    min_std: float = 1e-06


class MarketStateBuilder:
    """Build and normalize RL state matrices."""

    def __init__(self, config: StateBuilderConfig | None = None) -> None:
        self.config = config or StateBuilderConfig()
        self._schema: str | None = None
        self._input_columns: int | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    @staticmethod
    def _require_numpy() -> None:
        if np is None:
            raise ImportError("numpy is required for MarketStateBuilder")

    @staticmethod
    def _as_matrix(data: Any) -> np.ndarray:
        MarketStateBuilder._require_numpy()
        matrix = np.asarray(data, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("state builder input must be a 2D array")
        if matrix.shape[0] < 4:
            raise ValueError("state builder input requires at least 4 rows")
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _looks_like_ohlcv(matrix: np.ndarray) -> bool:
        if matrix.shape[1] < 5:
            return False
        sample = matrix[:, :4]
        valid = np.isfinite(sample).all(axis=1)
        if int(valid.sum()) < 5:
            return False
        rows = sample[valid]
        open_px = rows[:, 0]
        high_px = rows[:, 1]
        low_px = rows[:, 2]
        close_px = rows[:, 3]
        consistent = (high_px >= np.maximum(open_px, close_px)) & (
            low_px <= np.minimum(open_px, close_px)
        )
        ratio = float(np.mean(consistent))
        return ratio >= 0.8

    @staticmethod
    def _build_ohlcv_features(matrix: np.ndarray) -> np.ndarray:
        pd = load_pandas()
        frame = pd.DataFrame(
            {
                "open": matrix[:, 0],
                "high": matrix[:, 1],
                "low": matrix[:, 2],
                "close": matrix[:, 3],
                "volume": matrix[:, 4],
            }
        )
        close = pd.to_numeric(frame["close"], errors="coerce").ffill().fillna(0.0)
        high = pd.to_numeric(frame["high"], errors="coerce").fillna(close)
        low = pd.to_numeric(frame["low"], errors="coerce").fillna(close)
        volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)

        returns = close.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        rsi_s = rsi(close).fillna(50.0) / 100.0
        atr_s = atr(high, low, close).fillna(0.0)
        vwap_s = vwap_bias(close, high, low, volume).fillna(0.0)
        bb_s = bollinger_position(close).fillna(0.0)
        obv_s = obv(close, volume)
        obv_z = (
            (obv_s - obv_s.rolling(20, min_periods=1).mean())
            / (obv_s.rolling(20, min_periods=1).std() + 1e-08)
        ).fillna(0.0)
        features = np.column_stack(
            [
                returns.to_numpy(dtype=np.float32),
                rsi_s.to_numpy(dtype=np.float32),
                atr_s.to_numpy(dtype=np.float32),
                vwap_s.to_numpy(dtype=np.float32),
                bb_s.to_numpy(dtype=np.float32),
                obv_z.to_numpy(dtype=np.float32),
            ]
        )
        return cast(np.ndarray, np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0))

    def _build_features(
        self,
        matrix: np.ndarray,
        *,
        forced_schema: str | None = None,
    ) -> tuple[np.ndarray, str]:
        schema = forced_schema or "raw"
        if forced_schema not in {None, "raw", "ohlcv"}:
            raise ValueError(f"unsupported state builder schema: {forced_schema}")
        if (
            forced_schema == "ohlcv"
            or (
                forced_schema is None
                and self.config.use_ohlcv_features
                and self._looks_like_ohlcv(matrix)
            )
        ):
            if forced_schema == "ohlcv" and not self._looks_like_ohlcv(matrix):
                raise ValueError("state builder expected OHLCV input")
            return self._build_ohlcv_features(matrix), "ohlcv"
        return matrix, schema

    def _fit_normalizer(self, matrix: np.ndarray) -> None:
        self._require_numpy()
        self._mean = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
        min_std = max(float(self.config.min_std), 1e-08)
        self._std = np.where(std < min_std, min_std, std).astype(np.float32)

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        if not self.config.normalize:
            return matrix.astype(np.float32, copy=False)
        if self._mean is None or self._std is None:
            raise RuntimeError("state builder normalizer is not fitted")
        scaled = (matrix - self._mean) / self._std
        clip = float(self.config.clip_zscore)
        if clip > 0:
            scaled = np.clip(scaled, -clip, clip)
        return cast(np.ndarray, scaled.astype(np.float32, copy=False))

    def fit_transform(self, data: Any) -> np.ndarray:
        matrix = self._as_matrix(data)
        features, schema = self._build_features(matrix, forced_schema=None)
        self._schema = schema
        self._input_columns = int(matrix.shape[1])
        self._fit_normalizer(features)
        return self._normalize(features)

    def transform(self, data: Any) -> np.ndarray:
        matrix = self._as_matrix(data)
        if self._input_columns is not None and int(matrix.shape[1]) != self._input_columns:
            raise ValueError(
                "state builder input feature mismatch: "
                f"expected {self._input_columns} columns, got {matrix.shape[1]}"
            )
        features, _ = self._build_features(matrix, forced_schema=self._schema)
        if self._mean is not None and int(features.shape[1]) != int(self._mean.shape[0]):
            raise ValueError(
                "state builder output feature mismatch: "
                f"expected {self._mean.shape[0]} columns, got {features.shape[1]}"
            )
        return self._normalize(features)

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "metadata_version": STATE_BUILDER_METADATA_VERSION,
            "schema": self._schema or "unknown",
            "normalize": bool(self.config.normalize),
            "clip_zscore": float(self.config.clip_zscore),
            "min_std": float(self.config.min_std),
            "input_columns": int(self._input_columns) if self._input_columns is not None else 0,
            "fitted": bool(self._mean is not None and self._std is not None),
            "feature_count": int(self._mean.shape[0]) if self._mean is not None else 0,
        }

    def to_metadata(self) -> dict[str, Any]:
        """Return fitted transform metadata needed for inference parity."""

        if self._schema is None or self._mean is None or self._std is None:
            raise RuntimeError("state builder metadata requires a fitted builder")
        metadata = self.describe()
        metadata["config"] = asdict(self.config)
        metadata["mean"] = [float(value) for value in self._mean.tolist()]
        metadata["std"] = [float(value) for value in self._std.tolist()]
        return metadata

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "MarketStateBuilder":
        """Rehydrate a fitted builder from training metadata."""

        cls._require_numpy()
        if not isinstance(metadata, Mapping):
            raise ValueError("state builder metadata must be a mapping")
        if not bool(metadata.get("enabled", True)):
            raise ValueError("state builder metadata is disabled")
        schema = str(metadata.get("schema") or "").strip()
        if schema not in {"raw", "ohlcv"}:
            raise ValueError(f"unsupported state builder schema: {schema or 'missing'}")
        config_raw = metadata.get("config")
        if isinstance(config_raw, Mapping):
            config = StateBuilderConfig(**dict(config_raw))
        else:
            config = StateBuilderConfig(
                use_ohlcv_features=(schema == "ohlcv"),
                normalize=bool(metadata.get("normalize", True)),
                clip_zscore=float(metadata.get("clip_zscore", 6.0)),
                min_std=float(metadata.get("min_std", 1e-06)),
            )
        mean_raw = metadata.get("mean")
        std_raw = metadata.get("std")
        if mean_raw is None or std_raw is None:
            raise ValueError("state builder metadata missing normalizer stats")
        mean = np.asarray(mean_raw, dtype=np.float32).reshape(-1)
        std = np.asarray(std_raw, dtype=np.float32).reshape(-1)
        if mean.size == 0 or mean.shape != std.shape:
            raise ValueError("state builder normalizer stats shape mismatch")
        if not bool(np.isfinite(mean).all()) or not bool(np.isfinite(std).all()):
            raise ValueError("state builder normalizer stats must be finite")
        min_std = max(float(config.min_std), 1e-08)
        std = np.where(std < min_std, min_std, std).astype(np.float32)

        expected_features = int(metadata.get("feature_count") or mean.shape[0])
        if expected_features != int(mean.shape[0]):
            raise ValueError(
                "state builder feature_count mismatch: "
                f"expected {expected_features}, got {mean.shape[0]}"
            )
        builder = cls(config)
        builder._schema = schema
        input_columns = int(metadata.get("input_columns") or 0)
        builder._input_columns = input_columns if input_columns > 0 else None
        builder._mean = mean
        builder._std = std
        return builder


__all__ = ["MarketStateBuilder", "StateBuilderConfig"]
