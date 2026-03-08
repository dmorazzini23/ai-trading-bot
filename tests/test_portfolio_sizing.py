from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Iterator, cast

import numpy as np
import pandas as pd
import pytest

from ai_trading.portfolio import sizing


@pytest.fixture(autouse=True)
def _reset_equity_cache() -> Iterator[None]:
    sizing._equity_cache.ts = None
    sizing._equity_cache.equity = None
    yield
    sizing._equity_cache.ts = None
    sizing._equity_cache.equity = None


def _series(start: float, step: float, n: int = 80) -> pd.Series:
    return pd.Series([start + step * i for i in range(n)], dtype=float)


def test_fetch_equity_uses_cache_within_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    sizing._equity_cache.ts = now - timedelta(seconds=30)
    sizing._equity_cache.equity = 4321.0
    monkeypatch.setattr(sizing, "_now", lambda: now)

    class _Api:
        def get_account(self):  # pragma: no cover - should not be called
            raise AssertionError("cache should have been used")

    ctx = SimpleNamespace(api=_Api())
    assert sizing._fetch_equity(ctx, ttl_seconds=60) == 4321.0


def test_fetch_equity_refreshes_and_sets_paper_flag() -> None:
    class _Api:
        def get_account(self):
            return SimpleNamespace(equity="123.45")

    api = _Api()
    ctx = SimpleNamespace(api=api, alpaca_base_url="https://paper-api.alpaca.markets")
    equity = sizing._fetch_equity(ctx, force_refresh=True)

    assert equity == pytest.approx(123.45)
    assert getattr(api, "paper") is True


def test_fetch_equity_error_returns_zero_and_updates_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    monkeypatch.setattr(sizing, "_now", lambda: now)

    class _Api:
        def get_account(self):
            raise TypeError("boom")

    ctx = SimpleNamespace(api=_Api())
    equity = sizing._fetch_equity(ctx, force_refresh=True)

    assert equity == 0.0
    assert sizing._equity_cache.equity == 0.0
    assert sizing._equity_cache.ts == now


def test_volatility_targeting_applies_limits_and_normalizes() -> None:
    sizer = sizing.VolatilityTargetingSizer(min_weight=0.05, max_weight=0.6)
    adjusted = sizer._apply_position_limits({"A": 0.01, "B": 0.5, "C": 0.8})

    assert "A" not in adjusted
    assert adjusted["B"] == pytest.approx(0.45454545)
    assert adjusted["C"] == pytest.approx(0.54545454)
    assert sum(adjusted.values()) == pytest.approx(1.0)


def test_turnover_penalty_dampens_to_configured_limit() -> None:
    sizer = sizing.TurnoverPenaltySizer(max_turnover=0.25)
    current = {"A": 1.0}
    proposed = {"B": 1.0}

    adjusted = sizer.apply_turnover_penalty(proposed, current)

    assert adjusted["A"] == pytest.approx(0.75)
    assert adjusted["B"] == pytest.approx(0.25)
    assert sizer._calculate_turnover(adjusted, current) <= 0.250000001


def test_turnover_penalty_keeps_weights_when_already_within_limit() -> None:
    sizer = sizing.TurnoverPenaltySizer(max_turnover=0.5)
    current = {"A": 0.6, "B": 0.4}
    proposed = {"A": 0.55, "B": 0.45}

    adjusted = sizer.apply_turnover_penalty(proposed, current)

    assert adjusted == proposed
    assert len(sizer.position_history) == 1


def test_get_historical_turnover_returns_series() -> None:
    sizer = sizing.TurnoverPenaltySizer()
    sizer._update_position_history({"A": 1.0})
    sizer._update_position_history({"A": 0.5, "B": 0.5})
    sizer._update_position_history({"A": 0.25, "B": 0.75})

    turnover = sizer.get_historical_turnover()
    assert turnover == [0.5, 0.25]


def test_import_clustering_disabled_when_portfolio_features_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ai_trading.config.get_settings",
        lambda: SimpleNamespace(ENABLE_PORTFOLIO_FEATURES=False),
    )
    assert sizing._import_clustering() == (None, None, None, False)


def test_volatility_targeting_calculate_position_sizes_end_to_end() -> None:
    sizer = sizing.VolatilityTargetingSizer(min_weight=0.01, max_weight=0.9)
    signals = {"AAA": 1.0, "BBB": 0.8, "MISSING": 0.5}
    price_history = {
        "AAA": _series(100.0, 0.4),
        "BBB": _series(60.0, -0.15) + pd.Series(np.sin(np.arange(80)) * 0.3),
        "MISSING": _series(40.0, 0.2),
    }
    current_prices = {"AAA": 132.0, "BBB": 48.0}

    result = sizer.calculate_position_sizes(
        signals=signals,
        current_prices=current_prices,
        portfolio_value=25_000,
        price_history=price_history,
    )

    assert set(result.keys()) == {"AAA", "BBB"}
    assert all(details["shares"] >= 1 for details in result.values())
    assert all("estimated_vol" in details for details in result.values())


def test_volatility_targeting_calculate_position_sizes_safe_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizer = sizing.VolatilityTargetingSizer()

    def _boom(_symbols: list[str]) -> dict[str, float]:
        raise ValueError("boom")

    monkeypatch.setattr(sizer, "_estimate_volatilities", _boom)
    result = sizer.calculate_position_sizes(
        signals={"AAA": 1.0},
        current_prices={"AAA": 100.0},
        portfolio_value=1_000.0,
    )
    assert result == {}


def test_estimate_volatilities_short_history_and_cached_default() -> None:
    sizer = sizing.VolatilityTargetingSizer()
    sizer.volatility_estimates["BBB"] = 0.33
    sizer.price_history = {
        "AAA": pd.Series([1.0, 1.1, 1.2], dtype=float),
        "DDD": _series(50.0, 0.8),
    }

    vol = sizer._estimate_volatilities(["AAA", "BBB", "CCC", "DDD"])
    assert vol["AAA"] == pytest.approx(0.2)
    assert vol["BBB"] == pytest.approx(0.33)
    assert vol["CCC"] == pytest.approx(0.2)
    assert vol["DDD"] >= 0.05


def test_estimate_volatilities_error_returns_defaults() -> None:
    sizer = sizing.VolatilityTargetingSizer()
    sizer.price_history = {"BROKEN": pd.Series(["a", "b", "c", "d", "e", "f"])}
    vol = sizer._estimate_volatilities(["BROKEN"])
    assert vol == {"BROKEN": 0.2}


def test_estimate_correlations_branches() -> None:
    sizer = sizing.VolatilityTargetingSizer()
    assert np.array_equal(sizer._estimate_correlations(["AAA"]), np.array([[1.0]]))

    sizer.price_history = {"AAA": _series(10.0, 0.2)}
    fallback = sizer._estimate_correlations(["AAA", "BBB"])
    assert fallback.shape == (2, 2)
    assert fallback[0, 1] == pytest.approx(0.2)

    sizer.price_history = {
        "AAA": _series(100.0, 0.4),
        "BBB": _series(60.0, -0.15) + pd.Series(np.sin(np.arange(80)) * 0.3),
    }
    matrix = sizer._estimate_correlations(["AAA", "BBB"])
    assert matrix.shape == (2, 2)
    assert np.all(np.diag(matrix) > 0.99)


def test_scale_to_target_volatility_and_error_fallback() -> None:
    sizer = sizing.VolatilityTargetingSizer(max_weight=1.0)
    weights = {"AAA": 0.5, "BBB": 0.5}
    volatilities = {"AAA": 0.3, "BBB": 0.2}
    corr = np.array([[1.0, 0.2], [0.2, 1.0]])

    scaled = sizer._scale_to_target_volatility(weights, volatilities, corr)
    assert set(scaled) == {"AAA", "BBB"}
    assert sum(scaled.values()) == pytest.approx(1.0)

    zero = sizer._scale_to_target_volatility(
        weights,
        {"AAA": 0.0, "BBB": 0.0},
        np.eye(2),
    )
    assert zero == weights

    bad = sizer._scale_to_target_volatility(weights, volatilities, np.array([[1.0]]))
    assert bad == weights


def test_calculate_inverse_vol_weights_error_fallback() -> None:
    sizer = sizing.VolatilityTargetingSizer()
    result = sizer._calculate_inverse_vol_weights(
        signals=cast(dict[str, float], {"AAA": "bad", "BBB": "also_bad"}),
        volatilities={"AAA": 0.2, "BBB": 0.3},
    )
    assert result == {"AAA": 0.5, "BBB": 0.5}


def test_risk_parity_weight_paths_and_sign_handling() -> None:
    sizer = sizing.RiskParitySizer(max_weight=0.9)
    price_history = {
        "AAA": _series(100.0, 0.3),
        "BBB": _series(70.0, -0.1) + pd.Series(np.sin(np.arange(80)) * 0.2),
    }

    weights = sizer.calculate_risk_parity_weights(
        signals={"AAA": 1.0, "BBB": -0.5},
        price_history=price_history,
    )
    assert set(weights) == {"AAA", "BBB"}
    assert weights["AAA"] > 0
    assert weights["BBB"] < 0

    single = sizer.calculate_risk_parity_weights(
        signals={"AAA": 1.0},
        price_history=price_history,
    )
    assert single == {"AAA": 1.0}

    equal = sizer.calculate_risk_parity_weights(
        signals={"AAA": 1.0, "BBB": 1.0},
        price_history={"AAA": pd.Series([1.0, 1.1]), "BBB": pd.Series([2.0, 2.1])},
    )
    assert equal == {"AAA": 0.5, "BBB": 0.5}


def test_risk_parity_covariance_and_optimizer_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizer = sizing.RiskParitySizer(max_weight=0.9)
    history = {
        "AAA": _series(100.0, 0.3),
        "BBB": _series(70.0, -0.1) + pd.Series(np.sin(np.arange(80)) * 0.2),
    }
    cov = sizer._calculate_covariance_matrix(["AAA", "BBB"], history)
    assert cov is not None
    assert cov.shape == (2, 2)

    none_cov = sizer._calculate_covariance_matrix(["AAA", "BBB"], {"AAA": _series(1.0, 0.1, 5)})
    assert none_cov is None

    optimized = sizer._optimize_risk_parity(np.array([[0.04, 0.01], [0.01, 0.09]]), ["AAA", "BBB"])
    assert set(optimized) == {"AAA", "BBB"}
    assert sum(optimized.values()) == pytest.approx(1.0)

    fallback = sizer._optimize_risk_parity(np.array([[1.0]]), ["AAA", "BBB"])
    assert fallback == {"AAA": 0.5, "BBB": 0.5}

    def _raise(*_args, **_kwargs):
        raise ValueError("bad_cov")

    monkeypatch.setattr(sizer, "_calculate_covariance_matrix", _raise)
    safe = sizer.calculate_risk_parity_weights(
        signals={"AAA": 1.0, "BBB": 1.0},
        price_history=history,
    )
    assert safe == {"AAA": 0.5, "BBB": 0.5}


def test_cluster_sizer_apply_limits_and_matrix_paths() -> None:
    sizer = sizing.CorrelationClusterSizer(max_cluster_weight=0.5)
    base = {"AAA": 0.5, "BBB": 0.4, "CCC": 0.1}

    assert sizer.apply_cluster_limits({"AAA": 0.6, "BBB": 0.4}, {}) == {"AAA": 0.6, "BBB": 0.4}
    assert sizer.apply_cluster_limits(base, {"AAA": _series(1.0, 0.1, 5)}) == base

    matrix = sizer._calculate_correlation_matrix(
        ["AAA", "BBB"],
        {
            "AAA": _series(100.0, 0.2),
            "BBB": _series(90.0, -0.1) + pd.Series(np.sin(np.arange(80)) * 0.25),
        },
    )
    assert matrix is not None
    assert matrix.shape == (2, 2)

    assert sizer._calculate_correlation_matrix(["AAA", "BBB"], {"AAA": _series(1.0, 0.1, 5)}) is None


def test_cluster_sizer_perform_clustering_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    sizer = sizing.CorrelationClusterSizer()
    corr = np.array([[1.0, 0.9, 0.3], [0.9, 1.0, 0.2], [0.3, 0.2, 1.0]])

    monkeypatch.setattr(sizing, "_import_clustering", lambda: (None, None, None, False))
    fallback = sizer._perform_clustering(["AAA", "BBB", "CCC"], corr)
    assert sum(len(v) for v in fallback.values()) == 3

    def _fake_squareform(matrix: np.ndarray) -> np.ndarray:
        return matrix

    def _fake_linkage(_distance: np.ndarray, method: str) -> np.ndarray:
        assert method == "ward"
        return np.array([[0.0, 1.0, 0.5, 2.0], [2.0, 3.0, 0.7, 3.0]])

    def _fake_fcluster(
        _linkage: np.ndarray,
        _distance_threshold: float,
        criterion: str,
    ) -> np.ndarray:
        assert criterion == "distance"
        return np.array([1, 1, 2])

    monkeypatch.setattr(
        sizing,
        "_import_clustering",
        lambda: (_fake_fcluster, _fake_linkage, _fake_squareform, True),
    )
    clustered = sizer._perform_clustering(["AAA", "BBB", "CCC"], corr)
    assert clustered == {1: ["AAA", "BBB"], 2: ["CCC"]}

    def _raise_import(*_args, **_kwargs):
        raise ImportError("no_scipy")

    monkeypatch.setattr(sizing, "_import_clustering", _raise_import)
    error_fallback = sizer._perform_clustering(["AAA", "BBB"], np.eye(2))
    assert error_fallback == {0: ["AAA"], 1: ["BBB"]}


def test_cluster_constraints_error_fallback_and_scaling() -> None:
    sizer = sizing.CorrelationClusterSizer(max_cluster_weight=0.5)
    constrained = sizer._apply_cluster_constraints(
        {"AAA": 0.5, "BBB": 0.4, "CCC": 0.1},
        {1: ["AAA", "BBB"], 2: ["CCC"]},
    )
    assert sum(constrained.values()) == pytest.approx(1.0)
    assert constrained["CCC"] > 0.1

    bad = sizer._apply_cluster_constraints(cast(dict[str, float], {"AAA": "bad"}), {1: ["AAA"]})
    assert bad == {"AAA": "bad"}


def test_turnover_error_paths_and_history_trim() -> None:
    sizer = sizing.TurnoverPenaltySizer(lookback_periods=2)
    assert sizer._calculate_turnover({"AAA": 1.0}, cast(dict[str, float], {"AAA": "bad"})) == 0.0

    reduced = sizer._reduce_turnover(cast(dict[str, float], {"AAA": "bad"}), {"AAA": 0.5}, 1.0)
    assert reduced == {"AAA": "bad"}

    sizer._update_position_history({"AAA": 1.0})
    sizer._update_position_history({"AAA": 0.5, "BBB": 0.5})
    sizer._update_position_history({"BBB": 1.0})
    assert len(sizer.position_history) == 2

    sizer.position_history = [{"weights": {"AAA": 1.0}}, {"bad": {}}]
    assert sizer.get_historical_turnover() == []
