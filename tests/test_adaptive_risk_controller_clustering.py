"""Tests for adaptive risk controller clustering imports and fallbacks."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
import builtins
import importlib.util
import sys


try:  # pragma: no cover - optional dependency for production
    import numpy  # type: ignore  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - stub fallback for tests
    numpy_stub = ModuleType("numpy")

    def _sqrt(value):
        return value ** 0.5

    numpy_stub.sqrt = _sqrt  # type: ignore[attr-defined]
    sys.modules["numpy"] = numpy_stub

sys.modules.setdefault("pandas", ModuleType("pandas"))

import ai_trading  # noqa: F401 - ensure base package loaded

portfolio_stub = ModuleType("ai_trading.portfolio")
portfolio_stub.__path__ = [
    str(Path(__file__).resolve().parent.parent / "ai_trading" / "portfolio"),
]
sys.modules.setdefault("ai_trading.portfolio", portfolio_stub)

spec = importlib.util.spec_from_file_location(
    "ai_trading.portfolio.risk_controls",
    Path(__file__).resolve().parent.parent / "ai_trading" / "portfolio" / "risk_controls.py",
)
risk_controls = importlib.util.module_from_spec(spec)
sys.modules["ai_trading.portfolio.risk_controls"] = risk_controls
assert spec.loader is not None
spec.loader.exec_module(risk_controls)
portfolio_stub.risk_controls = risk_controls


class FakeCorrelationMatrix:
    """Minimal stand-in for a pandas correlation matrix."""

    def __init__(self, symbols: list[str]):
        self._symbols = symbols

    def fillna(self, _value):
        return self

    def abs(self):
        return self

    def __len__(self) -> int:
        return len(self._symbols)

    @property
    def index(self) -> list[str]:
        return self._symbols

    def __rsub__(self, _other):
        return self


class FakeReturnsData:
    """Simplified returns data structure exposing DataFrame-like APIs."""

    def __init__(self, symbols: list[str]):
        self._symbols = symbols

    @property
    def columns(self) -> list[str]:
        return self._symbols

    def tail(self, _n: int):
        return self

    def corr(self):
        return FakeCorrelationMatrix(self._symbols)


def _install_fake_scipy(monkeypatch, *, fcluster, linkage, squareform) -> None:
    """Install lightweight SciPy stubs for clustering imports."""

    scipy_pkg = ModuleType("scipy")
    scipy_pkg.__path__ = []  # mark as package

    cluster_pkg = ModuleType("scipy.cluster")
    cluster_pkg.__path__ = []
    hierarchy_pkg = ModuleType("scipy.cluster.hierarchy")
    hierarchy_pkg.fcluster = fcluster
    hierarchy_pkg.linkage = linkage
    cluster_pkg.hierarchy = hierarchy_pkg

    spatial_pkg = ModuleType("scipy.spatial")
    spatial_pkg.__path__ = []
    distance_pkg = ModuleType("scipy.spatial.distance")
    distance_pkg.squareform = squareform
    spatial_pkg.distance = distance_pkg

    scipy_pkg.cluster = cluster_pkg
    scipy_pkg.spatial = spatial_pkg

    monkeypatch.setitem(sys.modules, "scipy", scipy_pkg)
    monkeypatch.setitem(sys.modules, "scipy.cluster", cluster_pkg)
    monkeypatch.setitem(sys.modules, "scipy.cluster.hierarchy", hierarchy_pkg)
    monkeypatch.setitem(sys.modules, "scipy.spatial", spatial_pkg)
    monkeypatch.setitem(sys.modules, "scipy.spatial.distance", distance_pkg)


def test_import_clustering_success(monkeypatch):
    """_import_clustering should return SciPy callables when present."""

    with monkeypatch.context() as m:
        m.setattr(
            "ai_trading.config.get_settings",
            lambda: SimpleNamespace(ENABLE_PORTFOLIO_FEATURES=True),
        )

        def fake_fcluster(*_args, **_kwargs):
            return "fcluster"

        def fake_linkage(*_args, **_kwargs):
            return "linkage"

        def fake_squareform(*_args, **_kwargs):
            return "squareform"

        _install_fake_scipy(m, fcluster=fake_fcluster, linkage=fake_linkage, squareform=fake_squareform)

        fcluster, linkage, squareform, available = risk_controls._import_clustering()
        assert available is True
        assert fcluster is fake_fcluster
        assert linkage is fake_linkage
        assert squareform is fake_squareform


def test_import_clustering_missing_dependency(monkeypatch):
    """_import_clustering should signal unavailable when SciPy is missing."""

    with monkeypatch.context() as m:
        m.setattr(
            "ai_trading.config.get_settings",
            lambda: SimpleNamespace(ENABLE_PORTFOLIO_FEATURES=True),
        )
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("scipy"):
                raise ImportError("SciPy not installed")
            return real_import(name, globals, locals, fromlist, level)

        m.setattr(builtins, "__import__", fake_import)
        result = risk_controls._import_clustering()
        assert result == (None, None, None, False)


def test_calculate_correlation_clusters_with_clustering(monkeypatch):
    """AdaptiveRiskController should use clustering assignments when available."""

    controller = risk_controls.AdaptiveRiskController()

    symbols = [f"S{i}" for i in range(6)]
    returns_data = FakeReturnsData(symbols)

    def fake_squareform(_matrix, checks=False):  # noqa: ARG001 - signature parity
        return [0.0]

    def fake_linkage(_distances, method="ward"):  # noqa: ARG001
        return "fake-linkage"

    def fake_fcluster(_linkage, _threshold, criterion="maxclust"):  # noqa: ARG001
        return [1, 1, 1, 2, 2, 2]

    monkeypatch.setattr(
        risk_controls,
        "_import_clustering",
        lambda: (fake_fcluster, fake_linkage, fake_squareform, True),
    )

    clusters = controller.calculate_correlation_clusters(returns_data, max_clusters=3)
    assert clusters[symbols[0]] == 0
    assert clusters[symbols[-1]] == 1


def test_calculate_correlation_clusters_without_clustering(monkeypatch):
    """AdaptiveRiskController should fall back to a single cluster when unavailable."""

    controller = risk_controls.AdaptiveRiskController()

    symbols = ["AAA", "BBB", "CCC"]
    returns_data = FakeReturnsData(symbols)

    monkeypatch.setattr(
        risk_controls,
        "_import_clustering",
        lambda: (None, None, None, False),
    )

    clusters = controller.calculate_correlation_clusters(returns_data)
    assert set(clusters.values()) == {0}
