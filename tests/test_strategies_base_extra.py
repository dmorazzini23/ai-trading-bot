import pytest

base_mod = pytest.importorskip("ai_trading.strategies.base")
Strategy = getattr(base_mod, "Strategy", None)
asset_class_for = getattr(base_mod, "asset_class_for", None)
if Strategy is None or asset_class_for is None:
    pytest.skip("strategy base utilities unavailable", allow_module_level=True)


def test_asset_class_for_crypto():
    """Symbols starting with crypto prefixes are labelled crypto."""
    assert asset_class_for("ETHBTC") == "crypto"


def test_strategy_generate_base():
    """Base Strategy.generate returns empty list."""
    assert Strategy().generate(None) == []
