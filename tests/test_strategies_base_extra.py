from strategies.base import Strategy, asset_class_for


def test_asset_class_for_crypto():
    """Symbols starting with crypto prefixes are labelled crypto."""
    assert asset_class_for("ETHBTC") == "crypto"


def test_strategy_generate_base():
    """Base Strategy.generate returns empty list."""
    assert Strategy().generate(None) == []
