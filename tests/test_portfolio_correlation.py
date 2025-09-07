"""Test portfolio correlation utilities."""

from ai_trading.portfolio.correlation import calculate_correlation_matrix


def test_calculate_correlation_matrix_three_symbols():
    """Ensure all symbols in the portfolio are correlated against each other."""
    returns = {
        "AAPL": [0.01, 0.02, 0.03],
        "MSFT": [0.02, 0.04, 0.06],
        "GOOGL": [0.03, 0.06, 0.09],
    }
    matrix = calculate_correlation_matrix(returns)
    assert set(matrix.keys()) == {"AAPL", "MSFT", "GOOGL"}
    assert set(matrix["AAPL"].keys()) == {"MSFT", "GOOGL"}
    assert set(matrix["MSFT"].keys()) == {"AAPL", "GOOGL"}
    assert set(matrix["GOOGL"].keys()) == {"AAPL", "MSFT"}
