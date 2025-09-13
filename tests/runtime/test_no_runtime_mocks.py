import ai_trading as pkg


def test_no_mock_exports():
    assert not any(name.startswith('Mock') for name in dir(pkg))
