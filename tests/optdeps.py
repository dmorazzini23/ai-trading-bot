import pytest

HINT = {
    "pandas": 'pip install "ai-trading-bot[pandas]"',
    "matplotlib": 'pip install "ai-trading-bot[plot]"',
    "sklearn": 'pip install "ai-trading-bot[ml]"',
    "torch": 'pip install "ai-trading-bot[ml]"',
    "ta": 'pip install "ai-trading-bot[ta]"',
    "talib": 'pip install "ai-trading-bot[ta]"',
    "stable_baselines3": 'pip install "stable-baselines3 gymnasium torch"',
    "gymnasium": 'pip install "stable-baselines3 gymnasium torch"',
    "alpaca": 'pip install "alpaca-py"',
    "alpaca_api": 'pip install "ai-trading-bot"',
    "tenacity": 'pip install "tenacity"',
    "pandas_market_calendars": 'pip install "pandas-market-calendars"',
    "pydantic": 'pip install "pydantic"',
    "requests": 'pip install "requests"',
    "sqlalchemy": 'pip install "SQLAlchemy>=2.0,<3"',
    "psycopg": 'pip install "psycopg[binary]>=3.2,<4"',
    "hypothesis": 'pip install "hypothesis>=6.92"',
    "freezegun": 'pip install "freezegun>=1.5"',
    "psutil": 'pip install "psutil>=5.9,<6"',
    "pytest_asyncio": 'pip install "pytest-asyncio>=0.23"',
    "hmmlearn": 'pip install "hmmlearn==0.3.2"',
}

OPTDEPS = HINT  # AI-AGENT-REF: compatibility alias


def require(pkg: str):
    """Skip current test module unless `pkg` is importable, with a clear install hint."""
    return pytest.importorskip(pkg, reason=f"Install with: {HINT.get(pkg, 'pip install ' + pkg)}")
