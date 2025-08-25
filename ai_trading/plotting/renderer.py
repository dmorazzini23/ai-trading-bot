"""Basic plotting helpers."""
from __future__ import annotations
import logging
from ai_trading.utils.optdeps import OptionalDependencyError

logger = logging.getLogger(__name__)

def render_equity_curve(series, *, title: str = "Equity") -> None:
    """Render a simple equity curve using matplotlib if available."""
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        logger.info(
            'MATPLOTLIB_MISSING', extra={'hint': 'pip install "ai-trading-bot[plot]"'}
        )
        raise OptionalDependencyError(
            "matplotlib", feature="plotting", extra="plot"
        )
    plt.figure()
    plt.plot(series)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Equity")
    plt.close()
