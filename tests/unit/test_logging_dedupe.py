import pytest
from tests.optdeps import require

log_mod = require("ai_trading.logging")

@pytest.mark.unit
def test_phase_logger_no_propagation():
    get_phase_logger = getattr(log_mod, "get_phase_logger", None)
    if get_phase_logger is None:
        pytest.skip("get_phase_logger not available")
    logger = get_phase_logger("test_module", "TEST")
    assert hasattr(logger, "propagate")
    assert logger.propagate is False
