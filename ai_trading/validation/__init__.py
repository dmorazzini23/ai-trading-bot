"""Validation utilities and facades."""

from .require_env import (
    _require_env_vars,
    require_env_vars,
    should_halt_trading,
)
from .validate_env import debug_environment, validate_specific_env_var


def check_data_freshness(*args, **kwargs):
    """Lazy proxy to :func:`ai_trading.data_validation.check_data_freshness`.

    Importing :mod:`ai_trading.data_validation` can pull in heavy dependencies
    like pandas, so this function defers that import until it is actually
    needed.
    """

    from ai_trading.data_validation import check_data_freshness as _check

    return _check(*args, **kwargs)


__all__ = [
    "check_data_freshness",
    "require_env_vars",
    "_require_env_vars",
    "should_halt_trading",
    "debug_environment",
    "validate_specific_env_var",
]
