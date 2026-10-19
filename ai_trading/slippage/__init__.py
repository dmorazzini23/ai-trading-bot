"""Utilities for recording trade slippage."""
__all__ = ["SLIPPAGE_LOG_COLUMNS", "ensure_slippage_log_header", "log_slippage"]
from .recorder import SLIPPAGE_LOG_COLUMNS, ensure_slippage_log_header, log_slippage

