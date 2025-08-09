"""
DEPRECATED: This module is deprecated.
Use ai_trading.logging instead to avoid duplicate logging setup.
"""

import warnings
import logging
from typing import Dict, Any

# Issue deprecation warning
warnings.warn(
    "logger.py is deprecated. Use ai_trading.logging instead to prevent duplicate logging setup.",
    DeprecationWarning,
    stacklevel=2
)

# Delegate all functionality to ai_trading.logging
try:
    from ai_trading.logging import *
    from ai_trading.logging import (
        setup_logging,
        get_logger,
        init_logger,
        log_performance_metrics,
        log_trading_event,
        setup_enhanced_logging,
        get_rotating_handler,
    )
    # Create logger using centralized system
    logger = logging.getLogger(__name__)
    
except ImportError as e:
    # Fallback minimal implementations if ai_trading.logging not available
    warnings.warn(f"Failed to import ai_trading.logging: {e}. Using minimal fallback.", UserWarning)
    
    def setup_logging(debug: bool = False, log_file: str = None) -> logging.Logger:
        """Minimal fallback logging setup."""
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
            root_logger.addHandler(handler)
            root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
        return root_logger
    
    def get_logger(name: str) -> logging.Logger:
        """Minimal fallback logger getter."""
        setup_logging()  # Ensure basic setup
        return logging.getLogger(name)
    
    def init_logger(log_file: str) -> logging.Logger:
        """Minimal fallback logger initializer."""
        return setup_logging(log_file=log_file)
    
    def log_performance_metrics(*args, **kwargs):
        """Minimal fallback for performance metrics."""
        pass
    
    def log_trading_event(*args, **kwargs):
        """Minimal fallback for trading events.""" 
        pass
    
    def setup_enhanced_logging(*args, **kwargs):
        """Minimal fallback for enhanced logging."""
        return setup_logging()
    
    def get_rotating_handler(path: str, max_bytes: int = 5_000_000, backup_count: int = 5):
        """Minimal fallback for rotating handler."""
        import os
        import sys
        from logging.handlers import RotatingFileHandler
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
        except OSError:
            return logging.StreamHandler(sys.stderr)
    
    logger = logging.getLogger(__name__)




