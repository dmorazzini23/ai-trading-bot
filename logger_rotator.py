"""Utility wrapper exposing logger.get_rotating_handler for tests."""
from logging import Handler

from logger import get_rotating_handler

__all__ = ["get_rotating_handler"]

