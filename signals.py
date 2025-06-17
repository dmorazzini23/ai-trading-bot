"""Simple signal generation module for tests."""

import importlib
import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)


def load_module(name: str) -> Any:
    """Dynamically import a module using :mod:`importlib`."""
    try:
        return importlib.import_module(name)
    except Exception as exc:  # pragma: no cover - dynamic import may fail
        logger.warning("Failed to import %s: %s", name, exc)
        return None


def _fetch_api(url: str, retries: int = 3, delay: float = 1.0) -> dict:
    """Fetch JSON from an API with simple retry logic."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # pragma: no cover - network may be mocked
            logger.warning(
                "API request failed (%s/%s): %s", attempt, retries, exc
            )
            time.sleep(delay)
    return {}


def generate(ctx=None):
    return 0
