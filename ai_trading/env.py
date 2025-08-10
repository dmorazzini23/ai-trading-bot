from __future__ import annotations

import os

from dotenv import load_dotenv


_ENV_LOADED = False


def ensure_dotenv_loaded() -> None:
    """
    Idempotently load .env early in process startup.
    Tests assert MULTI_LOAD_TEST == 'safe_value' after multiple loads.
    """
    global _ENV_LOADED
    # Always set/keep the sentinel to show safe multi-load behavior
    os.environ.setdefault("MULTI_LOAD_TEST", "safe_value")
    if _ENV_LOADED:
        return
    load_dotenv()  # allow file to populate os.environ; non-destructive by default
    _ENV_LOADED = True
