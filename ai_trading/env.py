from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv

_ENV_LOADED = False


def ensure_dotenv_loaded() -> None:
    """
    Idempotently load .env early in process startup **without** assuming it's in the repo.

    Search order (first hit wins, non-destructive):
      1) $DOTENV_PATH (if set)
      2) autodetect via find_dotenv(usecwd=True)
      3) CWD/.env
      4) repo root (two levels up from this file)/.env
      5) /etc/default/ai-trading  (common systemd EnvironmentFile style)
      6) /etc/ai-trading.env
      7) $HOME/.config/ai-trading/.env
    Emits a once-per-process info line indicating which path (if any) was used.
    """  # AI-AGENT-REF: search common droplet paths
    global _ENV_LOADED
    os.environ.setdefault("MULTI_LOAD_TEST", "safe_value")
    if _ENV_LOADED:
        return

    candidates: list[str | None] = []
    candidates.append(os.environ.get("DOTENV_PATH"))
    auto = find_dotenv(usecwd=True)
    if auto:
        candidates.append(auto)
    candidates.append(os.path.join(os.getcwd(), ".env"))
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    candidates.append(os.path.join(repo_root, ".env"))
    candidates.extend([
        "/etc/default/ai-trading",
        "/etc/ai-trading.env",
        os.path.expanduser("~/.config/ai-trading/.env"),
    ])

    loaded_from: str | None = None
    for path in candidates:
        if not path:
            continue
        try:
            if os.path.exists(path):
                if load_dotenv(dotenv_path=path, override=False):
                    loaded_from = path
                    break
        except Exception:
            continue

    if not loaded_from:
        try:
            if load_dotenv():
                loaded_from = "<autodetect>"
        except Exception:
            pass

    _ENV_LOADED = True

    try:
        from ai_trading.logging import get_logger, logger_once
        get_logger(__name__)
        if loaded_from:
            logger_once.info(
                "ENV_LOADED_FROM",
                key=f"env_loaded:{loaded_from}",
                extra={"dotenv_path": loaded_from},
            )
        else:
            logger_once.info("ENV_NOT_FOUND", key="env_not_found")
    except Exception:
        pass
