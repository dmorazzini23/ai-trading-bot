from __future__ import annotations

import os
import sys
from pathlib import Path

from ai_trading.core.bot_engine import get_trade_logger
from ai_trading.logging import get_logger

logger = get_logger(__name__)


def _is_dir_writable(dir_path: Path) -> bool:
    """Return True if *dir_path* is writable for the current user."""
    try:
        st = dir_path.stat()
    except OSError:
        return False
    mode = st.st_mode
    uid = os.geteuid()
    gid = os.getegid()
    try:
        groups = set(os.getgroups())
    except Exception:  # pragma: no cover - platform specific
        groups = {gid}
    import stat as _stat
    if uid == st.st_uid:
        return bool(mode & _stat.S_IWUSR)
    if st.st_gid == gid or st.st_gid in groups:
        return bool(mode & _stat.S_IWGRP)
    return bool(mode & _stat.S_IWOTH)


def ensure_trade_log_path() -> None:
    """Ensure the trade log file exists and is writable.

    Creates parent directories as needed and exits with status 1 if the path
    cannot be written.
    """
    tl = get_trade_logger()
    path = Path(tl.path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not _is_dir_writable(path.parent):
            raise OSError("directory not writable")
        with open(path, "a"):
            pass
    except OSError as exc:  # pragma: no cover - fail fast
        logger.critical(
            "TRADE_LOG_PATH_UNWRITABLE",
            extra={"path": str(path), "error": str(exc)},
        )
        sys.exit(1)
