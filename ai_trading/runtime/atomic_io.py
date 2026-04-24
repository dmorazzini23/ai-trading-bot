"""Atomic filesystem write helpers for runtime artifacts."""

from __future__ import annotations

import os
import tempfile
from contextlib import suppress
from pathlib import Path


def atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Write ``content`` to ``path`` via fsynced temp file and atomic replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    tmp_path = Path(tmp_name)
    replaced = False
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, destination)
        replaced = True
        try:
            dir_fd = os.open(destination.parent, os.O_DIRECTORY)
        except OSError:
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if not replaced:
            with suppress(OSError):
                tmp_path.unlink(missing_ok=True)
    return destination


__all__ = ["atomic_write_text"]
