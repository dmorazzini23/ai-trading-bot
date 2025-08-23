"""Production ProcessManager (migrated from scripts/process_manager.py)."""
from __future__ import annotations
import atexit
import os
import pathlib
import signal
import sys
import fcntl

class ProcessManager:
    """Ensure only one ai-trading instance runs at a time."""

    def __init__(self, lock_name: str='ai-trading', dir_env: str='AI_TRADING_RUNTIME_DIR') -> None:
        runtime_dir = pathlib.Path(os.getenv(dir_env, '/tmp')).resolve()
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self._lockfile = runtime_dir / f'{lock_name}.lock'
        self._fd: int | None = None

    def lockfile_path(self) -> str:
        """Return absolute path to the process lock file."""
        return str(self._lockfile)

    def ensure_single_instance(self) -> bool:
        """Acquire lock or return False if another instance is running."""
        self._fd = os.open(self._lockfile, os.O_CREAT | os.O_RDWR, 420)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.write(self._fd, str(os.getpid()).encode('utf-8'))
            os.fsync(self._fd)
        except OSError:
            return False
        atexit.register(self._cleanup)
        signal.signal(signal.SIGTERM, self._sigexit)
        signal.signal(signal.SIGINT, self._sigexit)
        return True

    def _sigexit(self, *_args) -> None:
        """Handle termination signals and release lock."""
        self._cleanup()
        sys.exit(0)

    def _cleanup(self) -> None:
        """Release file lock and remove lockfile."""
        try:
            if self._fd is not None:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            if self._lockfile.exists():
                self._lockfile.unlink(missing_ok=True)
        except (ValueError, OSError, PermissionError, KeyError, TypeError):
            pass

    def __enter__(self) -> ProcessManager:
        if not self.ensure_single_instance():
            raise SystemExit('Another ai-trading instance is already running.')
        return self

    def __exit__(self, *exc) -> bool:
        self._cleanup()
        return False