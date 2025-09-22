"""Runtime shutdown coordination helpers.

This module centralizes graceful shutdown primitives so the service can play
nicely with systemd.  A global :class:`threading.Event` named ``stop_event`` is
exposed alongside convenience helpers for registering POSIX signal handlers,
triggering cooperative stop requests, and querying stop status from long
running loops.
"""

from __future__ import annotations

import logging
import signal
import threading
from types import FrameType
from typing import Iterable, Optional

_LOGGER = logging.getLogger("ai_trading.runtime.shutdown")

# Public event inspected by loops throughout the code base.
stop_event = threading.Event()

# Track whether the signal handlers have been registered so repeated imports do
# not override custom handlers installed during tests.
_handlers_installed = False
_installed_signals: set[int] = set()


def should_stop() -> bool:
    """Return ``True`` when a cooperative shutdown has been requested."""

    return stop_event.is_set()


def request_stop(reason: str | None = None) -> None:
    """Set :data:`stop_event` and emit a diagnostic log once.

    Parameters
    ----------
    reason:
        Optional human readable hint describing why the shutdown was requested.
    """

    if stop_event.is_set():
        return
    payload: dict[str, object] = {}
    if reason:
        payload["reason"] = reason
    _LOGGER.info("Shutdown signal received", extra=payload)
    stop_event.set()


def _handle_signal(signum: int, frame: Optional[FrameType]) -> None:  # pragma: no cover - exercised via integration
    try:
        signame = signal.Signals(signum).name
    except Exception:  # pragma: no cover - very defensive
        signame = str(signum)
    request_stop(f"signal:{signame}")


def register_signal_handlers(signals_to_handle: Iterable[int] | None = None) -> None:
    """Install signal handlers that trigger :func:`request_stop`.

    The registration is idempotent—subsequent calls only add missing signal
    registrations.  The default set includes ``SIGTERM`` and ``SIGINT`` and
    ``SIGQUIT`` when available.
    """

    global _handlers_installed
    if signals_to_handle is None:
        default_signals = [signal.SIGTERM, signal.SIGINT]
        if hasattr(signal, "SIGQUIT"):
            default_signals.append(signal.SIGQUIT)  # type: ignore[attr-defined]
        signals_to_handle = default_signals

    for sig in signals_to_handle:
        try:
            if sig in _installed_signals:
                continue
            signal.signal(sig, _handle_signal)
            _installed_signals.add(sig)
        except (ValueError, OSError) as exc:  # pragma: no cover - platform dependent
            _LOGGER.warning(
                "SIGNAL_REGISTRATION_FAILED",
                extra={"signal": sig, "error": str(exc)},
            )
    _handlers_installed = True


def install_runtime_timer(max_runtime_seconds: float) -> threading.Timer:
    """Arm a daemon timer that triggers a cooperative shutdown.

    Parameters
    ----------
    max_runtime_seconds:
        Number of seconds after which :func:`request_stop` is invoked.
    """

    delay = max(0.0, float(max_runtime_seconds))

    def _expire() -> None:
        request_stop("max-runtime-seconds")

    timer = threading.Timer(delay, _expire)
    timer.daemon = True
    timer.start()
    return timer


__all__ = ["stop_event", "should_stop", "request_stop", "register_signal_handlers", "install_runtime_timer"]

