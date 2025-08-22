# blocks all outbound sockets unless explicitly allowed
import os
import socket
import functools

class _NetworkBlockedError(RuntimeError): ...

def _deny(*args, **kwargs):
    raise _NetworkBlockedError("Network is blocked in test mode (RUN_INTEGRATION not set).")

def block_network():
    socket.socket = functools.partial(_deny)  # type: ignore[attr-defined]

def unblock_network():
    # Best-effort restore – only used inside integration tests gate
    from socket import socket as real_socket  # local import to avoid shadowing
    socket.socket = real_socket  # type: ignore[attr-defined]

def should_block():
    return os.environ.get("RUN_INTEGRATION") not in {"1", "true", "TRUE", "yes"}
