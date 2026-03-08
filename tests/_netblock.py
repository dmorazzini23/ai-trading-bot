# blocks all outbound sockets unless explicitly allowed
import os
import socket
import functools
from typing import Any, cast

class _NetworkBlockedError(RuntimeError): ...

def _deny(*args, **kwargs):
    raise _NetworkBlockedError("Network is blocked in test mode (RUN_INTEGRATION not set).")

def block_network():
    cast(Any, socket).socket = functools.partial(_deny)

def unblock_network():
    # Best-effort restore – only used inside integration tests gate
    from socket import socket as real_socket  # local import to avoid shadowing
    cast(Any, socket).socket = real_socket

def should_block():
    return os.environ.get("RUN_INTEGRATION") not in {"1", "true", "TRUE", "yes"}
