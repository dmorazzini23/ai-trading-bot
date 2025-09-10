import os
import socket

from ai_trading.utils import get_pid_on_port


def test_get_pid_on_port_ipv6():
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    s.bind(("::", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        pid = get_pid_on_port(port)
        assert pid == os.getpid()
    finally:
        s.close()
