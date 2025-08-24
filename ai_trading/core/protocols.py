from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TYPE_CHECKING  # AI-AGENT-REF: add TYPE_CHECKING to break cycle

if TYPE_CHECKING:  # AI-AGENT-REF: avoid runtime import cycle
    from .runtime import BotRuntime


class AllocatorProtocol(Protocol):
    def allocate(self, signals: Sequence[Mapping[str, Any]], runtime: 'BotRuntime') -> Mapping[str, Any]:  # AI-AGENT-REF: forward ref to runtime
        """Allocate positions based on signals."""
        raise NotImplementedError  # AI-AGENT-REF: remove placeholder ellipsis

