from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any, Protocol
from .runtime import BotRuntime


class AllocatorProtocol(Protocol):
    def allocate(self, signals: Sequence[Mapping[str, Any]], runtime: BotRuntime) -> Mapping[str, Any]:
        """Allocate positions based on signals."""
        raise NotImplementedError  # AI-AGENT-REF: remove placeholder ellipsis

