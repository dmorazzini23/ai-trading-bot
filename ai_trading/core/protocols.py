from __future__ import annotations
from typing import Protocol, Sequence, Mapping, Any


class AllocatorProtocol(Protocol):
    def allocate(self, signals: Sequence[Mapping[str, Any]], runtime: "BotRuntime") -> Mapping[str, Any]: ...
