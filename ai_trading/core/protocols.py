from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence


class AllocatorProtocol(Protocol):
    def allocate(self, signals: Sequence[Mapping[str, Any]], runtime: "BotRuntime") -> Mapping[str, Any]: ...
