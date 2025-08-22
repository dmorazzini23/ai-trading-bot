from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class AllocatorProtocol(Protocol):
    def allocate(self, signals: Sequence[Mapping[str, Any]], runtime: BotRuntime) -> Mapping[str, Any]: ...
