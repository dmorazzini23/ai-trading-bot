from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TYPE_CHECKING, runtime_checkable  # AI-AGENT-REF: add TYPE_CHECKING to break cycle

if TYPE_CHECKING:  # AI-AGENT-REF: avoid runtime import cycle
    from .runtime import BotRuntime


@runtime_checkable
class AllocatorProtocol(Protocol):
    def allocate(
        self,
        signals: Sequence[Mapping[str, Any]],
        runtime: 'BotRuntime',
    ) -> Mapping[str, Any]:  # AI-AGENT-REF: forward ref to runtime
        """Return target allocations derived from *signals* and runtime state.

        Implementations should examine each signal and may consult the runtime
        for context to produce a mapping of symbols to allocation metadata.
        """
        ...

