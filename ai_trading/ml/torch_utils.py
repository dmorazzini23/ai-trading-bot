from typing import TYPE_CHECKING

from ai_trading.imports import optional_import

# AI-AGENT-REF: central torch optional import and guard

torch = optional_import("torch")
if TYPE_CHECKING:  # pragma: no cover - type checker help
    import torch as _t  # noqa: F401


def ensure_torch() -> None:
    if torch is None:
        raise ImportError(
            "Torch is not installed. Install 'torch' to use this feature or run without Torch-dependent functionality."
        )
