from __future__ import annotations

from ai_trading.logging import get_logger

# torch is optional and heavy; import lazily inside functions
# Cache references for repeated calls
torch = None  # type: ignore[assignment]
_torch_tensor = None
_device = None

logger = get_logger(__name__)


def get_device() -> str:
    """Simple, import-safe detection of the ML device."""
    global torch, _device
    if _device is not None:
        return _device

    try:
        import torch as _torch  # type: ignore[assignment]
        torch = _torch
    except Exception:  # noqa: BLE001
        _device = "cpu"
    else:
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                _device = "cuda"
            else:
                _device = "cpu"
        except Exception:  # defensive: do not let CUDA probing blow up
            _device = "cpu"

    logger.info("ML_DEVICE_DETECTED", extra={"device": _device})  # AI-AGENT-REF: default CPU / optional CUDA
    return _device


def tensors_to_device(batch: dict, device: str):
    """Move a tokenizer batch (dict of tensors) to the target device."""
    global torch, _torch_tensor
    if torch is None:
        try:
            import torch as _torch  # type: ignore[assignment]
            torch = _torch
        except Exception:  # noqa: BLE001
            return batch
    if _torch_tensor is None:
        try:
            from torch import Tensor as _Tensor
            _torch_tensor = _Tensor
        except Exception:  # noqa: BLE001
            return batch
    return {k: v.to(device) if isinstance(v, _torch_tensor) else v for k, v in batch.items()}
