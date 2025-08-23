from __future__ import annotations

import logging

try:  # RL optional: never fail import if torch isn't installed
    import torch  # type: ignore[assignment]
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]

_log = logging.getLogger(__name__)


def get_device() -> str:
    """Simple, import-safe detection of the ML device."""
    if torch is not None:
        try:
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                dev = "cuda"
                _log.info("ML_DEVICE_DETECTED", extra={"device": dev})  # AI-AGENT-REF: optional CUDA
                return dev
        except Exception:  # defensive: do not let CUDA probing blow up
            pass
    dev = "cpu"
    _log.info("ML_DEVICE_DETECTED", extra={"device": dev})  # AI-AGENT-REF: default CPU
    return dev


def tensors_to_device(batch: dict, device: str):
    """Move a tokenizer batch (dict of tensors) to the target device."""
    if torch is None:
        return batch
    try:
        from torch import Tensor
    except Exception:  # noqa: BLE001
        return batch
    return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
