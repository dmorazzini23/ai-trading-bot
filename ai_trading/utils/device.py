import logging
import os
from typing import Any

_log = logging.getLogger(__name__)  # AI-AGENT-REF: structured logging


def pick_torch_device() -> tuple[str, Any | None]:
    try:
        import torch  # type: ignore
    except Exception:
        _log.info("ML_DEVICE_SELECTED", extra={"device": "cpu", "reason": "torch_unavailable"})
        return "cpu", None

    if os.getenv("CPU_ONLY", "").strip() == "1":
        dev = "cpu"
    elif torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # macOS
        dev = "mps"
    else:
        dev = "cpu"

    _log.info("ML_DEVICE_SELECTED", extra={"device": dev})
    return dev, torch


def tensors_to_device(batch: dict, device: str):
    """Move a tokenizer batch (dict of tensors) to the target device."""  # AI-AGENT-REF: device helper
    tv = []
    try:
        from torch import Tensor  # type: ignore
        tv.append(Tensor)
    except Exception:
        return batch
    return {k: (v.to(device) if isinstance(v, tuple(tv)) else v) for k, v in batch.items()}
