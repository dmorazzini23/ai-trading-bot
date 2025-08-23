"""Minimal validator expected by tests; conservative default behavior."""
from __future__ import annotations
from dataclasses import dataclass

# AI-AGENT-REF: restore short selling facade


@dataclass(frozen=True)
class ShortSellConfig:
    allow_shorts: bool = False


def validate_short_selling(symbol: str, *, cfg: ShortSellConfig | None = None) -> bool:
    cfg = cfg or ShortSellConfig()
    return bool(cfg.allow_shorts)

