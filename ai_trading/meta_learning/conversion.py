"""Trade log conversion helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .core import validate_trade_data_quality


def should_convert(trade_log_path: str | Path) -> bool:
    """Return True if trade log matches known formats.

    The function considers four acceptable scenarios:
    - pure meta-learning format
    - pure audit format
    - mixed format (meta + audit rows)
    - exact problem statement example

    It returns ``False`` when the file is missing or empty.
    """
    report: dict[str, Any] = validate_trade_data_quality(str(trade_log_path))
    if not report.get("file_exists"):
        return False
    if report.get("row_count", 0) == 0:
        return False

    audit_rows = report.get("audit_format_rows", 0)
    meta_rows = report.get("meta_format_rows", 0)
    mixed = report.get("mixed_format_detected", False)

    if mixed:
        return True
    if audit_rows > 0 and meta_rows == 0:
        return True
    if meta_rows > 0 and audit_rows == 0:
        return True
    return False
