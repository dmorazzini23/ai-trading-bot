"""Outcome-free diagnostics of saved corporate-action bar samples.

These checks establish sampled consistency only, never full adjustment certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_trading.logging import get_logger
from ai_trading.runtime.atomic_io import atomic_write_text


def factor_interval(before_adjusted: float, before_base: float,
                    after_adjusted: float, after_base: float) -> tuple[float, float]:
    """Conservative interval assuming each displayed price rounds to a cent."""
    values = (before_adjusted, before_base, after_adjusted, after_base)
    if not all(math.isfinite(v) and v > 0.005 for v in values):
        raise ValueError("invalid adjustment price")
    a, b, c, d = values
    return ((a - .005) * (d - .005) / ((b + .005) * (c + .005)),
            (a + .005) * (d + .005) / ((b - .005) * (c - .005)))


def diagnose(sample: dict[str, Any]) -> dict[str, Any]:
    event = sample["event"]
    if not "2024-01-01" <= event["ex_date"] <= "2025-12-31":
        raise ValueError("event outside development interval")
    prices: dict[str, dict[str, float]] = {}
    timestamps: dict[str, set[datetime]] = {}
    for side in ("before", "after"):
        prices[side] = {}
        timestamps[side] = set()
        for adjustment in ("raw", "split", "all"):
            item = sample["bars"][side][adjustment]
            bars = item["response"]["bars"]
            if len(bars) != 1 or item["response"].get("next_page_token"):
                raise ValueError("sample must contain exactly one complete bar")
            bar = bars[0]
            if item["request"]["adjustment"] != adjustment:
                raise ValueError("adjustment request mismatch")
            ts = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
            requested = datetime.fromisoformat(item["request"]["start"])
            if ts.tzinfo is None or requested.tzinfo is None or ts != requested:
                raise ValueError("sample timestamp mismatch")
            timestamps[side].add(ts)
            price = float(bar["c"])
            if not math.isfinite(price) or price <= .005:
                raise ValueError("invalid adjustment price")
            prices[side][adjustment] = price
        if len(timestamps[side]) != 1:
            raise ValueError("unaligned samples")
    if next(iter(timestamps["before"])) >= next(iter(timestamps["after"])):
        raise ValueError("sample chronology invalid")
    before, after = prices["before"], prices["after"]
    if sample["kind"] == "cash_dividends":
        adjusted, base = "all", "split"
        rate = float(event["rate"])
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("invalid dividend rate")
        expected = 1 - rate / before["raw"]
    elif sample["kind"] in ("forward_splits", "reverse_splits"):
        adjusted, base = "split", "raw"
        expected = float(event["old_rate"]) / float(event["new_rate"])
    else:
        raise ValueError("unsupported corporate action")
    low, high = factor_interval(before[adjusted], before[base], after[adjusted], after[base])
    if not math.isfinite(expected) or expected <= 0:
        raise ValueError("invalid action factor")
    return {"symbol": event["symbol"], "ex_date": event["ex_date"],
            "kind": sample["kind"], "reference_factor": expected,
            "rounding_interval": [low, high], "consistent": low <= expected <= high}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.samples.read_bytes()
    source = json.loads(raw)
    rows = [diagnose(sample) for sample in source["samples"]]
    report = {"source_sha256": hashlib.sha256(raw).hexdigest(), "samples": rows,
              "sample_count": len(rows), "consistent_count": sum(r["consistent"] for r in rows),
              "status": "diagnostic_only_not_full_certification",
              "limitations": ["Rounding interval is an explicit diagnostic assumption, not verified provider precision.",
                              "Dividend reference uses final regular minute close, not certified official close.",
                              "Samples do not prove complete action coverage or point-in-time availability."],
              "returns_computed": False, "promotion_authority": False}
    atomic_write_text(args.output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    get_logger(__name__).info("ADJUSTMENT_DIAGNOSTICS_COMPLETE", extra={"sample_count": len(rows)})


if __name__ == "__main__":
    main()
