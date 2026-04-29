from __future__ import annotations

from ai_trading.utils.universe import load_universe


def test_long_inline_csv_is_not_path_probed() -> None:
    inline_csv = ",".join(f"SYM{i}" for i in range(300))

    symbols = load_universe(inline_csv)

    assert symbols[:3] == ["SYM0", "SYM1", "SYM2"]
    assert symbols[-1] == "SYM299"
    assert len(symbols) == 300
