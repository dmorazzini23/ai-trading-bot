from __future__ import annotations

import subprocess


def test_ai_trading_package_has_no_literal_broad_except_handlers() -> None:
    result = subprocess.run(
        [
            "rg",
            "-n",
            "except Exception",
            "ai_trading",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout
