from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


def test_ai_trading_package_has_no_literal_broad_except_handlers() -> None:
    rg = shutil.which("rg")
    if rg is None:
        matches = [
            str(path)
            for path in Path("ai_trading").rglob("*.py")
            if "except Exception" in path.read_text(encoding="utf-8")
        ]
        assert not matches, "\n".join(matches)
        return
    result = subprocess.run(
        [
            rg,
            "-n",
            "except Exception",
            "ai_trading",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout
