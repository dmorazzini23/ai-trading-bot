from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


LEGACY_DUMMY_ENV_SCRIPTS = (
    "scripts/demo_enhanced_debugging.py",
    "scripts/demo_no_trade_bands.py",
    "scripts/demo_short_selling_implementation.py",
    "scripts/final_validation.py",
    "scripts/setup_test_env.py",
    "scripts/validate_critical_fix.py",
    "scripts/validate_critical_fixes.py",
    "scripts/validate_enhanced_debugging.py",
    "scripts/validate_problem_statement_fixes.py",
    "scripts/validate_standalone.py",
    "scripts/validate_startup_fixes.py",
    "scripts/validate_unified_config.py",
)


def test_legacy_dummy_env_scripts_require_explicit_demo_flag() -> None:
    for relative_path in LEGACY_DUMMY_ENV_SCRIPTS:
        content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
        assert "require_legacy_demo_flag" in content
        marker_index = content.index("require_legacy_demo_flag")
        seeded_env_indexes = [
            content.find("ALPACA_BASE_URL"),
            content.find("FLASK_PORT"),
            content.find("ALPACA_API_KEY"),
        ]
        seeded_env_indexes = [index for index in seeded_env_indexes if index >= 0]
        assert seeded_env_indexes
        assert marker_index < min(seeded_env_indexes)


def test_legacy_guard_rejects_without_flag_and_allows_with_env() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            f"sys.path.insert(0, {str(REPO_ROOT / 'scripts')!r}); "
            "from legacy_guard import require_legacy_demo_flag; "
            "require_legacy_demo_flag('unit-test')"
        ),
    ]

    rejected = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    assert rejected.returncode == 2
    assert "archival legacy/demo validation script" in rejected.stderr

    env = {**os.environ, "AI_TRADING_ENABLE_LEGACY_DEMO": "1"}
    allowed = subprocess.run(command, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    assert allowed.returncode == 0
