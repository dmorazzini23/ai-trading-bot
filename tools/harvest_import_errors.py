#!/usr/bin/env python
"""Harvest import errors and emit a report with an env header."""

# AI-AGENT-REF: env summary + assertion logic
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from packaging import tags as pkg_tags

# -- Environment detection ----------------------------------------------------
# AI-AGENT-REF: glibc detection with getconf fallback
def _detect_glibc_version() -> str:
    libc, ver = platform.libc_ver()
    if libc and ver:
        return ver
    try:
        out = subprocess.check_output(["getconf", "GNU_LIBC_VERSION"], text=True).strip()
        return out.split()[-1]
    except Exception:  # pragma: no cover - best effort
        return "unknown"

# AI-AGENT-REF: normalize first wheel tag
def _top_wheel_tag_normalized() -> str:
    t = next(iter(pkg_tags.sys_tags()))
    interp = t.interpreter
    plat = t.platform
    return f"{interp}-{plat}"

# AI-AGENT-REF: build human-readable env string
def compute_env_summary_line() -> str:
    distro = "Ubuntu 24.04"  # our canonical build host
    glibc = _detect_glibc_version()
    py = f"CPython {platform.python_version()}"
    tag = _top_wheel_tag_normalized()
    return f"{distro} | glibc {glibc} | {py} | tag {tag}"

# AI-AGENT-REF: canonical env assertion with escape hatch
def assert_expected_combo(line: str) -> None:
    if os.environ.get("DISABLE_ENV_ASSERT") == "1":
        return
    expected = "Ubuntu 24.04 | glibc 2.39 | CPython 3.12.3 | tag cp312-manylinux_2_39_x86_64"
    assert line == expected, (
        "Environment drift detected.\n"
        f" expected: {expected}\n"
        f"      got: {line}\n"
        "Set DISABLE_ENV_ASSERT=1 to bypass on non-canonical hosts."
    )


def main() -> None:
    # AI-AGENT-REF: prepend env header and assert canonical combo
    env_line = compute_env_summary_line()
    assert_expected_combo(env_line)

    root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "xdist",
        "-p",
        "pytest_timeout",
        "-p",
        "pytest_asyncio",
        "-q",
        "--collect-only",
        "-o",
        "log_cli=true",
        "-o",
        "log_cli_level=INFO",
    ]
    env = dict(os.environ)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    out = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    text = out.stdout + "\n" + out.stderr
    (root / "artifacts/test-collect.log").write_text(text, encoding="utf-8")

    mod_not_found = re.findall(r"No module named ['\"]([^'\"]+)['\"]", text)
    import_errors = sorted(set(mod_not_found))
    internal = sorted(m for m in import_errors if m.startswith("ai_trading"))
    external = sorted(m for m in import_errors if not m.startswith("ai_trading"))
    if os.getenv("WITH_RL", "0") != "1":
        suppress = {"torch", "stable_baselines3", "gymnasium"}
        external = [m for m in external if m not in suppress]

    body_lines: list[str] = []
    body_lines.append("# Import/Dependency Repair Report")
    body_lines.append("")
    body_lines.append("## Summary")
    body_lines.append(f"- Internal import errors (unique): {len(internal)}")
    if internal:
        body_lines.extend(f"- `{m}`" for m in internal)
    else:
        body_lines.append("- none")
    body_lines.append("")
    body_lines.append(f"- External import errors (unique): {len(external)}")
    if external:
        body_lines.extend(f"- `{m}`" for m in external)
    else:
        body_lines.append("- none")
    body_lines.append("")
    body_lines.append("## Notes")
    body_lines.append(
        "- Treat entries starting with `ai_trading.` as **internal**; fix by adding/renaming exports or updating the static rewrite map."
    )
    body_lines.append(
        "- Treat others as **external**; fix by adding pins to `requirements.txt` + `constraints.txt` (not to dev-only)."
    )

    artifact = Path(os.environ.get("IMPORT_REPAIR_REPORT", "artifacts/import-repair-report.md"))
    artifact.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"**Environment**: {env_line}", ""]
    lines.extend(body_lines)
    artifact.write_text("\n".join(lines), encoding="utf-8")

    sys.exit(out.returncode if internal or external else 0)


if __name__ == "__main__":  # pragma: no cover - script entry
    main()

