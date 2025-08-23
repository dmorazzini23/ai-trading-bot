#!/usr/bin/env python3
"""Harvest import errors and emit a report with an env header."""

# AI-AGENT-REF: new env summary + assertion logic
from __future__ import annotations

import os
import re
import sys
import platform
import subprocess
from pathlib import Path
from typing import Dict, Tuple

try:  # AI-AGENT-REF: optional packaging
    from packaging.tags import sys_tags  # type: ignore
except Exception:  # pragma: no cover - best effort
    sys_tags = None


# AI-AGENT-REF: expected header on canonical host
EXPECTED_ENV_LINE = (
    "Ubuntu 24.04 | glibc 2.39 | CPython 3.12.3 | tag cp312-manylinux_2_39_x86_64"
)


# AI-AGENT-REF: robust os-release reader
def _read_os_release() -> Dict[str, str]:
    data: Dict[str, str] = {}
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                data[k] = v.strip().strip('"')
    except Exception:
        pass
    return data


# AI-AGENT-REF: glibc version with getconf fallback
def _glibc_version() -> str:
    _, ver = platform.libc_ver()
    if ver:
        return ver
    try:
        out = subprocess.check_output(
            ["getconf", "GNU_LIBC_VERSION"], text=True
        ).strip()
        m = re.search(r"glibc\s+(\d+\.\d+)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "unknown"


# AI-AGENT-REF: normalize first wheel tag
def _top_normalized_tag() -> str:
    try:
        if sys_tags is None:
            return "unknown"
        tag = str(next(sys_tags()))
        return re.sub(r"^([^-]+)-[^-]+-", r"\1-", tag)
    except Exception:
        return "unknown"


# AI-AGENT-REF: compute human-readable env line
def compute_env_summary() -> Tuple[str, Dict[str, str]]:
    osr = _read_os_release()
    distro = (osr.get("ID") or "unknown").lower()
    version = osr.get("VERSION_ID") or "unknown"
    glibc = _glibc_version()
    py_impl = platform.python_implementation()
    py_ver = ".".join(platform.python_version_tuple())
    tag = _top_normalized_tag()

    distro_title = distro.capitalize()
    env_line = f"{distro_title} {version} | glibc {glibc} | {py_impl} {py_ver} | tag {tag}"
    parts = {
        "distro": distro,
        "version": version,
        "glibc": glibc,
        "py_impl": py_impl,
        "py_ver": py_ver,
        "tag": tag,
    }
    return env_line, parts


def main() -> None:
    # AI-AGENT-REF: prepend env header and assert on canonical host
    env_line, env_parts = compute_env_summary()
    if (
        env_parts["distro"] == "ubuntu"
        and env_parts["version"].startswith("24.04")
        and env_parts["py_impl"] == "CPython"
        and env_parts["py_ver"] == "3.12.3"
    ):
        assert (
            env_line == EXPECTED_ENV_LINE
        ), f"Env summary mismatch: '{env_line}' != '{EXPECTED_ENV_LINE}'"

    report_lines: list[str] = []
    report_lines.append("# Import/Dependency Repair Report\n\n")
    report_lines.append(f"**Environment:** {env_line}\n\n")

    # AI-AGENT-REF: existing harvesting logic
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

    report_lines.append("## Summary\n")
    report_lines.append(f"- Internal import errors (unique): {len(internal)}\n")
    report_lines.extend(f"- `{m}`\n" for m in internal) if internal else report_lines.append("- none\n")
    report_lines.append("\n")
    report_lines.append(f"- External import errors (unique): {len(external)}\n")
    report_lines.extend(f"- `{m}`\n" for m in external) if external else report_lines.append("- none\n")
    report_lines.append("\n## Notes\n")
    report_lines.append(
        "- Treat entries starting with `ai_trading.` as **internal**; fix by adding/renaming exports or updating the static rewrite map.\n"
    )
    report_lines.append(
        "- Treat others as **external**; fix by adding pins to `requirements.txt` + `constraints.txt` (not to dev-only).\n"
    )

    out_path = Path("artifacts/import-repair-report.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(report_lines), encoding="utf-8")

    sys.exit(out.returncode if internal or external else 0)


if __name__ == "__main__":  # pragma: no cover - script entry
    main()

