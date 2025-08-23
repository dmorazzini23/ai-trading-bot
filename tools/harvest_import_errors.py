#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
import pathlib
import platform
from packaging import tags


# AI-AGENT-REF: read release metadata
def _read_os_release() -> dict:
    info: dict[str, str] = {}
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    info[k] = v.strip().strip('"')
    except FileNotFoundError:
        pass
    return info


# AI-AGENT-REF: compute env line and components
def _top_wheel_tag() -> str:
    try:
        first = str(next(tags.sys_tags()))
        return first.replace("cp312-cp312", "cp312")
    except Exception:
        return "unknown"


def _env_line() -> tuple[str, str, str, str, str, str]:
    """Return (line, os_id, ver, glibc_ver, py, tag)."""
    osr = _read_os_release()
    os_id = osr.get("ID", "unknown").strip()
    ver = osr.get("VERSION_ID", "?").strip()
    libc, glibc_ver = platform.libc_ver()
    py = f"{platform.python_implementation()} {sys.version.split()[0]}"
    tag = _top_wheel_tag()
    line = f"{os_id.capitalize()} {ver} | glibc {glibc_ver or '?'} | {py} | tag {tag}"
    return line, os_id, ver, glibc_ver, py, tag


ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description="Harvest import errors")
parser.add_argument("--write", type=str, help="Path to write report", default=None)
args = parser.parse_args()

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
out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
text = out.stdout + "\n" + out.stderr

mod_not_found = re.findall(r"No module named ['\"]([^'\"]+)['\"]", text)
import_errors = sorted(set(mod_not_found))

(ART / "test-collect.log").write_text(text, encoding="utf-8")

internal = sorted(m for m in import_errors if m.startswith("ai_trading"))
external = sorted(m for m in import_errors if not m.startswith("ai_trading"))
is_rl_disabled = os.getenv("WITH_RL", "0") != "1"
if is_rl_disabled:
    suppress = {"torch", "stable_baselines3", "gymnasium"}
    external = [m for m in external if m not in suppress]

env_line, os_id, ver, glibc_ver, py, tag = _env_line()

TEMPLATE = """# Import/Dependency Repair Report

## Summary
- Internal import errors (unique): <!--ICOUNT-->
<!--INTERNAL-->

- External import errors (unique): <!--ECOUNT-->
<!--EXTERNAL-->

## Notes
- Treat entries starting with `ai_trading.` as **internal**; fix by adding/renaming exports or updating the static rewrite map.
- Treat others as **external**; fix by adding pins to `requirements.txt` + `constraints.txt` (not to dev-only).
"""

report = TEMPLATE
report = report.replace("<!--INTERNAL-->", "\n".join(f"- `{m}`" for m in internal) or "- none")
report = report.replace("<!--EXTERNAL-->", "\n".join(f"- `{m}`" for m in external) or "- none")
report = report.replace("<!--ICOUNT-->", str(len(internal)))
report = report.replace("<!--ECOUNT-->", str(len(external)))

header = f"**Environment:** {env_line}\n\n"
report = header + report

if (
    os_id == "ubuntu"
    and ver.startswith("24.04")
    and py.startswith("CPython 3.12.3")
    and "manylinux_2_39_x86_64" in tag
):
    expected = (
        "Ubuntu 24.04 | glibc 2.39 | CPython 3.12.3 | tag cp312-manylinux_2_39_x86_64"
    )
    assert env_line == expected, f"env line mismatch: {env_line!r} != {expected!r}"

if args.write:
    (ROOT / args.write).write_text(report, encoding="utf-8")
else:
    print(report)

# AI-AGENT-REF: harvest import errors deterministically
sys.exit(out.returncode if internal or external else 0)
