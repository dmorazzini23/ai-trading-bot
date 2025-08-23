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


# AI-AGENT-REF: compute env line
def _compute_env_line() -> str:
    """
    Example:
    Ubuntu 24.04 | glibc 2.39 | CPython 3.12.3 | tag cp312-manylinux_2_39_x86_64
    """
    osr = _read_os_release()
    name = osr.get("NAME", "Linux")
    ver = osr.get("VERSION_ID", "").strip()
    distro = f"{name} {ver}".strip()

    libc, libc_ver = platform.libc_ver()
    if platform.system() == "Linux":
        assert (
            libc == "glibc" and libc_ver
        ), "Expected glibc on Ubuntu; if using a musl-based image, update the harvester."

    py = f"{platform.python_implementation()} {sys.version.split()[0]}"

    try:
        top = str(next(tags.sys_tags()))
        top = top.replace("cp312-cp312-", "cp312-")
    except Exception:
        top = "unknown"

    return f"{distro} | {libc} {libc_ver} | {py} | tag {top}"


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

env_line = _compute_env_line()

TEMPLATE = """# Import/Dependency Repair Report

**Env:** <!--ENV-->

## Summary
- Internal import errors (unique): <!--ICOUNT-->
<!--INTERNAL-->

- External import errors (unique): <!--ECOUNT-->
<!--EXTERNAL-->

## Notes
- Treat entries starting with `ai_trading.` as **internal**; fix by adding/renaming exports or updating the static rewrite map.
- Treat others as **external**; fix by adding pins to `requirements.txt` + `constraints.txt` (not to dev-only).
"""

report = TEMPLATE.replace("<!--ENV-->", env_line)
report = report.replace("<!--INTERNAL-->", "\n".join(f"- `{m}`" for m in internal) or "- none")
report = report.replace("<!--EXTERNAL-->", "\n".join(f"- `{m}`" for m in external) or "- none")
report = report.replace("<!--ICOUNT-->", str(len(internal)))
report = report.replace("<!--ECOUNT-->", str(len(external)))

if args.write:
    (ROOT / args.write).write_text(report, encoding="utf-8")
else:
    print(report)

# AI-AGENT-REF: harvest import errors deterministically
sys.exit(out.returncode if internal or external else 0)
