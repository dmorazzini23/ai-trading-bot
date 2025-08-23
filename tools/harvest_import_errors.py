#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
import pathlib

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

TEMPLATE = """# Import/Dependency Repair Report

**Env**
- Distro: Ubuntu 24.04 (glibc 2.39)
- Python: CPython 3.12.3
- Wheel tag: `cp312-manylinux_2_39_x86_64`

## Summary
- Internal import errors (unique): <!--ICOUNT-->
<!--INTERNAL-->

- External import errors (unique): <!--ECOUNT-->
<!--EXTERNAL-->

## Notes
- Treat entries starting with `ai_trading.` as **internal**; fix by adding/renaming exports or updating the static rewrite map.
- Treat others as **external**; fix by adding pins to `requirements.txt` + `constraints.txt` (not to dev-only).
"""

report = TEMPLATE.replace("<!--INTERNAL-->", "\n".join(f"- `{m}`" for m in internal) or "- none")
report = report.replace("<!--EXTERNAL-->", "\n".join(f"- `{m}`" for m in external) or "- none")
report = report.replace("<!--ICOUNT-->", str(len(internal)))
report = report.replace("<!--ECOUNT-->", str(len(external)))

if args.write:
    (ROOT / args.write).write_text(report, encoding="utf-8")
else:
    print(report)

# AI-AGENT-REF: harvest import errors deterministically
sys.exit(out.returncode if internal or external else 0)
