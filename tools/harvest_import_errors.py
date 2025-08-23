#!/usr/bin/env python3
from __future__ import annotations
import os
import re
import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

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

TEMPLATE = """# Import/Dependency Repair Report

**Env**
- Distro: Ubuntu 24.04 (glibc 2.39)
- Python: CPython 3.12.3
- Wheel tag: `cp312-manylinux_2_39_x86_64`

## Summary
- Remaining import errors (unique): <!--COUNT-->
- Modules:
<!--IMPORT_ERRORS-->

## Notes
- Treat entries starting with `ai_trading.` as **internal**; fix by adding/renaming exports or updating the static rewrite map.
- Treat others as **external**; fix by adding pins to `requirements.txt` + `constraints.txt` (not to dev-only).
"""

report = TEMPLATE.replace("<!--IMPORT_ERRORS-->", "\n".join(f"- `{m}`" for m in import_errors))
report = report.replace("<!--COUNT-->", str(len(import_errors)))
print(report)

# AI-AGENT-REF: harvest import errors deterministically
sys.exit(out.returncode if import_errors else 0)
