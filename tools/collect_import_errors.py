#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

cmd = [
    sys.executable, "-m", "pytest",
    "-p", "xdist", "-p", "pytest_timeout", "-p", "pytest_asyncio",
    "-q", "--collect-only", "-o", "log_cli=true", "-o", "log_cli_level=INFO",
]
env = dict(os.environ)
env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
text = out.stdout + "\n" + out.stderr

mod_not_found = re.findall(r"No module named ['\"]([^'\"]+)['\"]", text)
import_errors = sorted(set(mod_not_found))

payload = {
    "status": out.returncode,
    "import_errors": import_errors,
    "raw_log_path": "artifacts/test-collect.log",
}
(ART / "test-collect.log").write_text(text, encoding="utf-8")
(ART / "import-errors.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

tmpl_path = ROOT / "artifacts" / "import-repair-report.md"
tmpl = tmpl_path.read_text(encoding="utf-8")
report = tmpl.replace("<!--IMPORT_ERRORS-->", "\n".join(f"- `{m}`" for m in import_errors))
report = report.replace("<!--COUNT-->", str(len(import_errors)))
tmpl_path.write_text(report, encoding="utf-8")

print(json.dumps(payload, indent=2))
# AI-AGENT-REF: harvest import errors deterministically
sys.exit(out.returncode if import_errors else 0)
