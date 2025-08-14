from __future__ import annotations
import sys

# AI-AGENT-REF: warn but allow Python 3.11/3.12
maj, min = sys.version_info[:2]
if not ((maj, min) in [(3, 12), (3, 11)]):
    print(f"WARNING: Python {sys.version.split()[0]} detected; expected 3.11 or 3.12. Continuing.")
else:
    print(f"Python OK: {maj}.{min}")
