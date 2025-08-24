"""Site customizations for test workers."""  # AI-AGENT-REF: ensure repo path and env
import os
import sys
from pathlib import Path

try:
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
except Exception:
    pass

os.environ.setdefault("AI_TRADING_FORCE_LOCAL_SLEEP", "1")
