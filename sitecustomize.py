"""Site customizations for test workers."""  # AI-AGENT-REF: ensure repo path and env
import os
import sys
from pathlib import Path

try:
    repo_root = Path(__file__).resolve().parent
    third_party_stubs = repo_root / "third_party_stubs"
    # Append stub paths so that bundled ``site-packages`` take precedence
    # over our third-party type stubs. This avoids situations where the stubs
    # overshadow the real packages that tests rely on (e.g. ``requests``).
    if str(third_party_stubs) not in sys.path:
        sys.path.append(str(third_party_stubs))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
except Exception:
    pass

os.environ.setdefault("AI_TRADING_FORCE_LOCAL_SLEEP", "1")
