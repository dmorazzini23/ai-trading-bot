
from __future__ import annotations

# Simple CLI pass-through to ai_trading.tools.env_validate (single source of truth)
from ai_trading.tools.env_validate import _main  # AI-AGENT-REF

if __name__ == "__main__":
    raise SystemExit(_main())
