from __future__ import annotations
from datetime import timezone

# AI-AGENT-REF: fallback timezone helper
try:
    from tzlocal import get_localzone  # type: ignore

    def local_tz():
        return get_localzone()
except Exception:  # pragma: no cover - only triggers in extra-minimal envs
    def local_tz():
        # fallback to UTC if tzlocal is unavailable
        return timezone.utc
