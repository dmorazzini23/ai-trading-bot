from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class IdleStatus:
    reason: Optional[str] = None
    next_check: Optional[datetime] = None

idle_status = IdleStatus()
