"""Performance metrics for trading results with numerical stability."""
from __future__ import annotations

import json
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, TYPE_CHECKING
from importlib.util import find_spec

from ai_trading.logging import logger
HAS_PANDAS = find_spec("pandas") is not None
HAS_NUMPY = True

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

