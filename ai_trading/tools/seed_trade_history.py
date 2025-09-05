from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable
from ai_trading.logging import get_logger
from ai_trading.database.connection import initialize_database, get_session
from ai_trading.database.models import Trade

logger = get_logger(__name__)
DEFAULT_PATH = "trade_history.json"

def load_history(path: str | Path = DEFAULT_PATH) -> list[dict]:
    """Return list of trade records from *path* or empty list when unavailable."""
    p = Path(path)
    if not p.exists():
        logger.warning(
            "TRADE_HISTORY_SEED_FILE_MISSING", extra={"path": str(p)}
        )
        return []
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        logger.error(
            "TRADE_HISTORY_SEED_PARSE_ERROR", extra={"path": str(p), "error": str(exc)}
        )
        return []
    if not isinstance(data, list):
        logger.error("TRADE_HISTORY_SEED_INVALID_FORMAT", extra={"path": str(p)})
        return []
    return data

def seed_database(trades: Iterable[dict]) -> int:
    """Insert *trades* into the database."""
    initialize_database()
    inserted = 0
    with get_session() as session:
        for record in trades:
            session.add(Trade(**record))
            inserted += 1
        session.commit()
    logger.info("TRADE_HISTORY_SEEDED", extra={"records": inserted})
    return inserted

def main(path: str = DEFAULT_PATH) -> int:
    """CLI entry point for seeding trade history."""
    trades = load_history(path)
    if not trades:
        logger.warning("TRADE_HISTORY_SEED_NO_RECORDS", extra={"path": path})
        return 0
    return seed_database(trades)

def _main(argv: list[str] | None=None) -> int:
    path = argv[0] if argv else DEFAULT_PATH
    main(path)
    return 0

if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))

__all__ = ["main", "_main", "seed_database", "load_history"]
