"""
Order idempotency module to prevent duplicate orders on retries.

Provides TTL cache keyed by (symbol, side, qty, intent_ts_bucket) to ensure
orders are idempotent across retry attempts.
"""
import hashlib
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

try:
    from cachetools import TTLCache  # type: ignore
except ImportError:  # pragma: no cover - exercised via explicit fallback tests
    import logging
    from collections import OrderedDict
    from ai_trading.utils.time import monotonic_time

    logger = logging.getLogger(__name__)
    logger.warning(
        "cachetools not available; using fallback TTL cache implementation (degraded idempotency cache performance)"
    )

    class TTLCache:  # type: ignore[override]
        """Lightweight TTL cache fallback with basic eviction semantics."""

        def __init__(self, maxsize: int, ttl: float) -> None:
            self.maxsize = maxsize
            self.ttl = ttl
            self._store: OrderedDict[str, tuple[object, float]] = OrderedDict()

        def _expire(self, *, now: float | None = None) -> None:
            """Remove expired entries in-place."""

            if not self._store:
                return

            current_time = monotonic_time() if now is None else now
            expired_keys = [key for key, (_, exp) in list(self._store.items()) if exp <= current_time]
            for key in expired_keys:
                self._store.pop(key, None)

        def __contains__(self, key: str) -> bool:
            now = monotonic_time()
            self._expire(now=now)

            item = self._store.get(key)
            if item is None:
                return False

            # touch to maintain recency semantics similar to OrderedDict move-to-end
            self._store.move_to_end(key)
            return True

        def __getitem__(self, key: str) -> object:
            now = monotonic_time()
            self._expire(now=now)

            value, _ = self._store[key]
            self._store.move_to_end(key)
            return value

        def __setitem__(self, key: str, value: object) -> None:
            now = monotonic_time()
            self._expire(now=now)

            if key in self._store:
                self._store.pop(key, None)
            elif self.maxsize and len(self._store) >= self.maxsize:
                self._store.popitem(last=False)

            self._store[key] = (value, now + self.ttl)

        def get(self, key: str, default: object | None=None) -> object | None:
            now = monotonic_time()
            self._expire(now=now)

            item = self._store.get(key)
            if item is None:
                return default

            value, _ = item
            self._store.move_to_end(key)
            return value

        def keys(self):  # noqa: D401 - mimic cachetools API
            """Return current cache keys after expiring stale entries."""

            self._expire()
            return list(self._store.keys())

        def __len__(self) -> int:
            return len(self._store)

        def __repr__(self) -> str:  # pragma: no cover - diagnostic helper
            return f"TTLCache(maxsize={self.maxsize}, ttl={self.ttl}, size={len(self)})"

from ai_trading.core.interfaces import OrderSide

@dataclass
class IdempotencyKey:
    """Represents a unique idempotency key for order deduplication."""
    symbol: str
    side: OrderSide
    quantity: float
    intent_bucket: int

    def __str__(self) -> str:
        return f'{self.symbol}_{self.side.value}_{self.quantity}_{self.intent_bucket}'

    def hash(self) -> str:
        """Generate hash for this idempotency key."""
        content = f'{self.symbol}{self.side.value}{self.quantity}{self.intent_bucket}'
        return hashlib.sha256(content.encode()).hexdigest()[:16]

class OrderIdempotencyCache:
    """
    TTL cache for order idempotency with thread-safe operations.

    Prevents duplicate order submissions by tracking recently submitted
    orders using (symbol, side, qty, intent_ts_bucket) as the key.
    """

    def __init__(self, ttl_seconds: int=300, max_size: int=10000):
        """
        Initialize idempotency cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default 5 minutes)
            max_size: Maximum cache size
        """
        self._cache: TTLCache = TTLCache(maxsize=max_size, ttl=ttl_seconds)
        self._lock = threading.RLock()

    def generate_key(self, symbol: str, side: OrderSide | str, quantity: float, intent_timestamp: datetime | None=None, bucket_minutes: int=1) -> IdempotencyKey:
        """
        Generate idempotency key for order parameters.

        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            intent_timestamp: When order was intended (defaults to now)
            bucket_minutes: Minute buckets for intent timestamp

        Returns:
            IdempotencyKey for these order parameters
        """
        if intent_timestamp is None:
            intent_timestamp = datetime.now(UTC)
        if isinstance(side, str):
            side = OrderSide(side.lower())
        bucket_ts = int(intent_timestamp.timestamp() // (bucket_minutes * 60))
        return IdempotencyKey(symbol=symbol.upper(), side=side, quantity=round(quantity, 6), intent_bucket=bucket_ts)

    def is_duplicate(self, key: IdempotencyKey) -> bool:
        """
        Check if this order key represents a duplicate submission.

        Args:
            key: Idempotency key to check

        Returns:
            True if this is a duplicate order
        """
        with self._lock:
            return key.hash() in self._cache

    def mark_submitted(self, key: IdempotencyKey, order_id: str) -> None:
        """
        Mark an order as submitted to prevent future duplicates.

        Args:
            key: Idempotency key for the order
            order_id: Broker order ID that was submitted
        """
        with self._lock:
            self._cache[key.hash()] = {'order_id': order_id, 'submitted_at': datetime.now(UTC), 'key': key}

    def check_and_mark_submitted(
        self,
        key: IdempotencyKey,
        order_id: str,
    ) -> tuple[bool, str | None]:
        """
        Atomically check duplicate status and mark a key as submitted.

        Returns:
            Tuple of (is_duplicate, existing_order_id)
        """
        with self._lock:
            key_hash = key.hash()
            existing = self._cache.get(key_hash)
            if isinstance(existing, dict):
                existing_order_id = existing.get("order_id")
                return (True, str(existing_order_id) if existing_order_id is not None else None)
            self._cache[key_hash] = {
                "order_id": order_id,
                "submitted_at": datetime.now(UTC),
                "key": key,
            }
            return (False, None)

    def get_existing_order(self, key: IdempotencyKey) -> dict | None:
        """
        Get existing order info if this is a duplicate.

        Args:
            key: Idempotency key to lookup

        Returns:
            Dict with order_id and submission info if duplicate, None otherwise
        """
        with self._lock:
            key_hash = key.hash()
            if key_hash not in self._cache:
                return None
            entry = self._cache.get(key_hash)
            if entry is None:
                return None
            submitted_at = entry.get("submitted_at") if isinstance(entry, dict) else None
            ttl_seconds = getattr(self._cache, "ttl", None)
            if (
                isinstance(submitted_at, datetime)
                and isinstance(ttl_seconds, (int, float))
                and ttl_seconds > 0
            ):
                age = (datetime.now(UTC) - submitted_at).total_seconds()
                if age > ttl_seconds:
                    try:
                        del self._cache[key_hash]
                    except Exception:
                        try:
                            self._cache.pop(key_hash, None)  # type: ignore[attr-defined]
                        except Exception:
                            pass
                    return None
            return entry

    def clear_expired(self) -> int:
        """
        Clear expired entries and return count cleared.

        Returns:
            Number of entries cleared
        """
        with self._lock:
            initial_size = len(self._cache)
            list(self._cache.keys())
            return initial_size - len(self._cache)

    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            return {'size': len(self._cache), 'max_size': self._cache.maxsize, 'ttl': self._cache.ttl}
_global_cache: OrderIdempotencyCache | None = None
_global_cache_lock = threading.Lock()

def get_idempotency_cache() -> OrderIdempotencyCache:
    """Get or create global idempotency cache instance."""
    global _global_cache
    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:
                _global_cache = OrderIdempotencyCache()
    return _global_cache

def is_duplicate_order(symbol: str, side: OrderSide | str, quantity: float, intent_timestamp: datetime | None=None) -> tuple[bool, str | None]:
    """
    Check if an order would be a duplicate submission.

    Args:
        symbol: Trading symbol
        side: Order side
        quantity: Order quantity
        intent_timestamp: When order was intended

    Returns:
        Tuple of (is_duplicate, existing_order_id)
    """
    cache = get_idempotency_cache()
    key = cache.generate_key(symbol, side, quantity, intent_timestamp)
    if cache.is_duplicate(key):
        existing = cache.get_existing_order(key)
        return (True, existing['order_id'] if existing else None)
    return (False, None)

def mark_order_submitted(symbol: str, side: OrderSide | str, quantity: float, order_id: str, intent_timestamp: datetime | None=None) -> None:
    """
    Mark an order as submitted to prevent future duplicates.

    Args:
        symbol: Trading symbol
        side: Order side
        quantity: Order quantity
        order_id: Broker order ID
        intent_timestamp: When order was intended
    """
    cache = get_idempotency_cache()
    key = cache.generate_key(symbol, side, quantity, intent_timestamp)
    cache.mark_submitted(key, order_id)
