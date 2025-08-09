"""
Order idempotency module to prevent duplicate orders on retries.

Provides TTL cache keyed by (symbol, side, qty, intent_ts_bucket) to ensure
orders are idempotent across retry attempts.
"""

import hashlib
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timezone
from cachetools import TTLCache
import threading

from ai_trading.core.interfaces import OrderSide


@dataclass
class IdempotencyKey:
    """Represents a unique idempotency key for order deduplication."""
    symbol: str
    side: OrderSide
    quantity: float
    intent_bucket: int  # Timestamp bucketed to nearest minute/interval
    
    def __str__(self) -> str:
        return f"{self.symbol}_{self.side.value}_{self.quantity}_{self.intent_bucket}"
        
    def hash(self) -> str:
        """Generate hash for this idempotency key."""
        content = f"{self.symbol}{self.side.value}{self.quantity}{self.intent_bucket}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class OrderIdempotencyCache:
    """
    TTL cache for order idempotency with thread-safe operations.
    
    Prevents duplicate order submissions by tracking recently submitted
    orders using (symbol, side, qty, intent_ts_bucket) as the key.
    """
    
    def __init__(self, ttl_seconds: int = 300, max_size: int = 10000):
        """
        Initialize idempotency cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries (default 5 minutes)
            max_size: Maximum cache size
        """
        self._cache: TTLCache = TTLCache(maxsize=max_size, ttl=ttl_seconds)
        self._lock = threading.RLock()
    
    def generate_key(
        self, 
        symbol: str, 
        side: Union[OrderSide, str], 
        quantity: float,
        intent_timestamp: Optional[datetime] = None,
        bucket_minutes: int = 1
    ) -> IdempotencyKey:
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
            intent_timestamp = datetime.now(timezone.utc)
            
        if isinstance(side, str):
            side = OrderSide(side.lower())
            
        # Bucket timestamp to reduce noise from small timing differences
        bucket_ts = int(intent_timestamp.timestamp() // (bucket_minutes * 60))
        
        return IdempotencyKey(
            symbol=symbol.upper(),
            side=side,
            quantity=round(quantity, 6),  # Round to avoid floating point noise
            intent_bucket=bucket_ts
        )
    
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
            self._cache[key.hash()] = {
                'order_id': order_id,
                'submitted_at': datetime.now(timezone.utc),
                'key': key
            }
    
    def get_existing_order(self, key: IdempotencyKey) -> Optional[Dict]:
        """
        Get existing order info if this is a duplicate.
        
        Args:
            key: Idempotency key to lookup
            
        Returns:
            Dict with order_id and submission info if duplicate, None otherwise
        """
        with self._lock:
            return self._cache.get(key.hash())
    
    def clear_expired(self) -> int:
        """
        Clear expired entries and return count cleared.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            initial_size = len(self._cache)
            # TTLCache automatically expires entries on access
            # Force expiration by accessing keys
            list(self._cache.keys())
            return initial_size - len(self._cache)
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self._cache.maxsize,
                'ttl': self._cache.ttl
            }


# Global idempotency cache instance
_global_cache: Optional[OrderIdempotencyCache] = None


def get_idempotency_cache() -> OrderIdempotencyCache:
    """Get or create global idempotency cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = OrderIdempotencyCache()
    return _global_cache


def is_duplicate_order(
    symbol: str,
    side: Union[OrderSide, str], 
    quantity: float,
    intent_timestamp: Optional[datetime] = None
) -> Tuple[bool, Optional[str]]:
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
        return True, existing['order_id'] if existing else None
    
    return False, None


def mark_order_submitted(
    symbol: str,
    side: Union[OrderSide, str],
    quantity: float, 
    order_id: str,
    intent_timestamp: Optional[datetime] = None
) -> None:
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