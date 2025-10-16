"""Rate Limiter for API Calls.

This module implements token bucket rate limiting to prevent
API throttling and improve data fetch reliability.
"""

import logging
import time
from threading import Lock
from typing import Optional
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket rate limiter implementation."""
    
    def __init__(
        self,
        rate: float,  # tokens per second
        capacity: int,  # maximum tokens
        name: str = "default"
    ):
        self.rate = rate
        self.capacity = capacity
        self.name = name
        self.tokens = float(capacity)
        self.last_update = time.time()
        self.lock = Lock()
    
    def consume(self, tokens: int = 1, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Consume tokens from the bucket.
        
        Args:
            tokens: Number of tokens to consume
            block: Whether to block until tokens are available
            timeout: Maximum time to wait (None = wait forever)
        
        Returns:
            True if tokens were consumed, False otherwise
        """
        
        start_time = time.time()
        
        while True:
            with self.lock:
                self._refill()
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    logger.debug(
                        "RATE_LIMIT_TOKEN_CONSUMED",
                        extra={
                            "limiter": self.name,
                            "tokens_consumed": tokens,
                            "tokens_remaining": round(self.tokens, 2)
                        }
                    )
                    return True
                
                if not block:
                    logger.warning(
                        "RATE_LIMIT_EXCEEDED",
                        extra={
                            "limiter": self.name,
                            "tokens_needed": tokens,
                            "tokens_available": round(self.tokens, 2)
                        }
                    )
                    return False
                
                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate
                
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed + wait_time > timeout:
                        logger.warning(
                            "RATE_LIMIT_TIMEOUT",
                            extra={
                                "limiter": self.name,
                                "timeout": timeout,
                                "elapsed": round(elapsed, 2)
                            }
                        )
                        return False
            
            # Wait outside the lock
            logger.info(
                "RATE_LIMIT_WAITING",
                extra={
                    "limiter": self.name,
                    "wait_seconds": round(wait_time, 2)
                }
            )
            time.sleep(wait_time)
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        
        now = time.time()
        elapsed = now - self.last_update
        
        # Add tokens based on time elapsed
        new_tokens = elapsed * self.rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_update = now
    
    def get_status(self) -> dict:
        """Get current rate limiter status."""
        
        with self.lock:
            self._refill()
            return {
                "name": self.name,
                "tokens_available": round(self.tokens, 2),
                "capacity": self.capacity,
                "rate_per_second": self.rate,
                "utilization_pct": round((1 - self.tokens / self.capacity) * 100, 1)
            }


class RateLimiterManager:
    """Manages multiple rate limiters for different API endpoints."""
    
    def __init__(self):
        self.limiters = {}
        self.lock = Lock()
    
    def get_or_create(
        self,
        name: str,
        rate: float = 10.0,  # 10 requests per second default
        capacity: int = 100
    ) -> TokenBucket:
        """Get existing rate limiter or create a new one."""
        
        with self.lock:
            if name not in self.limiters:
                self.limiters[name] = TokenBucket(rate, capacity, name)
                logger.info(
                    "RATE_LIMITER_CREATED",
                    extra={
                        "name": name,
                        "rate": rate,
                        "capacity": capacity
                    }
                )
            
            return self.limiters[name]
    
    def get_all_status(self) -> dict:
        """Get status of all rate limiters."""
        
        return {
            name: limiter.get_status()
            for name, limiter in self.limiters.items()
        }


# Global rate limiter manager
_rate_limiter_manager = RateLimiterManager()


def get_rate_limiter(
    name: str,
    rate: float = 10.0,
    capacity: int = 100
) -> TokenBucket:
    """Get or create a rate limiter."""
    return _rate_limiter_manager.get_or_create(name, rate, capacity)


def get_all_rate_limiter_status() -> dict:
    """Get status of all rate limiters."""
    return _rate_limiter_manager.get_all_status()


class SlidingWindowRateLimiter:
    """Sliding window rate limiter for more precise control."""
    
    def __init__(self, max_requests: int, window_seconds: int, name: str = "default"):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.name = name
        self.requests = deque()
        self.lock = Lock()
    
    def can_proceed(self) -> tuple[bool, float]:
        """
        Check if a request can proceed.
        
        Returns:
            (can_proceed, wait_time) tuple
        """
        
        with self.lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.window_seconds)
            
            # Remove old requests outside the window
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return (True, 0.0)
            
            # Calculate wait time until oldest request expires
            oldest = self.requests[0]
            wait_time = (oldest + timedelta(seconds=self.window_seconds) - now).total_seconds()
            
            return (False, max(0, wait_time))
    
    def wait_if_needed(self, timeout: Optional[float] = None) -> bool:
        """
        Wait if rate limit is exceeded.
        
        Returns:
            True if proceeded, False if timeout
        """
        
        start_time = time.time()
        
        while True:
            can_proceed, wait_time = self.can_proceed()
            
            if can_proceed:
                return True
            
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed + wait_time > timeout:
                    return False
            
            logger.info(
                "SLIDING_WINDOW_WAIT",
                extra={
                    "limiter": self.name,
                    "wait_seconds": round(wait_time, 2)
                }
            )
            time.sleep(wait_time)

