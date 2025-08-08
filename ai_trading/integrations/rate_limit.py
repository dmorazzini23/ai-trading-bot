"""
Central rate limiter for API orchestration with token bucket algorithm.

Provides process-wide rate limiting with burst capacity, jittered refill,
and async-safe interfaces to prevent 429 errors and API throttling.
"""

import asyncio
import time
import random
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """Token bucket for rate limiting with burst capacity."""
    capacity: int  # Maximum tokens
    refill_rate: float  # Tokens per second
    tokens: float = field(init=False)  # Current token count
    last_refill: float = field(init=False)  # Last refill timestamp
    
    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.time()
    
    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        if elapsed > 0:
            # Add jitter to refill to avoid thundering herd
            jitter_factor = 0.9 + 0.2 * random.random()  # 0.9 to 1.1
            tokens_to_add = elapsed * self.refill_rate * jitter_factor
            
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False otherwise
        """
        self._refill()
        
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def available_tokens(self) -> int:
        """Get current number of available tokens."""
        self._refill()
        return int(self.tokens)
    
    def time_until_available(self, tokens: int = 1) -> float:
        """
        Calculate time until specified tokens are available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Time in seconds until tokens are available
        """
        self._refill()
        
        if self.tokens >= tokens:
            return 0.0
        
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


@dataclass 
class RateLimitConfig:
    """Configuration for a rate-limited route."""
    capacity: int  # Burst capacity
    refill_rate: float  # Tokens per second
    queue_timeout: float = 30.0  # Maximum queue wait time
    enabled: bool = True


class RateLimiter:
    """
    Central rate limiter with per-route token buckets.
    
    Provides async-safe rate limiting with configurable limits per route,
    queuing with deadlines, and metrics collection.
    """
    
    def __init__(self, global_capacity: int = 1000, global_rate: float = 100.0):
        """
        Initialize rate limiter.
        
        Args:
            global_capacity: Global burst capacity across all routes
            global_rate: Global refill rate (tokens per second)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Global rate limiting
        self._global_bucket = TokenBucket(global_capacity, global_rate)
        
        # Per-route buckets
        self._route_buckets: Dict[str, TokenBucket] = {}
        self._route_configs: Dict[str, RateLimitConfig] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        
        # Metrics
        self._metrics = defaultdict(lambda: {
            'requests': 0,
            'denials': 0,
            'wait_time_total': 0.0,
            'max_wait_time': 0.0
        })
        
        # Default route configurations
        self._setup_default_routes()
    
    def _setup_default_routes(self) -> None:
        """Setup default rate limit configurations."""
        # AI-AGENT-REF: Default API route configurations
        default_routes = {
            'orders': RateLimitConfig(capacity=50, refill_rate=10.0),  # 10 orders/sec, burst 50
            'bars': RateLimitConfig(capacity=200, refill_rate=50.0),   # 50 requests/sec, burst 200
            'quotes': RateLimitConfig(capacity=100, refill_rate=20.0), # 20 quotes/sec, burst 100
            'positions': RateLimitConfig(capacity=20, refill_rate=5.0), # 5 requests/sec, burst 20
            'account': RateLimitConfig(capacity=10, refill_rate=2.0),   # 2 requests/sec, burst 10
        }
        
        for route, config in default_routes.items():
            self.configure_route(route, config)
    
    def configure_route(self, route: str, config: RateLimitConfig) -> None:
        """
        Configure rate limiting for a route.
        
        Args:
            route: Route identifier (e.g., 'orders', 'bars')
            config: Rate limit configuration
        """
        with self._lock:
            self._route_configs[route] = config
            self._route_buckets[route] = TokenBucket(config.capacity, config.refill_rate)
        
        self.logger.info(f"Configured rate limit for {route}: "
                        f"{config.capacity} capacity, {config.refill_rate}/sec refill")
    
    def _check_limits(self, route: str, tokens: int = 1) -> Tuple[bool, float]:
        """
        Check if request can proceed immediately.
        
        Args:
            route: Route identifier
            tokens: Number of tokens to consume
            
        Returns:
            Tuple of (can_proceed, wait_time)
        """
        config = self._route_configs.get(route)
        if config is None or not config.enabled:
            return True, 0.0
        
        # Check global limit first
        if not self._global_bucket.consume(tokens):
            global_wait = self._global_bucket.time_until_available(tokens)
            return False, global_wait
        
        # Check route-specific limit
        route_bucket = self._route_buckets.get(route)
        if route_bucket is None:
            return True, 0.0
        
        if not route_bucket.consume(tokens):
            route_wait = route_bucket.time_until_available(tokens)
            # Return global tokens since we couldn't proceed
            self._global_bucket.tokens += tokens
            return False, route_wait
        
        return True, 0.0
    
    async def acquire(self, route: str, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens for a route (async version).
        
        Args:
            route: Route identifier
            tokens: Number of tokens to acquire
            timeout: Maximum wait time (uses route default if None)
            
        Returns:
            True if tokens acquired, False if timed out
        """
        config = self._route_configs.get(route)
        if config is None or not config.enabled:
            return True
        
        if timeout is None:
            timeout = config.queue_timeout
        
        start_time = time.time()
        
        async with self._async_lock:
            # Update metrics
            self._metrics[route]['requests'] += 1
            
            # Try immediate acquisition
            can_proceed, wait_time = self._check_limits(route, tokens)
            
            if can_proceed:
                return True
            
            # Check if wait time exceeds timeout
            if wait_time > timeout:
                self._metrics[route]['denials'] += 1
                self.logger.warning(f"Rate limit denial for {route}: "
                                  f"wait time {wait_time:.2f}s > timeout {timeout}s")
                return False
            
            # Wait for tokens to become available
            self.logger.debug(f"Rate limiting {route}: waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
            
            # Try again after waiting
            can_proceed, additional_wait = self._check_limits(route, tokens)
            
            actual_wait = time.time() - start_time
            self._metrics[route]['wait_time_total'] += actual_wait
            self._metrics[route]['max_wait_time'] = max(
                self._metrics[route]['max_wait_time'], actual_wait
            )
            
            if can_proceed:
                return True
            else:
                self._metrics[route]['denials'] += 1
                self.logger.warning(f"Rate limit still exceeded for {route} after waiting")
                return False
    
    def acquire_sync(self, route: str, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens for a route (sync version).
        
        Args:
            route: Route identifier  
            tokens: Number of tokens to acquire
            timeout: Maximum wait time (uses route default if None)
            
        Returns:
            True if tokens acquired, False if timed out
        """
        config = self._route_configs.get(route)
        if config is None or not config.enabled:
            return True
        
        if timeout is None:
            timeout = config.queue_timeout
        
        start_time = time.time()
        
        with self._lock:
            # Update metrics
            self._metrics[route]['requests'] += 1
            
            # Try immediate acquisition
            can_proceed, wait_time = self._check_limits(route, tokens)
            
            if can_proceed:
                return True
            
            # Check if wait time exceeds timeout
            if wait_time > timeout:
                self._metrics[route]['denials'] += 1
                self.logger.warning(f"Rate limit denial for {route}: "
                                  f"wait time {wait_time:.2f}s > timeout {timeout}s")
                return False
        
        # Wait outside the lock
        self.logger.debug(f"Rate limiting {route}: waiting {wait_time:.2f}s")
        time.sleep(wait_time)
        
        # Try again after waiting
        with self._lock:
            can_proceed, additional_wait = self._check_limits(route, tokens)
            
            actual_wait = time.time() - start_time
            self._metrics[route]['wait_time_total'] += actual_wait
            self._metrics[route]['max_wait_time'] = max(
                self._metrics[route]['max_wait_time'], actual_wait
            )
            
            if can_proceed:
                return True
            else:
                self._metrics[route]['denials'] += 1
                self.logger.warning(f"Rate limit still exceeded for {route} after waiting")
                return False
    
    @asynccontextmanager
    async def limit(self, route: str, tokens: int = 1, timeout: Optional[float] = None):
        """
        Async context manager for rate limiting.
        
        Args:
            route: Route identifier
            tokens: Number of tokens to acquire
            timeout: Maximum wait time
            
        Yields:
            None if acquired, raises RuntimeError if denied
        """
        acquired = await self.acquire(route, tokens, timeout)
        if not acquired:
            raise RuntimeError(f"Rate limit exceeded for route: {route}")
        
        try:
            yield
        finally:
            # Context manager cleanup if needed
            pass
    
    def get_status(self, route: Optional[str] = None) -> Dict[str, Any]:
        """
        Get rate limiter status.
        
        Args:
            route: Specific route to check (None for all)
            
        Returns:
            Dictionary with status information
        """
        with self._lock:
            if route is not None:
                bucket = self._route_buckets.get(route)
                config = self._route_configs.get(route)
                metrics = self._metrics.get(route, {})
                
                if bucket is None:
                    return {'error': f'Route {route} not found'}
                
                return {
                    'route': route,
                    'available_tokens': bucket.available_tokens(),
                    'capacity': config.capacity if config else 0,
                    'refill_rate': config.refill_rate if config else 0,
                    'enabled': config.enabled if config else False,
                    'metrics': dict(metrics)
                }
            
            # Return status for all routes
            status = {
                'global': {
                    'available_tokens': self._global_bucket.available_tokens(),
                    'capacity': self._global_bucket.capacity,
                    'refill_rate': self._global_bucket.refill_rate
                },
                'routes': {}
            }
            
            for route_name in self._route_buckets:
                route_status = self.get_status(route_name)
                if 'error' not in route_status:
                    status['routes'][route_name] = route_status
            
            return status
    
    def reset_metrics(self, route: Optional[str] = None) -> None:
        """Reset metrics for a route or all routes."""
        with self._lock:
            if route is not None:
                if route in self._metrics:
                    self._metrics[route] = {
                        'requests': 0,
                        'denials': 0, 
                        'wait_time_total': 0.0,
                        'max_wait_time': 0.0
                    }
            else:
                self._metrics.clear()


# Global rate limiter instance
_global_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create global rate limiter instance."""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = RateLimiter()
    return _global_rate_limiter


# Convenience functions
async def rate_limit_async(route: str, tokens: int = 1, timeout: Optional[float] = None) -> bool:
    """Convenience function for async rate limiting."""
    limiter = get_rate_limiter()
    return await limiter.acquire(route, tokens, timeout)


def rate_limit_sync(route: str, tokens: int = 1, timeout: Optional[float] = None) -> bool:
    """Convenience function for sync rate limiting."""
    limiter = get_rate_limiter()
    return limiter.acquire_sync(route, tokens, timeout)