"""
Circuit breaker patterns and external service reliability utilities.
Implements circuit breaker pattern for external API calls to improve system resilience.
"""

import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass
from functools import wraps

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5      # Number of failures before opening
    recovery_timeout: int = 60      # Seconds before trying half-open
    success_threshold: int = 3      # Successes needed to close from half-open
    timeout: int = 30              # Request timeout in seconds


class CircuitBreaker:
    """
    Circuit breaker implementation for external service calls.
    
    Prevents cascading failures by failing fast when a service is down,
    and periodically checking if the service has recovered.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.last_attempt_time = 0
        
    def _record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close_circuit()
        elif self.state == CircuitState.CLOSED:
            self.success_count += 1
            
        logger.debug(f"Circuit breaker {self.name}: Success recorded (state: {self.state.value})")
    
    def _record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._open_circuit()
        elif self.state == CircuitState.HALF_OPEN:
            self._open_circuit()
            
        logger.warning(f"Circuit breaker {self.name}: Failure recorded (count: {self.failure_count}, state: {self.state.value})")
    
    def _open_circuit(self):
        """Open the circuit breaker."""
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
        logger.warning(f"Circuit breaker {self.name}: OPENED after {self.failure_count} failures")
    
    def _close_circuit(self):
        """Close the circuit breaker."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker {self.name}: CLOSED - service recovered")
    
    def _should_attempt_call(self) -> bool:
        """Check if we should attempt the call."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name}: Moving to HALF_OPEN state")
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        return False
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result if successful
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception from function
        """
        if not self._should_attempt_call():
            raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        self.last_attempt_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception:
            self._record_failure()
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "last_attempt_time": self.last_attempt_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold,
                "timeout": self.config.timeout,
            }
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator for applying circuit breaker pattern to functions.
    
    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration
        
    Usage:
        @circuit_breaker("external_api")
        def call_external_api():
            # API call code
            pass
    """
    # Global registry of circuit breakers
    if not hasattr(circuit_breaker, '_breakers'):
        circuit_breaker._breakers = {}
    
    if name not in circuit_breaker._breakers:
        circuit_breaker._breakers[name] = CircuitBreaker(name, config)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            breaker = circuit_breaker._breakers[name]
            return breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


def get_circuit_breaker_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all circuit breakers."""
    if not hasattr(circuit_breaker, '_breakers'):
        return {}
    
    return {name: breaker.get_stats() 
            for name, breaker in circuit_breaker._breakers.items()}


def reset_circuit_breaker(name: str) -> bool:
    """Reset a specific circuit breaker to closed state."""
    if not hasattr(circuit_breaker, '_breakers'):
        return False
    
    if name in circuit_breaker._breakers:
        breaker = circuit_breaker._breakers[name]
        breaker._close_circuit()
        logger.info(f"Circuit breaker {name} manually reset to CLOSED state")
        return True
    return False


# AI-AGENT-REF: Enhanced retry decorator with exponential backoff
def enhanced_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_factor: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
):
    """
    Enhanced retry decorator with exponential backoff and jitter.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_factor: Exponential backoff factor
        jitter: Whether to add random jitter to delays
        exceptions: Tuple of exceptions to retry on
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import random
            
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        # Last attempt, re-raise the exception
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_factor ** attempt), max_delay)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    logger.debug(f"Retry attempt {attempt + 1}/{max_attempts} for {func.__name__} after {delay:.2f}s delay")
                    time.sleep(delay)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


# AI-AGENT-REF: Health check utility for external services
def health_check_service(name: str, check_func: Callable[[], bool], timeout: int = 10) -> bool:
    """
    Perform a health check on an external service.
    
    Args:
        name: Service name for logging
        check_func: Function that returns True if service is healthy
        timeout: Timeout for the health check
        
    Returns:
        True if service is healthy, False otherwise
    """
    try:
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Health check for {name} timed out after {timeout}s")
        
        # Set timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        try:
            result = check_func()
            logger.debug(f"Health check for {name}: {'HEALTHY' if result else 'UNHEALTHY'}")
            return result
        finally:
            signal.alarm(0)  # Disable alarm
            
    except Exception as e:
        logger.warning(f"Health check for {name} failed: {e}")
        return False