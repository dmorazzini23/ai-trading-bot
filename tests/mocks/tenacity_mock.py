# Mock for tenacity
def retry(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

class RetryError(Exception):
    """Mock RetryError exception for testing."""
    pass

def stop_after_attempt(*args):
    return None

def wait_exponential(*args, **kwargs):
    return MockWait()

def wait_random(*args, **kwargs):
    return MockWait()

def retry_if_exception_type(*args):
    return None

def __getattr__(name):
    return lambda *args, **kwargs: lambda f: f
