# Mock for tenacity
def retry(*args, **kwargs):
    def decorator(func):
        return func
    return decorator

class MockWait:
    def __add__(self, other):
        return self

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
