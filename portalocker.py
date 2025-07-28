# Mock for portalocker
import contextlib

@contextlib.contextmanager
def Lock(*args, **kwargs):
    yield

LOCK_EX = 1
LOCK_NB = 2

def __getattr__(name):
    return lambda *args, **kwargs: None
