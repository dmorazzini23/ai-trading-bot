# Minimal numpy mock for testing
class ndarray:
    pass

def array(data):
    return ndarray()

def zeros(shape):
    return ndarray()

def __getattr__(name):
    return lambda *args, **kwargs: None
