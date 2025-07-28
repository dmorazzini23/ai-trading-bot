# Minimal numpy mock for testing
class ndarray:
    pass

class MockRandom:
    def seed(self, value):
        pass

def array(data):
    return ndarray()

def zeros(shape):
    return ndarray()

# Add random module mock
random = MockRandom()

def __getattr__(name):
    if name == 'random':
        return random
    return lambda *args, **kwargs: None
