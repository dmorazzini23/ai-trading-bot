# Mock for finnhub
class Client:
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return lambda *args, **kwargs: {}

def __getattr__(name):
    return lambda *args, **kwargs: None
