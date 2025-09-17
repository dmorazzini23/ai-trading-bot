class MockConfig:
    def __init__(self, **kwargs):
        self.position_size_min_usd = 25.0
        self.__dict__.update(kwargs)
