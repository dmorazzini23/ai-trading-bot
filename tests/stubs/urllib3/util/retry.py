class Retry:
    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total", 0)
        self.backoff_factor = kwargs.get("backoff_factor", 0)
