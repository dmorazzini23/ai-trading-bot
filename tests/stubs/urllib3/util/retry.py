class Retry:
    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total", 0)
        self.backoff_factor = kwargs.get("backoff_factor", 0)

    @classmethod
    def from_int(cls, retries, redirect=True, status_forcelist=None, **kwargs):
        return cls(total=retries, **kwargs)
