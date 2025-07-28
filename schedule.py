# Mock for schedule
def every(*args):
    class MockSchedule:
        def minute(self):
            return self
        def minutes(self):
            return self
        def hour(self):
            return self
        def hours(self):
            return self
        def day(self):
            return self
        def do(self, func, *args):
            return self
    return MockSchedule()

def run_pending():
    pass

def __getattr__(name):
    return lambda *args, **kwargs: None
