# Mock for pandas_market_calendars
def get_calendar(name):
    class MockCalendar:
        def is_session(self, *args):
            return True
        def session_break_start(self, *args):
            return None
        def session_break_end(self, *args):
            return None
    return MockCalendar()

def __getattr__(name):
    return lambda *args, **kwargs: None
