# Minimal pandas mock for testing
import datetime

class Timestamp:
    def __init__(self, *args, **kwargs):
        pass
    def __getattr__(self, name):
        return lambda *args, **kwargs: self
    def __call__(self, *args, **kwargs):
        return self

class DataFrame:
    def __init__(self, data=None, index=None, columns=None):
        pass
    def __getattr__(self, name):
        return lambda *args, **kwargs: self
    def __getitem__(self, key):
        return self
    def __setitem__(self, key, value):
        pass

class Series:
    def __init__(self, data=None, index=None):
        pass
    def __getattr__(self, name):
        return lambda *args, **kwargs: self

def read_csv(*args, **kwargs):
    return DataFrame()

def read_parquet(*args, **kwargs):
    return DataFrame()

def concat(*args, **kwargs):
    return DataFrame()

def to_datetime(*args, **kwargs):
    return datetime.datetime.now()

def __getattr__(name):
    if name == 'DataFrame':
        return DataFrame
    elif name == 'Series':
        return Series
    elif name == 'Timestamp':
        return Timestamp
    return lambda *args, **kwargs: None

# Make the module look like pandas
pd = type('MockPandas', (), {
    'DataFrame': DataFrame,
    'Series': Series,
    'Timestamp': Timestamp,
    'read_csv': read_csv,
    'read_parquet': read_parquet,
    'concat': concat,
    'to_datetime': to_datetime,
})()
