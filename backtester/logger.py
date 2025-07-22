import time

class MetricsLogger:
    def __init__(self, name: str):
        self.name = name
        self.start = time.time()

    def log(self, **metrics):
        # implement whatever no-op or console logging you need
        print(f"[{self.name}] ", metrics)

    def flush(self):
        pass
