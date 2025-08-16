class _MockFinBERT:
    def __call__(self, text: str) -> dict:
        return {"pos": 0.34, "neg": 0.33, "neu": 0.33}
