"""Minimal stub for pydantic-settings.
"""
from typing import Any

class SettingsConfigDict(dict):
    pass

class BaseSettings:
    model_config: SettingsConfigDict | None = None
    def __init__(self, **data: Any):
        for k, v in data.items():
            setattr(self, k, v)

