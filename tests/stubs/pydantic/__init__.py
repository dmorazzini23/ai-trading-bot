"""Minimal pydantic stub for tests.
Provides only interfaces used in ai_trading.settings.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

class ValidationError(Exception):
    pass

class SecretStr:
    def __init__(self, value: str):
        self._value = value
    def get_secret_value(self) -> str:
        return self._value
    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "SecretStr(****)"

@dataclass
class FieldInfo:
    default: Any = None

def Field(default: Any = None, *_, **__):
    return default

class AliasChoices(tuple):
    def __new__(cls, *names: str):
        return super().__new__(cls, names)

# Decorators used in Settings; they simply return the function unchanged.
def computed_field(*_, **__):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func
    return decorator

def field_validator(*_, **__):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func
    return decorator

class BaseModel:
    def __init__(self, **data: Any):
        for k, v in data.items():
            setattr(self, k, v)
    def model_dump(self) -> dict[str, Any]:  # pragma: no cover
        return self.__dict__.copy()
