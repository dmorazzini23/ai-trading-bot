"""Examples to assist with Pydantic v2 migration."""

from pydantic import BaseModel
from pydantic import field_validator, Field


class ExampleModel(BaseModel):
    """Simple model demonstrating v2-style validators."""

    name: str = Field(...)

    @field_validator('name')
    @classmethod
    def _name_must_not_be_empty(cls, value: str) -> str:
        if not value:
            raise ValueError('name must not be empty')
        return value


__all__ = ['ExampleModel']
