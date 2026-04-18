from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def has_default(name: str) -> bool:
    return files(__name__).joinpath(name).is_file()


def load_default_json(name: str) -> Any:
    return json.loads(files(__name__).joinpath(name).read_text(encoding="utf-8"))
