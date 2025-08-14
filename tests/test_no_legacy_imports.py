import importlib
import pytest


@pytest.mark.parametrize("modname", ["sentiment"])
def test_legacy_modules_not_importable(modname):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(modname)
