import os

def get_model_registry_dir(base: str) -> str:
    path = os.path.join(base, "custom_models")
    os.makedirs(path, exist_ok=True)
    return path

def get_walkforward_artifacts_dir(base: str) -> str:
    path = os.path.join(base, "walkforward_artifacts")
    os.makedirs(path, exist_ok=True)
    return path

__all__ = ["get_model_registry_dir", "get_walkforward_artifacts_dir"]
