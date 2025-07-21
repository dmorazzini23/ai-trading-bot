from __future__ import annotations
from typing import Optional, Union
import os
import joblib
import logging
import numpy as np

logger = logging.getLogger(__name__)
MODEL_PATH = "model.pkl"

class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        probs = [m.predict_proba(X) for m in self.models]
        return np.mean(probs, axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def load_model(path: str = MODEL_PATH) -> Optional[Union[dict, EnsembleModel]]:
    # return None if file doesn’t exist
    if not os.path.exists(path):
        return None
    try:
        obj = joblib.load(path)
        # single-model PKL should be a dict
        if isinstance(obj, dict):
            logger.info("MODEL_LOADED")
            return obj
        # ensemble files are lists
        if isinstance(obj, list):
            model = EnsembleModel(obj)
            logger.info("MODEL_LOADED")
            return model
        # fallback
        logger.info("MODEL_LOADED")
        return obj
    except Exception as e:
        logger.exception("MODEL_LOAD_FAILED: %s", e)
        return None
