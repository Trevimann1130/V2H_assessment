# Optimization/framework/engines/Surrogat_model/evaluate/evaluate_run.py
from __future__ import annotations
import joblib
import numpy as np
from typing import Tuple, Dict, Any, List


class SurrogateModel:
    """
    Lädt persistiertes RF-Artefakt und bietet predict(X)->(F,G).
    Erwartet joblib-Objekt mit Keys {"F": list[Model], "G": list[Model], "meta": dict}.
    """

    def __init__(self, models_F, models_G, meta: Dict[str, Any]):
        if not isinstance(models_F, list) or len(models_F) == 0:
            raise ValueError("SurrogateModel: Liste 'F' fehlt/leer.")
        self.models_F = models_F
        self.models_G = models_G or []
        self.meta = meta or {}

    @classmethod
    def load(cls, path: str) -> "SurrogateModel":
        obj = joblib.load(path)
        return cls(obj.get("F", []), obj.get("G", []), obj.get("meta", {}))

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        F = np.column_stack([m.predict(X) for m in self.models_F]) if self.models_F else np.zeros((len(X), 0), float)
        G = np.column_stack([m.predict(X) for m in self.models_G]) if self.models_G else np.zeros((len(X), 0), float)
        return np.asarray(F, float), np.asarray(G, float)


def load_surrogate(path: str) -> SurrogateModel:
    return SurrogateModel.load(path)

def load_model(path: str) -> SurrogateModel:
    return SurrogateModel.load(path)

def load(path: str) -> Dict[str, Any]:
    return joblib.load(path)
