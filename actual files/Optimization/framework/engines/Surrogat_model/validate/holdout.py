# Optimization/framework/engines/Surrogat_model/validate/holdout.py
from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import r2_score

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    if y_true.size == 0:
        return float("nan")
    diff = y_true - y_pred
    return float(np.sqrt(np.mean(diff * diff)))

def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))

def _rel_mae_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).ravel()
    mae = _mae(y_true, y_pred)
    denom = float(np.mean(np.abs(y_true)))
    if denom <= 1e-12:
        return float("nan")
    return float(mae / denom * 100.0)

def metrics_by_column(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> Dict[str, List[float]]:
    """
    Liefert Metriken je Spalte:
      {
        "targets":[...],
        "r2":[...],
        "rmse":[...],
        "mae":[...],
        "rel_mae_percent":[...],
      }
    """
    Y_true = np.asarray(Y_true, float)
    Y_pred = np.asarray(Y_pred, float)
    assert Y_true.shape == Y_pred.shape, "[holdout] Shapes mismatch."

    n_cols = Y_true.shape[1]
    names = target_names if (target_names and len(target_names) == n_cols) else [f"col_{i}" for i in range(n_cols)]

    r2_list, rmse_list, mae_list, rel_list = [], [], [], []
    for j in range(n_cols):
        yt = Y_true[:, j]
        yp = Y_pred[:, j]
        # robust bei Konstanten/NaNs
        try:
            r2 = float(r2_score(yt, yp)) if np.std(yt) > 1e-12 else float("nan")
        except Exception:
            r2 = float("nan")
        r2_list.append(r2)
        rmse_list.append(_rmse(yt, yp))
        mae_list.append(_mae(yt, yp))
        rel_list.append(_rel_mae_percent(yt, yp))

    return {
        "targets": names,
        "r2": r2_list,
        "rmse": rmse_list,
        "mae": mae_list,
        "rel_mae_percent": rel_list,
    }
