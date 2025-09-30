# -*- coding: utf-8 -*-
from __future__ import annotations
import os, numpy as np, pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---- Metriken ---------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    rel_mae = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100.0)
    return dict(R2=r2, RMSE=rmse, MAE=mae, RelMAE_percent=rel_mae)

def save_metrics_table(rows: list[dict], out_csv: str):
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv

# ---- Feature-Sampling & Rastern ---------------------------------
import numpy as np
def lhs(n: int, low_high: list[tuple[float,float]], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = []
    for lo, hi in low_high:
        cut = np.linspace(0, 1, n+1)
        u = rng.uniform(cut[:-1], cut[1:])
        rng.shuffle(u)
        cols.append(lo + u*(hi-lo))
    return np.column_stack(cols)

def snap_to_steps(X: np.ndarray, steps: list[float]) -> np.ndarray:
    Xs = X.copy()
    for j, st in enumerate(steps):
        if st and st > 0:
            Xs[:, j] = np.round(Xs[:, j] / st) * st
    return np.unique(Xs, axis=0)

# ---- Zielgrößen je nach V2H -------------------------------------
def expected_targets(use_v2h: bool) -> list[str]:
    base = ["E_import_kWh", "E_export_kWh", "E_bess_throughput_kWh"]
    if use_v2h:
        base += ["E_ev_charged_kWh", "E_ev_discharged_kWh"]
    return base

# ---- Surrogates laden -------------------------------------------
from joblib import load
def load_surrogates(models_dir: str, location: str, tag: str, use_v2h: bool) -> dict:
    mdir = os.path.join(models_dir, location, tag, "training")
    files = [f for f in os.listdir(mdir) if f.startswith(f"{location}.{tag}_flow_") and f.endswith(".joblib")]
    models = {}
    for f in files:
        tgt = f.split("_flow_")[-1].replace(".joblib","")
        models[tgt] = load(os.path.join(mdir, f))
    # Sanity: alle erwarteten Targets vorhanden?
    need = set(expected_targets(use_v2h))
    have = set(models.keys())
    missing = need - have
    if missing:
        raise RuntimeError(f"Fehlende Surrogate in {mdir}: {sorted(missing)}")
    return models

def surrogate_predict(models: dict, pv: float, bess: float, targets: list[str]) -> dict:
    X = np.array([[pv, bess]], dtype=float)
    out = {}
    for t in targets:
        out[t] = float(models[t].predict(X)[0])
    return out

# ---- FAST-Skalierung (1 Jahr → Lifetime) ------------------------
import numpy as np, pandas as pd
def scale_one_year_to_lifetime(sim_result: dict, lifetime: int, pv_degradation: float) -> dict:
    """Skaliert 1-Jahres-Zeitreihen auf lifetime unter Annahme linearer PV-Amplitudenänderung.
       Erwartet Keys: pv_generation, grid_import, grid_export, bess_charged, bess_discharged, optional EV/HP.
    """
    def arr1d(k):
        a = np.asarray(sim_result.get(k, np.zeros_like(sim_result["grid_import"])), dtype=float)
        return a
    pv = arr1d("pv_generation")
    gi = arr1d("grid_import")
    ge = arr1d("grid_export")
    bc = arr1d("bess_charged")
    bd = arr1d("bess_discharged")
    evc = np.asarray(sim_result.get("ev_charged", np.zeros_like(gi)), dtype=float)
    evd = np.asarray(sim_result.get("ev_discharged", np.zeros_like(gi)), dtype=float)

    pv_self = np.clip(pv - ge, 0.0, None)
    with np.errstate(divide='ignore', invalid='ignore'):
        s_self = np.where(pv > 1e-9, np.clip(pv_self / pv, 0.0, 1.0), 0.0)

    f_year = np.array([(1.0 - pv_degradation)**y for y in range(lifetime)], dtype=float)

    gi_all, ge_all, bc_all, bd_all, evc_all, evd_all, pv_all = [],[],[],[],[],[],[]
    for f in f_year:
        pv_y = pv * f
        ge_y = np.clip(ge + (pv_y - pv) * (1.0 - s_self), 0.0, None)
        pv_self_y = pv_y * s_self
        gi_y = np.clip(gi + (pv_self - pv_self_y), 0.0, None)
        gi_all.append(gi_y); ge_all.append(ge_y); pv_all.append(pv_y)
        bc_all.append(np.clip(bc * f, 0.0, None))
        bd_all.append(np.clip(bd * f, 0.0, None))
        evc_all.append(np.clip(evc * f, 0.0, None))
        evd_all.append(np.clip(evd * f, 0.0, None))

    cat = lambda lst: np.concatenate(lst) if lst else np.zeros(0)
    return dict(
        pv_generation = cat(pv_all),
        grid_import   = cat(gi_all),
        grid_export   = cat(ge_all),
        bess_charged  = cat(bc_all),
        bess_discharged = cat(bd_all),
        ev_charged    = cat(evc_all),
        ev_discharged = cat(evd_all),
    )
