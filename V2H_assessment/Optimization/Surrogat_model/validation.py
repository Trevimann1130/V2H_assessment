# -*- coding: utf-8 -*-
# validation.py – Validierung der Surrogates gegen FAST

from __future__ import annotations
import os, argparse
import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm

from Optimization.Surrogat_model.LHS import make_lhs_matrix, snap_to_steps, evaluate_batch
from Optimization.Vectorized_model.precompute import prepare_profiles
from Data.data import get_parameters, load_profiles


def parse_args():
    ap = argparse.ArgumentParser(description="Validate surrogate models against FAST.")
    ap.add_argument("--location", required=True)
    ap.add_argument("--v2h", type=str, default="false")
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--pv-min", type=float, required=True)
    ap.add_argument("--pv-max", type=float, required=True)
    ap.add_argument("--pv-step", type=float, default=1.0)

    ap.add_argument("--bess-min", type=float, required=True)
    ap.add_argument("--bess-max", type=float, required=True)
    ap.add_argument("--bess-step", type=float, default=1.0)

    ap.add_argument("--models-dir", default="results")
    ap.add_argument("--out-dir", default="results")
    return ap.parse_args()


def get_targets(use_v2h: bool, df_cols):
    base = [
        "E_import_grid_kWh",
        "E_import_ec_pv_kWh",
        "E_import_ec_ev_kWh",
        "E_export_grid_kWh",
        "E_bess_throughput_kWh",
        "E_ev_charged_kWh",
    ]
    if use_v2h and "E_ev_discharged_kWh" in df_cols:
        base += ["E_ev_discharged_kWh"]
    return [t for t in base if t in df_cols]   # nur vorhandene nehmen



def main():
    args = parse_args()

    LOCATION = args.location
    USE_V2H  = str(args.v2h).lower() in ("1","true","yes","y")
    TAG      = "V2H" if USE_V2H else "NoV2H"

    MODELS_DIR = os.path.join(args.models_dir, LOCATION, TAG, "training")
    OUT_DIR    = os.path.join(args.out_dir, LOCATION, TAG, "validation")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Parameter & Profile laden
    base_params = get_parameters(LOCATION)
    base_params["location"] = LOCATION
    profiles_raw = load_profiles(LOCATION)
    profiles = prepare_profiles(base_params, profiles_raw, do_hp_electricity=True, do_coeffs=False)

    # LHS-Samples im gewählten Designraum erzeugen
    bounds = [(args.pv_min, args.pv_max), (args.bess_min, args.bess_max)]
    X0 = make_lhs_matrix(args.samples, bounds, args.seed)
    X  = snap_to_steps(X0, [args.pv_step, args.bess_step])
    X  = np.unique(X, axis=0)

    # FAST-Simulation (liefert Lebensdauer-Summen für die Energiegrößen)
    df_fast = evaluate_batch(
        X,
        base_params=base_params,
        profiles=profiles,
        use_v2h=USE_V2H,
        backend="threads",
        n_workers=max(1, (os.cpu_count() or 4) - 1),
        show_progress=True,
    )

    # Targets filtern (keine HP-Targets trainieren/validieren)
    targets = get_targets(USE_V2H, df_fast.columns)

    # Surrogates laden
    models = {}
    for t in targets:
        path = os.path.join(MODELS_DIR, f"{LOCATION}.{TAG}_flow_{t}.joblib")
        if os.path.exists(path):
            models[t] = load(path)
        else:
            print(f"⚠ Surrogat fehlt: {path}")

    # Validation-Metriken
    metrics_rows = []
    X_feat = df_fast[["PV_kWp", "BESS_kWh"]].to_numpy()

    for t in tqdm(targets, desc="Validation Targets"):
        if t not in models:
            continue
        y_true = df_fast[t].to_numpy()
        y_pred = models[t].predict(X_feat)

        r2 = float(np.corrcoef(y_true, y_pred)[0, 1] ** 2) if len(y_true) > 1 else np.nan
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mean_true = np.mean(np.abs(y_true)) if np.mean(np.abs(y_true)) > 1e-9 else np.nan
        rel_mae = (mae / mean_true) * 100 if mean_true else np.nan

        metrics_rows.append({
            "location": LOCATION,
            "tag": TAG,
            "target": t,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "rel_mae_percent": rel_mae,
        })

    out_csv = os.path.join(OUT_DIR, f"{LOCATION}.{TAG}_validation_metrics.csv")
    pd.DataFrame(metrics_rows).to_csv(out_csv, index=False)
    print(f"✔ Validation-Metriken gespeichert: {out_csv}")


if __name__ == "__main__":
    main()
