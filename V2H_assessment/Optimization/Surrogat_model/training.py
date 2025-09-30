# -*- coding: utf-8 -*-
# training.py – Trainiere Surrogates (RandomForest) auf FAST-Datensatz

from __future__ import annotations
import os, argparse, pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", required=True)
    ap.add_argument("--v2h", type=str, default="false")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--cv", type=int, default=5)       # bleibt für spätere Erweiterungen
    ap.add_argument("--n-iter", type=int, default=80)  # bleibt für spätere Erweiterungen
    ap.add_argument("--n-jobs", type=int, default=-1)
    return ap.parse_args()

def select_targets(df_cols, use_v2h: bool):
    # Grundmenge – genau diese willst du trainieren
    targets = [
        "E_import_grid_kWh",  # Summe aus grid_import
        "E_import_ec_pv_kWh",  # Summe aus ec_import_from_pv
        "E_import_ec_ev_kWh",  # Summe aus ec_import_from_ev
        "E_export_grid_kWh",  # Summe aus grid_export
        "E_bess_throughput_kWh",  # BESS Throughput
        "E_ev_charged_kWh"
    ]
    if use_v2h:
        targets += ["E_ev_discharged_kWh"]

    # Nur vorhandene Spalten zurückgeben (robust ggü. Datensatzvarianten)
    return [t for t in targets if t in df_cols]

def main():
    args = parse_args()
    LOCATION = args.location
    USE_V2H  = str(args.v2h).lower() in ("1","true","yes","y")
    TAG      = "V2H" if USE_V2H else "NoV2H"

    in_dir  = os.path.join(args.data_dir, LOCATION, TAG, "dataset")
    out_dir = os.path.join(args.models_dir, LOCATION, TAG, "training")
    os.makedirs(out_dir, exist_ok=True)

    base = f"{LOCATION}.{TAG}_fast_dataset.csv"
    csv_path = os.path.join(in_dir, base)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ Dataset fehlt: {csv_path}")

    df = pd.read_csv(csv_path)

    X = df[["PV_kWp", "BESS_kWh"]].to_numpy()

    # Zielspalten explizit wählen (keine HP-Targets etc.)
    y_cols = select_targets(df.columns.tolist(), USE_V2H)
    if len(y_cols) == 0:
        raise ValueError("❌ Keine passenden Target-Spalten im Dataset gefunden.")

    trained = 0
    for target in tqdm(y_cols, desc="Training Surrogates"):
        y = df[target].to_numpy()
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=args.n_jobs
        )
        model.fit(X, y)
        out_path = os.path.join(out_dir, f"{LOCATION}.{TAG}_flow_{target}.joblib")
        dump(model, out_path)
        trained += 1

    print(f"✔ Training abgeschlossen. {trained} Modelle gespeichert in {out_dir}")

if __name__=="__main__":
    main()
