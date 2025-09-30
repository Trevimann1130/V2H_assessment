# -*- coding: utf-8 -*-
from __future__ import annotations
import os, argparse, numpy as np, pandas as pd
from tqdm import tqdm

from Data.data import get_parameters, load_profiles
from Optimization.Vectorized_model.precompute import prepare_profiles
from Optimization.Vectorized_model.system_model_precomputed import (
    simulate_energy_system as simulate_energy_system_fast,
    simulate_energy_system_with_v2h as simulate_energy_system_with_v2h_fast,
)

from .utils_metrics import compute_metrics, lhs, snap_to_steps, expected_targets, load_surrogates, surrogate_predict, scale_one_year_to_lifetime, save_metrics_table

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", required=True)
    ap.add_argument("--v2h", type=str, default="false")
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--pv-min", type=float, required=True)
    ap.add_argument("--pv-max", type=float, required=True)
    ap.add_argument("--bess-min", type=float, required=True)
    ap.add_argument("--bess-max", type=float, required=True)
    ap.add_argument("--pv-step", type=float, default=1.0)
    ap.add_argument("--bess-step", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=321)
    ap.add_argument("--models-dir", default="Optimization/Surrogat_model/results")
    ap.add_argument("--out-dir",     default="Optimization/Surrogat_model/results")
    return ap.parse_args()

def main():
    a = parse_args()
    use_v2h = str(a.v2h).lower() in ("1","true","yes","y")
    TAG = "V2H" if use_v2h else "NoV2H"
    targets = expected_targets(use_v2h)

    models = load_surrogates(a.models_dir, a.location, TAG, use_v2h)

    params = get_parameters(a.location); params["location"]=a.location
    prof_raw = load_profiles(a.location)
    profiles = prepare_profiles(params, prof_raw, do_hp_electricity=True, do_coeffs=False)

    X = lhs(a.samples, [(a.pv_min,a.pv_max),(a.bess_min,a.bess_max)], a.seed)
    X = snap_to_steps(X, [a.pv_step, a.bess_step])

    rec = []
    for pv, bess in tqdm(X, desc="SM↔FAST validation"):
        p = dict(params); p["pv_size"]=float(pv); p["battery_capacity_kWh"]=float(bess)
        simF = simulate_energy_system_with_v2h_fast if use_v2h else simulate_energy_system_fast
        resF_1y = simF(p, profiles, pv)
        resF_L  = scale_one_year_to_lifetime(resF_1y, int(p["lifetime"]), float(p["PV"]["PVdegradation"]))

        # Summen über Lifetime (FAST)
        f_import = float(np.sum(resF_L["grid_import"]))
        f_export = float(np.sum(resF_L["grid_export"]))
        f_bess   = float(np.sum(resF_L["bess_charged"]))
        f_ev_ch  = float(np.sum(resF_L.get("ev_charged", np.zeros(1))))
        f_ev_ds  = float(np.sum(resF_L.get("ev_discharged", np.zeros(1))))

        # Surrogat (liefert Lebensdauer-Summen, da auf FAST-Summen trainiert)
        sp = surrogate_predict(models, pv, bess, targets)

        row = dict(location=a.location, tag=TAG, PV_kWp=pv, BESS_kWh=bess,
                   F_import=f_import, S_import=sp.get("E_import_kWh",0.0),
                   F_export=f_export, S_export=sp.get("E_export_kWh",0.0),
                   F_bess=f_bess,   S_bess=sp.get("E_bess_throughput_kWh",0.0))
        if use_v2h:
            row["F_ev_ch"]=f_ev_ch; row["S_ev_ch"]=sp.get("E_ev_charged_kWh",0.0)
            row["F_ev_ds"]=f_ev_ds; row["S_ev_ds"]=sp.get("E_ev_discharged_kWh",0.0)
        rec.append(row)

    df = pd.DataFrame(rec)
    metrics = []
    metrics.append({"location":a.location,"tag":TAG,"target":"E_import_kWh",**compute_metrics(df["F_import"], df["S_import"])})
    metrics.append({"location":a.location,"tag":TAG,"target":"E_export_kWh",**compute_metrics(df["F_export"], df["S_export"])})
    metrics.append({"location":a.location,"tag":TAG,"target":"E_bess_throughput_kWh",**compute_metrics(df["F_bess"],  df["S_bess"])})
    if use_v2h:
        metrics.append({"location":a.location,"tag":TAG,"target":"E_ev_charged_kWh",**compute_metrics(df["F_ev_ch"], df["S_ev_ch"])})
        metrics.append({"location":a.location,"tag":TAG,"target":"E_ev_discharged_kWh",**compute_metrics(df["F_ev_ds"], df["S_ev_ds"])})

    out_base = os.path.join(a.out_dir, a.location, TAG, "validation")
    os.makedirs(out_base, exist_ok=True)
    df.to_csv(os.path.join(out_base, f"{a.location}.{TAG}_surrogate_vs_fast_samples.csv"), index=False)
    save_metrics_table(metrics, os.path.join(out_base, f"{a.location}.{TAG}_surrogate_vs_fast_metrics.csv"))
    print("✔ Surrogate↔FAST validation done.")

if __name__=="__main__":
    main()
