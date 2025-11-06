# -*- coding: utf-8 -*-
from __future__ import annotations
import os, argparse, numpy as np, pandas as pd
from tqdm import tqdm

from Data.data import get_parameters, load_profiles
from Technical_model.energy_system.precompute.precompute import prepare_profiles
# GOLD (High-Fidelity):
from Technical_model.energy_system.systems.PV_BESS_HP_EV import (
    simulate_energy_system as simulate_energy_system_gold)
from Technical_model.energy_system.systems.PV_BESS_HP_V2H import (
    simulate_energy_system_with_v2h as simulate_energy_system_with_v2h_gold,
)
# FAST (Low-Fidelity):
from Technical_model.energy_system.systems.PV_BESS_HP_EV import (
    simulate_energy_system as simulate_energy_system_fast)
from Technical_model.energy_system.systems.PV_BESS_HP_V2H import (
    simulate_energy_system_with_v2h as simulate_energy_system_with_v2h_fast,
)

from Optimization.framework.validation.utils_metrics import compute_metrics, lhs, snap_to_steps, expected_targets, scale_one_year_to_lifetime, save_metrics_table

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", required=True)
    ap.add_argument("--v2h", type=str, default="false")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--pv-min", type=float, required=True)
    ap.add_argument("--pv-max", type=float, required=True)
    ap.add_argument("--bess-min", type=float, required=True)
    ap.add_argument("--bess-max", type=float, required=True)
    ap.add_argument("--pv-step", type=float, default=1.0)
    ap.add_argument("--bess-step", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-dir", default="Optimization/Surrogat_model/results")
    return ap.parse_args()

def main():
    a = parse_args()
    use_v2h = str(a.v2h).lower() in ("1","true","yes","y")
    TAG = "V2H" if use_v2h else "NoV2H"
    targets = expected_targets(use_v2h)

    # Params & Profiles (einmalig vorberechnen)
    params = get_parameters(a.location); params["location"]=a.location
    prof_raw = load_profiles(a.location)
    profiles = prepare_profiles(params, prof_raw, do_hp_electricity=True, do_coeffs=False)

    # Stichprobe
    X = lhs(a.samples, [(a.pv_min,a.pv_max),(a.bess_min,a.bess_max)], a.seed)
    X = snap_to_steps(X, [a.pv_step, a.bess_step])

    rec = []
    for pv, bess in tqdm(X, desc="FAST↔GOLD validation"):
        p = dict(params); p["pv_size"]=float(pv); p["battery_capacity_kWh"]=float(bess)

        # --- GOLD: 1 Jahr simulieren, dann auf lifetime skalieren
        simG = simulate_energy_system_with_v2h_gold if use_v2h else simulate_energy_system_gold
        resG_1y = simG(p, profiles, pv)
        resG_L  = scale_one_year_to_lifetime(resG_1y, int(p["lifetime"]), float(p["PV"]["PVdegradation"]))

        # --- FAST: 1 Jahr (precomputed Pfad), dann skalieren
        simF = simulate_energy_system_with_v2h_fast if use_v2h else simulate_energy_system_fast
        resF_1y = simF(p, profiles, pv)
        resF_L  = scale_one_year_to_lifetime(resF_1y, int(p["lifetime"]), float(p["PV"]["PVdegradation"]))

        row = dict(location=a.location, tag=TAG, PV_kWp=pv, BESS_kWh=bess)
        # Summen über Lifetime
        row.update({
            "G_import": np.sum(resG_L["grid_import"]),
            "F_import": np.sum(resF_L["grid_import"]),
            "G_export": np.sum(resG_L["grid_export"]),
            "F_export": np.sum(resF_L["grid_export"]),
            "G_bess":   np.sum(resG_L["bess_charged"]),
            "F_bess":   np.sum(resF_L["bess_charged"]),
        })
        if use_v2h:
            row["G_ev_ch"] = np.sum(resG_L["ev_charged"])
            row["F_ev_ch"] = np.sum(resF_L["ev_charged"])
            row["G_ev_ds"] = np.sum(resG_L["ev_discharged"])
            row["F_ev_ds"] = np.sum(resF_L["ev_discharged"])
        rec.append(row)

    df = pd.DataFrame(rec)

    # Metriken: F vs G
    metrics = []
    metrics.append({"location":a.location,"tag":TAG,"target":"E_import_kWh",**compute_metrics(df["G_import"], df["F_import"])})
    metrics.append({"location":a.location,"tag":TAG,"target":"E_export_kWh",**compute_metrics(df["G_export"], df["F_export"])})
    metrics.append({"location":a.location,"tag":TAG,"target":"E_bess_throughput_kWh",**compute_metrics(df["G_bess"],  df["F_bess"])})
    if use_v2h:
        metrics.append({"location":a.location,"tag":TAG,"target":"E_ev_charged_kWh",**compute_metrics(df["G_ev_ch"], df["F_ev_ch"])})
        metrics.append({"location":a.location,"tag":TAG,"target":"E_ev_discharged_kWh",**compute_metrics(df["G_ev_ds"], df["F_ev_ds"])})

    out_base = os.path.join(a.out_dir, a.location, TAG, "validation")
    os.makedirs(out_base, exist_ok=True)
    df.to_csv(os.path.join(out_base, f"{a.location}.{TAG}_fast_vs_gold_samples.csv"), index=False)
    save_metrics_table(metrics, os.path.join(out_base, f"{a.location}.{TAG}_fast_vs_gold_metrics.csv"))
    print("✔ FAST↔GOLD validation done.")

if __name__=="__main__":
    main()
