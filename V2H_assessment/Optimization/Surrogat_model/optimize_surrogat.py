# -*- coding: utf-8 -*-
# optimize_surrogat.py – NSGA-II Optimierung mit Surrogates
from __future__ import annotations
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import all the algorithms
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES

from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.problem import ElementwiseProblem
from joblib import load

from Cost_model.financial_model import calculate_npc_yearly
from Data.data import get_parameters, load_profiles
from Optimization.Vectorized_model.precompute import prepare_profiles


def get_algorithm(name: str, pop_size: int, ref_dirs=None):
    name = name.lower()
    if name == "nsga2":
        return NSGA2(pop_size=pop_size)
    elif name == "nsga3":
        # NSGA-III braucht Referenzrichtungen
        from pymoo.util.ref_dirs import get_reference_directions
        if ref_dirs is None:
            ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)
        return NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)
    elif name == "moead":
        from pymoo.util.ref_dirs import get_reference_directions
        if ref_dirs is None:
            ref_dirs = get_reference_directions("uniform", 2, n_partitions=12)
        return MOEAD(ref_dirs=ref_dirs, n_neighbors=15, prob_neighbor_mating=0.7)
    elif name == "agemoea":
        return AGEMOEA(pop_size=pop_size)
    elif name == "smsemoa":
        return SMSEMOA(pop_size=pop_size)
    else:
        raise ValueError(f"Unbekannter Algorithmus: {name}")


# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Surrogate-based NSGA-II Optimization")
    ap.add_argument("--location", required=True)
    ap.add_argument("--v2h", type=str, default="false")
    ap.add_argument("--pv-min", type=float, required=True)
    ap.add_argument("--pv-max", type=float, required=True)
    ap.add_argument("--bess-min", type=float, required=True)
    ap.add_argument("--bess-max", type=float, required=True)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gen", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models-dir", default="results")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--algo", type=str, default="nsga2",
                    help="Optimierungsalgorithmus: nsga2, nsga3, cmaes, moead, agemoea, smsemoa")
    ap.add_argument("--amin", type=float, default=None,
                    help="min. Lastautarkie (0..1). None = kein Constraint")
    return ap.parse_args()


# -------------------- Main --------------------
def main():
    args = parse_args()

    LOCATION = args.location
    USE_V2H  = str(args.v2h).lower() in ("1","true","yes","y")
    TAG      = "V2H" if USE_V2H else "NoV2H"

    # Pfade
    MODELS_DIR = os.path.join(args.models_dir, LOCATION, TAG, "training")
    OUT_DIR    = os.path.join(args.out_dir, LOCATION, TAG, "optimization")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Modelle laden
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
    models = {}
    for mf in model_files:
        # erwartet: <Location>.<TAG>_flow_<target>.joblib
        target = mf.split("_flow_")[-1].replace(".joblib", "")
        models[target] = load(os.path.join(MODELS_DIR, mf))
    print(f"✔ {len(models)} Surrogates geladen aus {MODELS_DIR}")

    # Parameter & Profile (Profile nur, falls später benötigt)
    base_params = get_parameters(LOCATION)
    base_params["location"] = LOCATION
    _profiles_raw = load_profiles(LOCATION)
    _profiles = prepare_profiles(base_params, _profiles_raw, True, False)


    # Optimierungsproblem
    class SurrogateProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(n_var=2, n_obj=2,
                             xl=np.array([args.pv_min, args.bess_min]),
                             xu=np.array([args.pv_max, args.bess_max]))

        def _evaluate(self, x, out, *_, **__):
            X_feat = np.array(x, dtype=float).reshape(1, -1)

            # Vorhersagen (Surrogates liefern Lebensdauer-Summen!)
            def pred(name, default=0.0):
                return float(models[name].predict(X_feat)[0]) if name in models else default

            e_import_grid_L = pred("E_import_grid_kWh", 0.0)
            e_import_ec_pv_L = pred("E_import_ec_pv_kWh", 0.0)
            e_import_ec_ev_L = pred("E_import_ec_ev_kWh", 0.0)
            e_export_grid_L = pred("E_export_grid_kWh", 0.0)
            e_export_pv_ec_L = pred("E_export_pv_ec_kWh", 0.0)
            e_export_ev_ec_L = pred("E_export_ev_ec_kWh", 0.0)

            # Szenarioparameter
            params = dict(base_params)
            params["pv_size"] = float(x[0])
            params["battery_capacity_kWh"] = float(x[1])

            # Jahreswerte aus Lebensdauer-Summen ableiten
            # Jahreswerte aus Lebensdauer-Summen ableiten
            lifetime = int(params.get("lifetime"))
            e_import_grid_Y = e_import_grid_L / lifetime
            e_import_ec_pv_Y = e_import_ec_pv_L / lifetime
            e_import_ec_ev_Y = e_import_ec_ev_L / lifetime

            e_export_grid_Y = e_export_grid_L / lifetime
            e_export_pv_ec_Y = e_export_pv_ec_L / lifetime
            e_export_ev_ec_Y = e_export_ev_ec_L / lifetime

            # NPC (neues Financial Model mit 5 Flüssen)
            npc = calculate_npc_yearly(
                params,
                e_import_grid_year=e_import_grid_Y,
                e_import_ec_pv_year=e_import_ec_pv_Y,
                e_import_ec_ev_year=e_import_ec_ev_Y,
                e_export_grid_year=e_export_grid_Y,
                e_export_pv_ec_year=e_export_pv_ec_Y,
                e_export_ev_ec_year=e_export_ev_ec_Y
            )

            # PEF: PV (kWp), BESS (Throughput lebensdauer), Grid (Import lebensdauer)
            pef = (
                    params["PV"]["PEF"] * params["pv_size"] +
                    params["BESS"]["PEF"] * params.get("battery_capacity_kWh") +
                    params["Grid"]["PEF"] * e_import_grid_L +
                    params["PV"]["PEF"] * e_import_ec_pv_L +
                    params["EV"]["PEF"] * e_import_ec_ev_L
            )

            # EV-PEF optional (derzeit 0 in Daten), falls gewünscht:
            # pef += params["EV"]["PEF"] * max(0.0, e_ev_ch_L)

            out["F"] = [npc, pef]

    problem = SurrogateProblem()
    algorithm = get_algorithm(args.algo, args.pop)
    termination = get_termination("n_gen", args.gen)

    res = minimize(problem, algorithm, termination, seed=args.seed, verbose=True)

    # Ergebnisse speichern
    pareto = pd.DataFrame(res.F, columns=["NPC", "PEF"])
    pareto["PV_kWp"]   = res.X[:, 0]
    pareto["BESS_kWh"] = res.X[:, 1]

    algo_tag = args.algo.lower()

    out_csv = os.path.join(OUT_DIR, f"pareto_{algo_tag}_{LOCATION}.{TAG}.csv")
    pareto.to_csv(out_csv, index=False)
    print(f"✔ Pareto-CSV: {out_csv}")

    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(pareto["NPC"], pareto["PEF"], s=40, alpha=0.7)
    plt.xlabel("Net Present Cost (NPC) [€]")
    plt.ylabel("Product Environmental Footprint (PEF) [Pt]")
    plt.title(f"Pareto-Front Surrogate | {LOCATION} {TAG} ({algo_tag.upper()})")
    plt.grid(True)
    out_png = os.path.join(OUT_DIR, f"pareto_{algo_tag}_{LOCATION}.{TAG}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✔ Pareto-Plot: {out_png}")


# === Debug/Test-Run für feste Parameter ===
if __name__ == "__main__":

    # Fixe Testparameter
    TEST_PV   = 1   # kWp
    TEST_BESS = 1    # kWh
    USE_V2H   = False
    LOCATION  = "Vienna"

    print("\n=== Test-Run für feste Parameter ===")
    base_params = get_parameters(LOCATION)
    base_params["location"] = LOCATION
    profiles_raw = load_profiles(LOCATION)
    profiles = prepare_profiles(base_params, profiles_raw, do_hp_electricity=True)

    params = dict(base_params)
    params["pv_size"] = TEST_PV
    params["battery_capacity_kWh"] = TEST_BESS

    # Vorhersagen mit Surrogates laden
    TAG = "V2H" if USE_V2H else "NoV2H"
    MODELS_DIR = os.path.join("results", LOCATION, TAG, "training")
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
    models = {}
    for mf in model_files:
        target = mf.split("_flow_")[-1].replace(".joblib", "")
        models[target] = load(os.path.join(MODELS_DIR, mf))

    def pred(name, default=0.0):
        X_feat = np.array([[TEST_PV, TEST_BESS]], dtype=float)
        return float(models[name].predict(X_feat)[0]) if name in models else default

    # Lebensdauer-Summen
    e_import_grid_L = pred("E_import_grid_kWh")
    e_import_ec_pv_L = pred("E_import_ec_pv_kWh")
    e_import_ec_ev_L = pred("E_import_ec_ev_kWh")
    e_export_grid_L = pred("E_export_grid_kWh")
    e_export_pv_ec_L = pred("E_export_pv_ec_kWh")
    e_export_ev_ec_L = pred("E_export_ev_ec_kWh")
    e_bess_L = pred("E_bess_throughput_kWh")
    e_ev_charged_L = pred("E_ev_charged_kWh")
    e_ev_discharged_L = pred("E_ev_discharged_kWh")

    lifetime = int(params.get("lifetime", 25))

    # Pro Jahr
    e_import_grid_Y = e_import_grid_L / lifetime
    e_import_ec_pv_Y = e_import_ec_pv_L / lifetime
    e_import_ec_ev_Y = e_import_ec_ev_L / lifetime
    e_export_grid_Y = e_export_grid_L / lifetime
    e_export_pv_ec_Y = e_export_pv_ec_L / lifetime
    e_export_ev_ec_Y = e_export_ev_ec_L / lifetime

    # NPC mit Financial Model
    npc = calculate_npc_yearly(
        params,
        e_import_grid_year=e_import_grid_Y,
        e_import_ec_pv_year=e_import_ec_pv_Y,
        e_import_ec_ev_year=e_import_ec_ev_Y,
        e_export_grid_year=e_export_grid_Y,
        e_export_pv_ec_year=e_export_pv_ec_Y,
        e_export_ev_ec_year=e_export_ev_ec_Y
    )

    # PEF (aktuelle einfache Variante)
    pef = (
        params["PV"]["PEF"] * params["pv_size"] +
        params["BESS"]["PEF"] * params.get("battery_capacity_kWh") +
        params["Grid"]["PEF"] * e_import_grid_L +
        params["PV"]["PEF"] * e_import_ec_pv_L +
        params["EV"]["PEF"] * e_import_ec_ev_L
    )

    print(f"PV: {TEST_PV} kWp | BESS: {TEST_BESS} kWh | Mode: {TAG}")
    print(f"Grid Import:     {e_import_grid_Y:,.0f} kWh/a")
    print(f"Grid Export:     {e_export_grid_Y:,.0f} kWh/a")
    print(f"EC Import (PV):  {e_import_ec_pv_Y:,.0f} kWh/a")
    print(f"EC Import (EV):  {e_import_ec_ev_Y:,.0f} kWh/a")
    print(f"EC Export (PV):  {e_export_pv_ec_Y:,.0f} kWh/a")
    print(f"EC Export (EV):  {e_export_ev_ec_Y:,.0f} kWh/a")
    print(f"EV Charged:      {e_ev_charged_L/lifetime:,.0f} kWh/a")
    print(f"EV Discharged:   {e_ev_discharged_L/lifetime:,.0f} kWh/a")
    print(f"BESS Throughput: {e_bess_L/lifetime:,.0f} kWh/a")
    print("\n--- KPIs ---")
    print(f"NPC: {npc:,.0f} €")
    print(f"PEF: {pef:,.0f} Pt")
