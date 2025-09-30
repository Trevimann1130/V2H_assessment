# Optimization/Vectorized_model/Optimization_fast.py

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.optimize import minimize
from pymoo.termination import get_termination

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
from multiprocessing.pool import ThreadPool

from Data.data import get_parameters, load_profiles
from Optimization.Vectorized_model.precompute import prepare_profiles
from system_model_precomputed import (
    simulate_energy_system as simulate_energy_system_fast,
    simulate_energy_system_with_v2h as simulate_energy_system_with_v2h_fast,
)


from Cost_model.financial_model import calculate_npc


# ----------------- Konfiguration ----------------- #
np.random.seed(42)        # optional für reproduzierbare V2H-Stochastik
location = "Vienna"

# Basis-Parameter (WP-Parameter etc. sind fix und hängen NICHT von PV/BESS ab)
base_params = get_parameters(location)
base_params['location'] = location

# Rohprofile laden und EINMAL vorrechnen
profiles_raw = load_profiles(location)
profiles = prepare_profiles(
    base_params,
    profiles_raw,
    do_hp_electricity=True,   # WP-Strom vorab
    do_coeffs=False           # optional
)

# Sanity-Check: Precompute vorhanden?
_required = ["hp_elec_heat", "hp_elec_cool", "hotwater_HH_kWh", "pv_generation", "load", "T_outdoor"]
_missing = [k for k in _required if k not in profiles]
assert not _missing, f"Precompute missing: {_missing}"

# FAST-Flags
USE_V2H = False
USE_TQDM_FAST = False
N_GEN = 30
POP_SIZE = 20

# --- Parallelisierung: Thread-Pool Runner ---
N_THREADS = max(1, (os.cpu_count() or 4) - 1)
_pool = ThreadPool(N_THREADS)
_runner = StarmapParallelization(_pool.starmap)


# ----------------- 1-Year + Scaling (FAST) ----------------- #
def simulate_over_lifetime_fast(sim, params, profiles, pv_size):
    """
    Simuliere 1 Jahr stundenweise und skaliere dann analytisch auf 'lifetime'.
    - PV-Degradation f_y = (1 - degr)^y
    - Import/Export via stündlichem Selbstverbrauchsanteil linear nachgeführt
    - BESS-Durchsatz ~ linear mit PV-Amplitude
    - WP/thermisch: identische Jahresreihen (aus Precompute)
    """
    lifetime = int(params.get("lifetime"))
    degr = float(params['PV'].get('PVdegradation'))
    nH = len(profiles['load'])

    # 1) Basissimulation (1 Jahr)
    params['location'] = base_params['location']
    base = sim(params, profiles, pv_size)

    def arr(key, default=0.0):
        if key in base:
            return np.asarray(base[key], dtype=float)
        # standardisiere auf Nullen, falls Keys (z. B. thermal outputs) im Modell nicht befüllt werden
        return np.zeros(nH, dtype=float)

    pv_base   = arr("pv_generation")
    gi_base   = arr("grid_import")
    ge_base   = arr("grid_export")
    bc_base   = arr("bess_charged")
    bd_base   = arr("bess_discharged")
    hpH_base  = arr("heatpump_results_heating")
    hpC_base  = arr("heatpump_results_cooling")
    thH_base  = arr("thermal_output_heating")
    thC_base  = arr("thermal_output_cooling")

    # stündlicher Selbstverbrauchsanteil
    pv_self_base = pv_base - ge_base
    with np.errstate(divide='ignore', invalid='ignore'):
        s_self = np.where(pv_base > 1e-9, np.clip(pv_self_base / pv_base, 0.0, 1.0), 0.0)

    f_year = np.array([(1.0 - degr)**y for y in range(lifetime)], dtype=float)

    pv_gen_all, grids_import, grids_export = [], [], []
    bess_ch_all, bess_dis_all = [], []
    hpH_all, hpC_all, thH_all, thC_all, timestamps_all = [], [], [], [], []

    iterator = range(lifetime)
    if USE_TQDM_FAST:
        iterator = tqdm(iterator, desc="FAST scaling over years", leave=False)

    for y in iterator:
        f = f_year[y]
        pv_y = pv_base * f
        ge_y = np.clip(ge_base + (pv_y - pv_base) * (1.0 - s_self), 0.0, None)
        pv_self_y = pv_y * s_self
        gi_y = np.clip(gi_base + (pv_self_base - pv_self_y), 0.0, None)

        bc_y = np.clip(bc_base * f, 0.0, None)
        bd_y = np.clip(bd_base * f, 0.0, None)

        pv_gen_all.append(pv_y)
        grids_import.append(gi_y)
        grids_export.append(ge_y)
        bess_ch_all.append(bc_y)
        bess_dis_all.append(bd_y)
        hpH_all.append(hpH_base)
        hpC_all.append(hpC_base)
        thH_all.append(thH_base)
        thC_all.append(thC_base)

        start = pd.Timestamp("2023-01-01") + pd.DateOffset(years=y)
        timestamps_all.append(pd.date_range(start=start, periods=nH, freq="h"))

    def cat(lst): return np.concatenate(lst) if len(lst) else np.array([], dtype=float)
    results = {
        "pv_generation":             cat(pv_gen_all),
        "grid_import":               cat(grids_import),
        "grid_export":               cat(grids_export),
        "bess_charged":              cat(bess_ch_all),
        "bess_discharged":           cat(bess_dis_all),
        "heatpump_results_heating":  cat(hpH_all),
        "heatpump_results_cooling":  cat(hpC_all),
        "thermal_output_heating":    cat(thH_all),
        "thermal_output_cooling":    cat(thC_all),
        "timestamps":                np.concatenate(timestamps_all)
    }
    return results


# ----------------- Optimierungsproblem ----------------- #
class EnergyOptimizationProblem(ElementwiseProblem):
    def __init__(self, **kwargs):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_ieq_constr=0,
            xl=np.array([3, 3]),
            xu=np.array([20, 20]),
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        pv_size, battery_capacity = x
        params = get_parameters(location)
        params['location'] = base_params['location']
        params.update({'pv_size': float(pv_size), 'battery_capacity_kWh': float(battery_capacity)})

        sim = simulate_energy_system_with_v2h_fast if USE_V2H else simulate_energy_system_fast
        results = simulate_over_lifetime_fast(sim, params, profiles, pv_size)

        npc = calculate_npc(params, results)

        grid_import   = float(np.sum(results["grid_import"]))
        bess_through  = float(np.sum(results["bess_charged"]))
        hp_heating    = float(np.sum(results["heatpump_results_heating"]))
        hp_cooling    = float(np.sum(results["heatpump_results_cooling"]))

        pef_pv  = float(pv_size) * params['PV']['PEF']
        pef_bes = bess_through * params['BESS']['PEF']
        pef_grd = grid_import * params['Grid']['PEF']
        pef_hp  = (hp_heating + hp_cooling) * params['heatpump']['PEF']
        pef_ev  = (params['EV']['PEF'] if USE_V2H else 0.0)
        pef = pef_pv + pef_bes + pef_grd + pef_hp + pef_ev

        out["F"] = [npc, pef]


# ----------------- NSGA-II Lauf ----------------- #
problem = EnergyOptimizationProblem(elementwise_runner=_runner)
algorithm = NSGA2(pop_size=POP_SIZE)
termination = get_termination("n_gen", N_GEN)

gen_bar = tqdm(total=N_GEN, desc="NSGA-II generations", mininterval=0.2, leave=True)
def on_callback(alg, **kwargs):
    gen_bar.update(1)

result = minimize(problem, algorithm, termination, callback=on_callback, verbose=False)
gen_bar.close()

_pool.close()
_pool.join()

# ----------------- Pareto-Front ----------------- #
pareto = result.F
plt.figure(figsize=(8, 6))
plt.scatter(pareto[:, 0], pareto[:, 1], s=60)
plt.xlabel("Net Present Cost (NPC) [€]")
plt.ylabel("Product Environmental Footprint (PEF) [Pt]")
plt.title(f"Pareto-Front NPC vs. PEF (NSGA-II) | FAST")
plt.grid(True)
plt.show()

# ----------------- CSV & TERMINAL-PRINTS ----------------- #
overview = []
timeseries_dir = "results_optimization_fast_timeseries"
os.makedirs(timeseries_dir, exist_ok=True)

for i, ((pv, batt), (npc, pef)) in enumerate(zip(result.X, result.F), start=1):
    print(f"[FAST] PV={pv:.1f} kWp, BESS={batt:.1f} kWh -> NPC={npc:,.2f} €, PEF={pef:,.4f} Pt")

    params = get_parameters(location)
    params['location'] = base_params['location']
    params.update({'pv_size': float(pv), 'battery_capacity_kWh': float(batt)})
    sim = simulate_energy_system_with_v2h_fast if USE_V2H else simulate_energy_system_fast
    res = simulate_over_lifetime_fast(sim, params, profiles, pv)

    # --- Summenübersicht ---
    overview.append({
        "Mode": "FAST",
        "PV_kWp": float(pv),
        "BESS_kWh": float(batt),
        "NPC": float(npc),
        "PEF": float(pef),
        "Grid_Import_kWh": float(np.sum(res["grid_import"])),
        "Grid_Export_kWh": float(np.sum(res["grid_export"])),
        "HP_Heating_kWh": float(np.sum(res["heatpump_results_heating"])),
        "HP_Cooling_kWh": float(np.sum(res["heatpump_results_cooling"]))
    })

    # --- Zeitreihen-Export für 1 Jahr ---
    n = len(res["grid_import"])
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "pv_generation_kWh": res["pv_generation"],
        "grid_import_kWh": res["grid_import"],
        "grid_export_kWh": res["grid_export"],
        "bess_charged_kWh": res["bess_charged"],
        "bess_discharged_kWh": res["bess_discharged"],
        "hp_heating_elec_kWh": res["heatpump_results_heating"],
        "hp_cooling_elec_kWh": res["heatpump_results_cooling"],
        "thermal_heating_kWh": res["thermal_output_heating"],
        "thermal_cooling_kWh": res["thermal_output_cooling"],
        "total_load_kWh": res.get("total_load", np.zeros(n)),
        "hotwater_load_kWh": res.get("hotwater_load", np.zeros(n))
    })

    fname = os.path.join(timeseries_dir, f"sol_{i:03d}_PV{pv:.0f}_BESS{batt:.0f}.csv")
    df.to_csv(fname, index=False)
    print(f"   ↳ Zeitreihe exportiert: {fname}")

pd.DataFrame(overview).to_csv("pareto_overview.csv", index=False)
print("'pareto_overview.csv' exportiert.")