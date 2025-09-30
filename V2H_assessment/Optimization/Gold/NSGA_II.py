from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm  # <<< NEU
from Data.data import get_parameters, load_profiles
from Technical_model.system_model import simulate_energy_system_with_v2h, simulate_energy_system
from Cost_model.financial_model import calculate_npc


# Hilfsfunktion für Mehrjahressimulation
def simulate_over_lifetime(sim, params, profiles, pv_size):
    lifetime = params.get("lifetime")
    combined = {
        "grid_import": [], "grid_export": [], "bess_charged": [], "bess_discharged": [],
        "bess_soc": [], "ev_charged": [], "ev_discharged": [], "ev_soc": [],
        "trip_loss": [], "driving_energy": [], "availability_profile": [], "ev_active": [],
        "pv_generation": [], "BESS_replacement_events": [], "BESS_degradation": [], "timestamps": [],
        "EV_temperature": [], "EV_c_rate_charge": [], "EV_charge_power_limit": [], "EV_min_SOC": [],
        "total_load": [], "heatpump_results_heating": [], "heatpump_results_cooling": [],
        "thermal_output_heating": [], "thermal_output_cooling": []
    }

    for year in range(lifetime):
        params['location'] = location  # Location
        results = sim(params, profiles, pv_size)
        for key in combined:
            if key == "timestamps":
                start = pd.Timestamp("2023-01-01") + pd.DateOffset(years=year)
                combined[key].extend(pd.date_range(start=start, periods=len(profiles['load']), freq="h"))
            else:
                combined[key].extend(results.get(key, [np.nan] * len(profiles['load'])))
    return combined


# Parameter
location = "Vienna"
profiles = load_profiles(location)
USE_V2H = True


# Problemdefinition
class EnergyOptimizationProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_ieq_constr=0,
            xl=np.array([10, 10]),   # Untergrenzen für PV und BESS
            xu=np.array([100, 100])  # Obergrenzen für PV und BESS
        )

    def _evaluate(self, x, out, *args, **kwargs):
        pv_size, battery_capacity = x
        params = get_parameters(location)
        params['location'] = location
        params.update({'pv_size': pv_size, 'battery_capacity_kWh': battery_capacity})

        sim = simulate_energy_system_with_v2h if USE_V2H else simulate_energy_system
        results = simulate_over_lifetime(sim, params, profiles, pv_size)

        # NPC
        npc = calculate_npc(params, results)

        # Kennzahlen
        grid_import = np.sum(results["grid_import"])
        bess_throughput = np.sum(results["bess_charged"])
        hp_heating = np.sum(results["heatpump_results_heating"])
        hp_cooling = np.sum(results["heatpump_results_cooling"])

        # PEF
        pef_pv = pv_size * params['PV']['PEF']
        pef_bess = bess_throughput * params['BESS']['PEF']
        pef_grid = grid_import * params['Grid']['PEF']
        pef_hp = (hp_heating + hp_cooling) * params['heatpump']['PEF']
        pef_ev = (params['EV']['PEF'] if USE_V2H else 0)

        pef = pef_pv + pef_bess + pef_grid + pef_hp + pef_ev

        out["F"] = [npc, pef]


# ----------------- Algorithmus (NSGA-II) + tqdm ----------------- #
problem = EnergyOptimizationProblem()
algorithm = NSGA2(pop_size=5)

# Anzahl Generationen
n_gens = 5
termination = get_termination("n_gen", n_gens)

# Fortschrittsbalken für Generationen (tqdm)
pbar = tqdm(total=n_gens, desc="NSGA-II optimization", mininterval=0.5)

def on_callback(algorithm, **kwargs):
    # Wird am Ende jeder Generation aufgerufen → +1
    pbar.update(1)

result = minimize(problem,
                  algorithm,
                  termination,
                  callback=on_callback,
                  verbose=False)  # wichtig, damit tqdm „sauber“ bleibt

pbar.close()

# ----------------- Pareto-Front plotten -----------------
pareto_objectives = result.F
plt.figure(figsize=(8, 6))
plt.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], c='red', s=60)
plt.xlabel("Net Present Cost (NPC) [€]")
plt.ylabel("Product Environmental Footprint (PEF) [Pt]")
plt.title("Pareto-Front NPC vs. PEF (NSGA-II)")
plt.grid(True)
plt.show()

# ----------------- CSV Export mit allen Kennzahlen -----------------
overview_data = []
for (pv, batt), (npc, pef) in zip(result.X, result.F):
    params = get_parameters(location)
    params['location'] = location
    params.update({'pv_size': pv, 'battery_capacity_kWh': batt})
    sim = simulate_energy_system_with_v2h if USE_V2H else simulate_energy_system
    results = simulate_over_lifetime(sim, params, profiles, pv)

    grid_import = np.sum(results["grid_import"])
    grid_export = np.sum(results["grid_export"])
    hp_heating = np.sum(results["heatpump_results_heating"])
    hp_cooling = np.sum(results["heatpump_results_cooling"])
    thermal_heating = np.sum(results.get("thermal_output_heating", [0]))
    thermal_cooling = np.sum(results.get("thermal_output_cooling", [0]))

    overview_data.append({
        "PV_kWp": pv,
        "BESS_kWh": batt,
        "NPC": npc,
        "PEF": pef,
        "Grid_Import_kWh": grid_import,
        "Grid_Export_kWh": grid_export,
        "HP_Heating_kWh": hp_heating,
        "HP_Cooling_kWh": hp_cooling,
        "Thermal_Heating_kWh": thermal_heating,
        "Thermal_Cooling_kWh": thermal_cooling
    })

pd.DataFrame(overview_data).to_csv("pareto_overview.csv", index=False)
print("'pareto_overview.csv' exportiert.")
