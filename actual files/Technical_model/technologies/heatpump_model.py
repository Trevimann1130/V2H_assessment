import numpy as np
import matplotlib.pyplot as plt
from Data import data as data
from Technical_model.consumption.heating_anc_cooling_consumption.heating_and_cooling import (
    get_heating_load_on_days,
    get_cooling_load_on_days,
)

def calculate_cop_heating(T_outdoor, T_flow, eta_cop=0.4, cop_max=6):
    if T_flow <= T_outdoor:
        return 1.0
    cop_carnot = T_flow / (T_flow - T_outdoor)
    cop_real = eta_cop * cop_carnot
    return min(max(cop_real, 1.0), cop_max)

def calculate_cop_cooling(T_outdoor, T_flow_cool, eta_cop, eer_max):
    if T_outdoor <= T_flow_cool:
        return 1.0
    cop_carnot = T_flow_cool / (T_outdoor - T_flow_cool)
    cop_real = eta_cop * cop_carnot
    return min(max(cop_real, 1.0), eer_max)

def simulate_heatpump_heating_system(params, profiles):

    heating_load_on_days = get_heating_load_on_days(location=params['location'])
    T_outdoor_array = profiles['T_outdoor']

    hp = params['heatpump']
    hours = len(heating_load_on_days)
    elec = np.zeros(hours)
    thermal = np.zeros(hours)
    cop_series = np.zeros(hours)

    for t in range(hours):
        Q_demand = heating_load_on_days[t]
        T_outdoor = T_outdoor_array[t]

        if Q_demand <= 0:
            continue

        COP = calculate_cop_heating(
            T_outdoor=T_outdoor,
            T_flow=hp['T_flow'],
            eta_cop=hp.get('eta_cop', 0.4),
            cop_max=hp.get('cop_max', 6)
        )

        elec[t] = Q_demand / COP if COP > 0 else 0.0
        thermal[t] = Q_demand
        cop_series[t] = COP

    return {
        'electric_consumption_series': elec,
        'thermal_output_series': thermal,
        'cop_series': cop_series,
    }

def simulate_heatpump_cooling_system(params, profiles):
    cooling_load_on_days = get_cooling_load_on_days(location=params['location'])
    T_outdoor_array = profiles['T_outdoor']

    hp = params['heatpump']
    building = params.get('building', {})
    T_in_summer = building.get('T_max')

    hours = len(cooling_load_on_days)
    elec = np.zeros(hours)
    cooling = np.zeros(hours)
    cop_series = np.zeros(hours)

    for t in range(hours):
        Q_cool_demand = cooling_load_on_days[t]
        T_outdoor = T_outdoor_array [t]
        T_flow_cool = hp.get('T_flow_cool', T_in_summer)

        if Q_cool_demand <= 0:
            continue

        COP_cool = calculate_cop_cooling(
            T_outdoor,
            T_flow_cool,
            eta_cop=hp.get('eta_cop', 0.4),
            eer_max=hp.get('eer_max', 5)
        )

        elec[t] = Q_cool_demand / COP_cool if COP_cool > 0 else 0.0
        cooling[t] = Q_cool_demand
        cop_series[t] = COP_cool

    return {
        'electric_consumption_series': elec,
        'cooling_output_series': cooling,
        'cop_series': cop_series
    }

if __name__ == "__main__":
    location = "Vienna"
    profiles = data.load_profiles(location)
    params = data.get_parameters(location)
    params['location'] = location

    # Heizbetrieb
    results_heating = simulate_heatpump_heating_system(params, profiles)
    print("Heizbetrieb (Winter):")
    print(f"Gesamter Stromverbrauch (Heizen): {np.sum(results_heating['electric_consumption_series']):.2f} kWh")

    # Kühlbetrieb
    results_cooling = simulate_heatpump_cooling_system(params, profiles)
    print("Kühlbetrieb (Sommer):")
    print(f"Gesamter Stromverbrauch (Kühlen): {np.sum(results_cooling['electric_consumption_series']):.2f} kWh")

    # Wöchentliche Summen
    hours_per_week = 24 * 7
    num_weeks = len(results_heating['electric_consumption_series']) // hours_per_week

    weekly_heat = results_heating['electric_consumption_series'][:num_weeks * hours_per_week].reshape((num_weeks, hours_per_week)).sum(axis=1)
    weekly_cool = results_cooling['electric_consumption_series'][:num_weeks * hours_per_week].reshape((num_weeks, hours_per_week)).sum(axis=1)

    # Plot
    plt.figure(figsize=(15, 5))
    plt.bar(np.arange(num_weeks) - 0.2, weekly_heat, width=0.4, color='tab:red', label='WP-Heizen')
    plt.bar(np.arange(num_weeks) + 0.2, weekly_cool, width=0.4, color='tab:blue', label='WP-Kühlen')
    plt.xlabel('Woche im Jahr')
    plt.ylabel('Wöchentlicher Stromverbrauch WP [kWh]')
    plt.title('Wöchentlicher Stromverbrauch Wärmepumpe (Heizen & Kühlen)')
    plt.legend()
    plt.tight_layout()
    plt.show()