import numpy as np

def get_c_rate(temp, c_rate_table):

    temp = float(temp)
    for entry in c_rate_table:
        if entry['min_temp'] <= temp < entry['max_temp']:
            return entry['c_rate']
    return 0.1  # Rückgabewert, wenn die Temperatur außerhalb der angegebenen Bereiche liegt


def get_availability_for_hour(hour, availability_profile):

    availability_value = availability_profile[hour]  # Aus dem Availability Profile

    # Generiere eine Zufallszahl zwischen 0 und 1
    random_number = np.random.random()

    # Fahrzeug ist verfügbar, wenn die Zufallszahl kleiner als der Verfügbarkeitswert im Profile ist
    return random_number <= availability_value  # True, wenn verfügbar


def simulate_v2h_battery(temperature, params):
    capacity_kWh = params['capacity_kWh']
    charge_c_rate_table = params['charge_c_rate_table']
    discharge_c_rate_table = params['discharge_c_rate_table']
    charge_efficiency = params['charging_efficiency']
    discharge_efficiency = params['discharging_efficiency']

    n_steps = len(temperature)

    charge_power_limit = []
    discharge_power_limit = []
    temperature_series = []
    c_rate_charge_series = []
    charge_power_limit_series = []

    for t in range(n_steps):
        T = temperature[t]
        c_rate_charge = get_c_rate(T, charge_c_rate_table)
        c_rate_discharge = get_c_rate(T, discharge_c_rate_table)

        max_charge_power = c_rate_charge * capacity_kWh * charge_efficiency
        max_discharge_power = c_rate_discharge * capacity_kWh * discharge_efficiency

        temperature_series.append(T)
        c_rate_charge_series.append(c_rate_charge)
        charge_power_limit_series.append(min(max_charge_power, params['max_charge_power']))

        charge_power_limit.append(max_charge_power)
        discharge_power_limit.append(max_discharge_power)

    return {
        'charge_power_limit': np.array(charge_power_limit),
        'discharge_power_limit': np.array(discharge_power_limit),
        'temperature': np.array(temperature),
        'temperature_series': np.array(temperature_series),
        'c_rate_charge': np.array(c_rate_charge_series),
        'charge_power_limit_capped': np.array(charge_power_limit_series)
    }
