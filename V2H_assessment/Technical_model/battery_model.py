
def simulate_battery_flow(battery_in_request, battery_out_request, capacity_kWh, power_kW, efficiency, self_discharge, max_cycles, battery_eol_capacity, DoD):
    if capacity_kWh <= 0 or DoD <= 0:
        return {
            'battery_in_series': [0] * len(battery_in_request),
            'battery_out_series': [0] * len(battery_out_request),
            'soc_history': [0.0] * len(battery_in_request),
            'BESS_replacements': 0,
            'BESS_replacement_events': []
        }
    soc = 0.5  # Initial SoC (50 %)
    soc_history = []
    battery_in_series = []
    battery_out_series = []
    replacement_events = []

    battery_in_total = 0.0
    battery_out_total = 0.0
    replacements = 0

    for t in range(len(battery_in_request)):
        charge_request = battery_in_request[t]
        discharge_request = battery_out_request[t]

        # Selbstentladung einmal pro Tag
        if t % 24 == 0 and t > 0:
            soc *= (1 - self_discharge)

        # --- Zyklenzählung und degradierte Kapazität berechnen ---
        throughput = (battery_in_total + battery_out_total) / 2
        cycles = throughput / (DoD * capacity_kWh)
        degradation_factor = 1 - (1 - battery_eol_capacity) * min(cycles / max_cycles, 1.0)
        current_capacity = capacity_kWh * degradation_factor

        # Batterie laden (normiert, begrenzt)
        max_charge = min(charge_request, power_kW, current_capacity * (1 - soc))
        actual_charge = max(0, max_charge)
        soc += (actual_charge * efficiency) / current_capacity
        battery_in_series.append(actual_charge)
        battery_in_total += actual_charge

        # Batterie entladen (normiert, begrenzt durch DoD)
        min_soc = 1 - DoD
        max_discharge = min(discharge_request, power_kW, current_capacity * (soc - min_soc))
        actual_discharge = max(0, max_discharge)
        soc -= (actual_discharge / efficiency) / current_capacity
        battery_out_series.append(actual_discharge)
        battery_out_total += actual_discharge

        # Begrenzung des normierten SoC auf [min_soc, 1.0]
        soc = min(soc, 1.0)
        soc = max(soc, min_soc)

        # SoC Verlauf speichern
        soc_history.append(soc)

        # Batterieaustausch bei Erreichen der maximalen Zyklen
        if cycles >= (replacements + 1) * max_cycles:
            replacements += 1
            replacement_events.append(t)
            soc = 0.5 * capacity_kWh  # Reset nach Austausch
            battery_in_total = 0.0
            battery_out_total = 0.0

    return {
        'battery_in_series': battery_in_series,
        'battery_out_series': battery_out_series,
        'soc_history': soc_history,
        'BESS_replacements': replacements,
        'BESS_replacement_events': replacement_events
    }
