import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data/data.py als Paket importieren
from Data import data as data

def calculate_dynamic_heating_cooling(T_outdoor, solar_gains, building_params, usage_df):
    # --- Gebäudeeigenschaften ---
    cp_air = building_params["cp_air"]          # [Wh/m³K]
    room_height = building_params["room_height"]
    A_floor = building_params["A_floor"]
    heat_capacity = building_params["heat_capacity"]  # [Wh/K]
    T_min = building_params["T_min"]           # Komfort-Heiztemperatur [K]
    T_max = building_params["T_max"]           # Komfort-Kühltemperatur [K]

    # --- Nutzungsprofile ---
    QI_winter = usage_df["Qi Winter W/m2"].to_numpy()
    QI_summer = usage_df["Qi Sommer W/m2"].to_numpy()
    ACH_V = usage_df["Luftwechsel_Anlage_1_h"].to_numpy()
    ACH_I = usage_df["Luftwechsel_Infiltration_1_h"].to_numpy()

    # --- U-Werte & Flächen ---
    U_wall = building_params["U_wall"]
    U_window = building_params["U_window"]
    U_roof = building_params["U_roof"]
    U_floor = building_params["U_floor"]
    A_wall = building_params["A_wall"]
    A_window = sum(building_params["A_window"].values())
    A_roof = building_params["A_roof"]
    A_floor_building = building_params["A_floor"]

    # Wärmeverlustkoeffizient (L_T)
    L_B = (U_wall*A_wall) + (U_window*A_window) + (U_roof*A_roof) + (U_floor*A_floor_building)
    A_B = A_wall + A_window + A_roof + A_floor_building
    L_PX = max(0, 0.2 * (0.75 - (L_B/A_B)) * L_B)
    L_T = L_B + L_PX  # [W/K]

    # --- Initialisierung ---
    n_steps = len(T_outdoor)
    TI = np.zeros(n_steps)
    QT = np.zeros(n_steps)
    QV = np.zeros(n_steps)
    QI = np.zeros(n_steps)
    QS = np.zeros(n_steps)
    heating_load = np.zeros(n_steps)
    cooling_load = np.zeros(n_steps)
    TI[0] = T_min  # Startwert: Komforttemperatur

    # Heiz-/Kühlflags bestimmen
    daily_avg_temp, heating_days, cooling_days, heating_flags, cooling_flags = analyze_heating_cooling_days(T_outdoor)

    # --- Simulation ---
    for t in range(1, n_steps):
        # Verluste und Gewinne berechnen
        dT = TI[t-1] - T_outdoor[t]
        QT[t] = (L_T * dT) / A_floor
        QV[t] = ((ACH_I[t] + ACH_V[t]) * room_height * cp_air * dT)
        QI[t] = QI_winter[t] if heating_flags[t] else QI_summer[t]
        QS[t] = solar_gains[t]

        # Temperatur nach Verlusten/Gewinnen
        Q_net = -QT[t] - QV[t] + QI[t] + QS[t]
        TI[t] = TI[t-1] + Q_net / heat_capacity

        # --- Komfortregelung wie im EnergyModel ---
        if TI[t] < T_min:
            # Heizen immer, wenn TI zu kalt ist (EnergyModel-Logik)
            QH = (T_min - TI[t]) * heat_capacity
            heating_load[t] = QH / 1000
            TI[t] = T_min
        elif TI[t] > T_max:
            # Kühlen immer, wenn TI zu warm ist (EnergyModel-Logik)
            QC = (T_max - TI[t]) * heat_capacity
            cooling_load[t] = abs(QC) / 1000
            TI[t] = T_max

    # DataFrame erstellen (alle relevanten Größen wie im EnergyModel)
    df_hourly = pd.DataFrame({
        "T_innen [°C]": TI - 273.15,
        "T_außen [°C]": T_outdoor - 273.15,
        "QT [W/m²]": QT,
        "QV [W/m²]": QV,
        "QI [W/m²]": QI,
        "QS [W/m²]": QS,
        "Heizlast [kWh/m²]": heating_load,
        "Kühllast [kWh/m²]": cooling_load
    })

    return df_hourly



def analyze_heating_cooling_days(T_outdoor, heating_threshold=285.15, cooling_threshold=291.45):
    daily_avg_temp = T_outdoor.reshape(-1, 24).mean(axis=1)
    heating_days = daily_avg_temp < heating_threshold
    cooling_days = daily_avg_temp > cooling_threshold
    heating_flags = np.repeat(heating_days, 24)
    cooling_flags = np.repeat(cooling_days, 24)
    return daily_avg_temp, heating_days, cooling_days, heating_flags, cooling_flags


# --- Kompatibilitätsfunktionen für system_model.py und heatpump_model.py ---

def calculate_heating_load(T_outdoor, irradiance, building_params, usage_df):
    df = calculate_dynamic_heating_cooling(T_outdoor, irradiance, building_params, usage_df)
    return df["Heizlast [kWh/m²]"].to_numpy() * building_params["A_floor"]

def calculate_cooling_load(T_outdoor, irradiance, building_params, usage_df):
    df = calculate_dynamic_heating_cooling(T_outdoor, irradiance, building_params, usage_df)
    return df["Kühllast [kWh/m²]"].to_numpy() * building_params["A_floor"]

def get_heating_day_flags(T_outdoor):
    _, _, _, heating_flags, _ = analyze_heating_cooling_days(T_outdoor)
    return heating_flags.astype(int)

def get_cooling_day_flags(T_outdoor):
    _, _, _, _, cooling_flags = analyze_heating_cooling_days(T_outdoor)
    return cooling_flags.astype(int)

def get_heating_load_on_days(location):
    profiles = data.load_profiles(location)
    building_params = data.technologies_global['building']
    df = calculate_dynamic_heating_cooling(profiles['T_outdoor'], profiles['solargains'], building_params, profiles['usage_profile'])
    return df["Heizlast [kWh/m²]"].to_numpy() * building_params["A_floor"]

def get_cooling_load_on_days(location):
    profiles = data.load_profiles(location)
    building_params = data.technologies_global['building']
    df = calculate_dynamic_heating_cooling(profiles['T_outdoor'], profiles['solargains'], building_params, profiles['usage_profile'])
    return df["Kühllast [kWh/m²]"].to_numpy() * building_params["A_floor"]


def plot_results(df_hourly, building_params, daily_avg_temp, heating_days, cooling_days, heating_threshold, cooling_threshold):
    # --- Jahresbilanz ---
    print("\n--- Jahres-Energiebilanz ---")
    print(f"QT (Transmission): {df_hourly['QT [W/m²]'].sum() / 1000:.2f} kWh/m²/a")
    print(f"QV (Lüftung):      {df_hourly['QV [W/m²]'].sum() / 1000:.2f} kWh/m²/a")
    print(f"QI (Intern):       {df_hourly['QI [W/m²]'].sum() / 1000:.2f} kWh/m²/a")
    print(f"QS (Solar):        {df_hourly['QS [W/m²]'].sum() / 1000:.2f} kWh/m²/a")
    print(f"Heizlast:          {df_hourly['Heizlast [kWh/m²]'].sum():.2f} kWh/m²/a")
    print(f"Kühllast:          {df_hourly['Kühllast [kWh/m²]'].sum():.2f} kWh/m²/a")

    # --- Plot 1: Temperatur ---
    plt.figure(figsize=(12, 5))
    plt.plot(df_hourly["T_innen [°C]"], label="T_innen", color='black')
    plt.plot(df_hourly["T_außen [°C]"], label="T_außen", color='gray', linestyle=':')
    plt.axhline(y=building_params["T_min"] - 273.15, color='blue', linestyle='--', label="T_min")
    plt.axhline(y=building_params["T_max"] - 273.15, color='red', linestyle='--', label="T_max")
    plt.legend()
    plt.title("Innen- und Außentemperatur über das Jahr")
    plt.grid()
    plt.show()

    # --- Plot 2: Jahres-Energiebilanz ---
    plt.figure(figsize=(8, 5))
    components = ["QT", "QV", "QI", "QS", "Heizlast", "Kühllast"]
    values = [
        df_hourly['QT [W/m²]'].sum() / 1000,
        df_hourly['QV [W/m²]'].sum() / 1000,
        df_hourly['QI [W/m²]'].sum() / 1000,
        df_hourly['QS [W/m²]'].sum() / 1000,
        df_hourly['Heizlast [kWh/m²]'].sum(),
        df_hourly['Kühllast [kWh/m²]'].sum()
    ]
    plt.barh(components, values, color=['orange', 'lightblue', 'green', 'yellow', 'red', 'purple'])
    plt.xlabel("Energie [kWh/m²/a]")
    plt.title("Jahres-Energiebilanz")
    plt.grid(axis='x')
    plt.show()

    # --- Plot 3: Heiz- und Kühltage ---
    plt.figure(figsize=(12, 4))
    plt.plot(daily_avg_temp - 273.15, label="Tagesmittel Außen", color='black')
    plt.axhline(y=heating_threshold - 273.15, color='blue', linestyle='--', label="Heizschwelle")
    plt.axhline(y=cooling_threshold - 273.15, color='red', linestyle='--', label="Kühlschwelle")
    plt.fill_between(range(len(daily_avg_temp)), -20, 50, where=heating_days, color='blue', alpha=0.2, label="Heiztage")
    plt.fill_between(range(len(daily_avg_temp)), -20, 50, where=cooling_days, color='red', alpha=0.2, label="Kühltage")
    plt.legend()
    plt.title("Heiz- und Kühltage basierend auf Tagesmitteltemperaturen")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    location = "Vienna"
    profiles = data.load_profiles(location)
    T_outdoor = profiles['T_outdoor']
    solar_gains = profiles['solargains']
    building_params = data.technologies_global['building']
    usage_df = profiles['usage_profile']

    df_hourly = calculate_dynamic_heating_cooling(T_outdoor, solar_gains, building_params, usage_df)

    heating_threshold = 285.15  # 12°C
    cooling_threshold = 291.45  # 18°C
    daily_avg_temp, heating_days, cooling_days, heating_flags, cooling_flags = analyze_heating_cooling_days(
        T_outdoor, heating_threshold, cooling_threshold
    )

    plot_results(df_hourly, building_params, daily_avg_temp, heating_days, cooling_days, heating_threshold, cooling_threshold)

    # ➡ Gesamtenergiemengen berechnen
    total_heating_energy = df_hourly["Heizlast [kWh/m²]"].sum() * building_params["A_floor"]
    total_cooling_energy = df_hourly["Kühllast [kWh/m²]"].sum() * building_params["A_floor"]

    print("\n--- Gesamte Energiemengen ---")
    print(f"Gesamte Heizenergie: {total_heating_energy:.2f} kWh")
    print(f"Gesamte Kühlenergie: {total_cooling_energy:.2f} kWh")
