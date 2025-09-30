# evaluator.py – deterministische Auswertung von NPC & PEF
#                aus (Größen x) + (Flüsse aus Surrogat) + (aktuellen Parametern)

from __future__ import annotations            # moderne Typannotationen
from typing import Dict, Any, Tuple           # Typen für Klarheit
import numpy as np                            # Numerik


# ---------- kleine Hilfen ----------

def present_worth_sum(wacc: float, growth: float, years: int) -> float:
    """
    Barwert-Summe S = sum_{y=0..L-1} ((1+growth)^y / (1+wacc)^y)
    -> skaliert *konstante* Jahresmengen/Einnahmen mit Preiswachstum & Diskontierung.
    """
    r = 1.0 + float(wacc)                     # Diskontfaktor pro Jahr
    g = 1.0 + float(growth)                   # Preiswachstum pro Jahr
    # Geometrische Reihe: sum (g/r)^y  = (1 - (g/r)^L) / (1 - g/r), robust codiert:
    q = g / r                                 # Verhältnis g/r
    if abs(q - 1.0) < 1e-12:                  # Spezialfall g ≈ r: Summe ≈ L
        return float(years)
    return float((1.0 - q**years) / (1.0 - q))

def clamp_nonneg(x: float) -> float:
    """Sicherheitsklemme: negative Rundungsartefakte verhindern."""
    return float(max(0.0, x))

# ---------- Kern: NPC & PEF Berechnung ----------

def evaluate_npc_and_pef(params: Dict[str, Any],
                         x: Dict[str, float],
                         flows: Dict[str, float]) -> Tuple[float, float, Dict[str, float]]:
    """
    Berechnet NPC & PEF aus:
      - params: Preis-/WACC-/PEF-/CAPEX-Infos (dein get_parameters-Output)
      - x:      Größen {"PV_kWp": .., "BESS_kWh": ..}
      - flows:  Lebensdauersummen {"E_import_kWh", "E_export_kWh", "E_bess_throughput_kWh", ...}
    Rückgabe:
      (npc, pef, breakdown_dict) – breakdown für Reporting/Debug.
    """

    # ---- Lebensdauer & Makro-Parameter (mit robusten Fallbacks) ----
    L = int(params.get("lifetime", 25))                    # Lebensdauer (Jahre)
    WACC = float(params.get("WACC", 0.08))                 # Diskontsatz
    Cbuy0 = float(params.get("Cbuy", 0.30))                # Start-Strompreis Bezug [€/kWh]
    Csell0 = float(params.get("Csell", 0.08))              # Start-Einspeisetarif [€/kWh]
    g_buy = float(params.get("electricity_price_growth", 0.0))    # jährl. Bezugspreis-Wachstum
    g_sell = float(params.get("feedin_growth_rate", 0.0))         # jährl. Einspeisepreis-Wachstum

    # ---- Größen (Designvariablen) ----
    PV = float(x.get("PV_kWp", 0.0))                       # PV kWp
    BESS = float(x.get("BESS_kWh", 0.0))                   # BESS kWh

    # ---- Flüsse (Lebensdauer-Summen); falls fehlen -> 0 ----
    E_imp = clamp_nonneg(float(flows.get("E_import_kWh", 0.0)))           # ∑ Import [kWh]
    E_exp = clamp_nonneg(float(flows.get("E_export_kWh", 0.0)))           # ∑ Export [kWh]
    E_bes = clamp_nonneg(float(flows.get("E_bess_throughput_kWh", 0.0)))  # ∑ BESS Durchsatz (geladen) [kWh]
    E_hpH = clamp_nonneg(float(flows.get("E_hp_heat_kWh", 0.0)))          # optional: ∑ HP Heizen [kWh]
    E_hpC = clamp_nonneg(float(flows.get("E_hp_cool_kWh", 0.0)))          # optional: ∑ HP Kühlen [kWh]

    # ---- CAPEX
    PVp = params.get("PV", {})
    BSp = params.get("BESS", {})

    capex_pv_unit = float(PVp.get("capex_per_kWp",
                                  PVp.get("CAPEX_per_kWp",
                                          params.get("CPV"))))

    capex_b_unit = float(BSp.get("capex_per_kWh",
                                 BSp.get("CAPEX_per_kWh",
                                         params.get("CBESS"))))

    capex_pv = capex_pv_unit * float(x["PV_kWp"])
    capex_b = capex_b_unit * float(x["BESS_kWh"])

    # Optionale Ersatzkosten (BESS kürzer als Systemlebensdauer)
    bess_life = int(BSp.get("lifetime_years", L))              # BESS-Lebensdauer (J), Standard = L
    repl_frac = float(BSp.get("replacement_cost_fraction", 1.0))  # Ersatzanteil (0..1) vom initialen CAPEX
    n_repl = 0 if bess_life <= 0 else max(0, (L - 1) // bess_life)  # Anzahl ganzer Ersatzzeitpunkte

    # Barwert der Ersatzkosten: Summe CAPEX_repl / (1+WACC)^t (ohne Preiswachstum, CAPEX oft fix in realen €)
    capex_b_repl_pv = 0.0                                      # init
    for k in range(1, n_repl + 1):                              # für jeden Ersatz
        t = k * bess_life                                      # Ersatzzeitpunkt in Jahren
        capex_b_repl_pv += (repl_frac * capex_b) / ((1.0 + WACC) ** t)

    # ---- OPEX-ähnliche Kosten/Erträge aus Flüssen (Post-Processing) ----
    # Idee: wir nehmen ∑E über die Lebensdauer und verteilen *gleichmäßig* auf Jahre.
    # Dann ist NPC_OPEX = (E/L) * C0 * present_worth_sum(WACC, growth, L).
    # -> Sehr schnell & konsistent mit deinem FAST-Ansatz.
    pw_buy  = present_worth_sum(WACC, g_buy,  L)               # Barwert-Faktor für Bezug
    pw_sell = present_worth_sum(WACC, g_sell, L)               # Barwert-Faktor für Einspeisung

    # Jahresmittel der Energien (kWh/a), konservativ gleichmäßig verteilt
    A_imp = E_imp / L
    A_exp = E_exp / L

    # Barwert der Strombezugskosten (positiv = Kosten)
    npv_buy  = A_imp * Cbuy0  * pw_buy
    # Barwert der Einspeiseerträge (negativ in NPC-Summen – wir ziehen ab)
    npv_sell = A_exp * Csell0 * pw_sell

    # Optionale O&M (fix pro kWp/kWh und/oder variable Anteile) – Fallback = 0
    opex_pv_per_kwp_a   = float(PVp.get("opex_per_kWp_per_year", 0.0))  # €/kWp/a
    opex_bess_per_kwh_a = float(BSp.get("opex_per_kWh_per_year", 0.0))  # €/kWh/a
    # O&M NPV (fixe jährliche Zahlungen -> Wachstumsrate 0)
    pw_fix = present_worth_sum(WACC, 0.0, L)
    npv_om = PV * opex_pv_per_kwp_a * pw_fix + BESS * opex_bess_per_kwh_a * pw_fix

    # Gesamt-NPC: CAPEX initial + PV der Ersatzkosten + PV(Importkosten) - PV(Einspeiseerträge) + O&M
    npc = capex_pv + capex_b + capex_b_repl_pv + npv_buy - npv_sell + npv_om

    # ---- PEF (Größen- + Flusskanal) ----
    # Embodied / verkörperter Fußabdruck (pro kWp/kWh) – robuste Fallbacks:
    pef_pv_emb_unit   = float(PVp.get("PEF_embodied_per_kWp", PVp.get("PEF", 0.0)))    # Pt/kWp
    pef_bess_emb_unit = float(BSp.get("PEF_embodied_per_kWh", BSp.get("PEF", 0.0)))    # Pt/kWh
    pef_embodied = PV * pef_pv_emb_unit + BESS * pef_bess_emb_unit                     # verkörpert

    # Operativer Fußabdruck (pro kWh):
    Grid = params.get("Grid", {})
    HP   = params.get("heatpump", {})
    pef_grid   = float(Grid.get("PEF", 0.0))         # Pt/kWh Import
    pef_export = float(Grid.get("PEF_export", 0.0))  # Pt/kWh Export (oft 0 oder Gutschrift)
    pef_bess   = float(BSp.get("PEF_operational_per_kWh", BSp.get("PEF", 0.0)))  # Pt/kWh BESS-Durchsatz
    pef_hp     = float(HP.get("PEF", 0.0))           # Pt/kWh WP-Strom (falls separat bewertet)

    # Operative PEF-Anteile (Lebensdauer-Summen)
    pef_oper = (
        pef_grid * E_imp
        - pef_export * E_exp
        + pef_bess * E_bes
        + pef_hp * (E_hpH + E_hpC)
    )

    pef_total = pef_embodied + pef_oper              # Gesamt-PEF

    # ---- Breakdown für Reporting ----
    breakdown = {
        "CAPEX_PV": capex_pv,
        "CAPEX_BESS_initial": capex_b,
        "CAPEX_BESS_replacements_PV": capex_b_repl_pv,
        "NPV_buy": npv_buy,
        "NPV_sell": npv_sell,
        "NPV_OM": npv_om,
        "PEF_embodied": pef_embodied,
        "PEF_operational": pef_oper,
        "PW_factor_buy": pw_buy,
        "PW_factor_sell": pw_sell,
        "A_imp_kWh_per_a": A_imp,
        "A_exp_kWh_per_a": A_exp,
    }

    return float(npc), float(pef_total), breakdown
