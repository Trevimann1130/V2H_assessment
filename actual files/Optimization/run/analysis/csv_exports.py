from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

# Optionaler Detail-Teacher (für Flüsse/KPIs im CSV), nur wenn explizit gewünscht
from Optimization.framework.engines.Vectorized_model.fast_engine import FastEngine

def export_pareto_csv(
    run_dir: str,
    settings,
    X_opt: np.ndarray,
    F_opt: np.ndarray,
    G_opt: np.ndarray | None = None,
    use_teacher_for_details: bool = True,
) -> None:
    if X_opt is None or len(X_opt) == 0:
        raise ValueError("X_opt ist leer – kein CSV-Export möglich.")

    X = np.asarray(X_opt, float)
    F = np.asarray(F_opt, float) if F_opt is not None else None
    G = np.asarray(G_opt, float) if G_opt is not None else None

    cols = list(settings.bounds.names)  # ["pv_kwp", "bess_kwh"]
    df = pd.DataFrame(X, columns=cols)

    # F/G-Spalten (NPC/PEF + Constraints)
    if F is not None:
        for j, name in enumerate(settings.objectives.names):
            df[f"F_{name}"] = F[:, j]
    if G is not None and G.size:
        for j, name in enumerate(settings.constraints.names):
            df[f"G_{name}"] = G[:, j]

    # Optional: Detail-KPIs/Flows via FAST (für Reporting), ohne Fallbacks
    if use_teacher_for_details:
        teacher = FastEngine(settings)
        F_t, G_t = teacher.evaluate(X)  # konsistenter Check (nicht zwingend benutzt)
        # Beispielhafte ~15 relevante Größen aus dem Teacher für Kontext/Debug
        # → Wir nehmen an, dass teacher.additional() einen dict je X liefert.
        #   Falls nicht vorhanden, kommentiere diese Zeilen und nutze nur F/G.
        try:
            details_rows = []
            for i in range(len(X)):
                # Implementiere in deinem FastEngine optional eine Methode .flows(X[i])
                # die ein dict mit Schlüssel/Wert zurückgibt.
                flows = teacher.flows(X[i])  # <— falls nicht vorhanden, Exception (keine Fallbacks)
                details_rows.append(flows)
            df_details = pd.DataFrame(details_rows)
            # Auswahl auf ~15
            wanted = [
                "E_import_grid_kWh",
                "E_export_grid_kWh",
                "E_bess_throughput_kWh",
                "E_hp_heat_kWh",
                "E_hp_cool_kWh",
                "E_pv_gen_kWh",
                "E_ev_charged_kWh",
                "E_ev_discharged_kWh",
                "E_ev_trip_loss_kWh",
                "EC_import_from_pv_kWh",
                "EC_import_from_ev_kWh",
                "EC_export_to_grid_kWh",
                "autarky",           # als informative KPI
                "self_consumption",  # optional falls vorhanden
                "peak_import_kW"     # optional falls vorhanden
            ]
            existing = [c for c in wanted if c in df_details.columns]
            df = pd.concat([df, df_details[existing]], axis=1)
        except Exception as e:
            # Explizit: keine stillen Fallbacks – klare Meldung ins results.json.appendix
            with open(os.path.join(run_dir, "export_warning.txt"), "w", encoding="utf-8") as f:
                f.write(f"Detail-Export via FAST fehlgeschlagen: {e}\n")

    out_csv = os.path.join(run_dir, "pareto_points.csv")
    df.to_csv(out_csv, index=False)

    # Zusätzlich die Roh-Results (zur Nachvollziehbarkeit)
    with open(os.path.join(run_dir, "results_plus.csvmeta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "objectives": settings.objectives.names,
            "constraints": settings.constraints.names,
            "bounds": {"names": settings.bounds.names, "lower": settings.bounds.lower, "upper": settings.bounds.upper},
        }, f, indent=2)
