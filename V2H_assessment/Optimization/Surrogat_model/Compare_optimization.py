# -*- coding: utf-8 -*-
# compare_pareto.py – Plotte Pareto-Fronts V2H vs. NoV2H

from __future__ import annotations
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_pareto_csv(base_dir: str, location: str, tag: str) -> pd.DataFrame:
    """Lädt die Pareto-CSV aus dem Optimizer-Output."""
    tag_lower = tag.lower()
    fname = os.path.join(
        base_dir,
        tag_lower,
        location,
        "solutions", "optimization",
        f"pareto_surrogate_{location}.{tag}.csv"
    )
    if not os.path.exists(fname):
        raise FileNotFoundError(f"❌ Datei nicht gefunden: {fname}")
    return pd.read_csv(fname)

def plot_compare(base_dir: str, location: str, out_dir: str):
    """Vergleicht V2H vs. NoV2H Pareto-Fronten und speichert die Grafik."""
    df_nov2h = load_pareto_csv(base_dir, location, "NoV2H")
    df_v2h   = load_pareto_csv(base_dir, location, "V2H")

    plt.figure(figsize=(8,6))
    plt.scatter(df_nov2h["NPC"], df_nov2h["PEF"],
                label="No V2H", color="tab:blue", alpha=0.7, s=60)
    plt.scatter(df_v2h["NPC"], df_v2h["PEF"],
                label="V2H", color="tab:orange", alpha=0.7, s=60)

    plt.xlabel("Net Present Cost (NPC) [€]")
    plt.ylabel("Product Environmental Footprint (PEF) [Pt]")
    plt.title(f"Pareto-Front Vergleich – {location}")
    plt.legend()
    plt.grid(True)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"pareto_compare_{location}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✔ Vergleichsplot gespeichert: {out_path}")

if __name__ == "__main__":
    BASE_DIR = r"Optimization/Surrogat_model/results/optimization"
    OUT_DIR  = r"Optimization/Surrogat_model/results/plots"
    LOCATION = "Vienna"

    plot_compare(BASE_DIR, LOCATION, OUT_DIR)
