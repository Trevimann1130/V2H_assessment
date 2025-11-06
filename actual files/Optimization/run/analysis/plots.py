from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

def save_pareto_plot(run_dir: str, F_opt: np.ndarray, objective_names: list[str]) -> None:
    if F_opt is None or len(F_opt) == 0:
        raise ValueError("F_opt ist leer – kein Pareto-Plot möglich.")
    F = np.asarray(F_opt, float)
    if F.shape[1] != 2:
        raise ValueError("Pareto-Plot unterstützt aktuell nur genau 2 Ziele (z.B. NPC & PEF).")

    x, y = F[:, 0], F[:, 1]
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=28, alpha=0.8)
    plt.xlabel(objective_names[0])
    plt.ylabel(objective_names[1])
    plt.title("Pareto-Front")
    plt.grid(True)
    out_png = os.path.join(run_dir, "pareto.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
