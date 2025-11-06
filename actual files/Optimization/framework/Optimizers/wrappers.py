# Optimization/framework/Optimizers/wrappers.py
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np

from pymoo.optimize import minimize
from pymoo.termination import get_termination

# MOO-Algorithmen
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.age import AGEMOEA
from pymoo.algorithms.moo.sms import SMSEMOA

# SOO (nur 1 Ziel)
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES

from pymoo.util.ref_dirs import get_reference_directions


def _build_ref_dirs(algo_name: str, n_obj: int, ref_dirs_param: Optional[Any]) -> np.ndarray:
    """
    Liefert ein Array an Referenzrichtungen.
    - Wenn ref_dirs_param ein dict ist: {"method": "...", "n_partitions": int} -> get_reference_directions(...)
    - Wenn ref_dirs_param bereits ein Array ist: zurückgeben
    - Sonst: sinnvolle Defaults pro Algo
    """
    if ref_dirs_param is None:
        method = "das-dennis" if algo_name == "nsga3" else "uniform"
        n_partitions = 12
        return get_reference_directions(method, n_obj, n_partitions=n_partitions)

    if isinstance(ref_dirs_param, dict):
        method = ref_dirs_param.get("method", "das-dennis" if algo_name == "nsga3" else "uniform")
        n_partitions = int(ref_dirs_param.get("n_partitions", 12))
        return get_reference_directions(method, n_obj, n_partitions=n_partitions)

    # bereits konkrete ref_dirs (numpy)
    return ref_dirs_param


def get_algorithm(name: str, n_obj: int, kwargs: Dict[str, Any]) -> Any:
    """
    Fabrik für Pymoo-Algorithmen.
    - Setzt für NSGA-III/MOEA/D ref_dirs und pop_size robust.
    - Lässt sonst pop_size aus kwargs direkt durch.
    """
    name = str(name).lower().strip()
    kw = dict(kwargs or {})

    if name == "nsga2":
        pop_size = int(kw.get("pop_size", 100))
        return NSGA2(pop_size=pop_size)

    elif name == "nsga3":
        ref_dirs = _build_ref_dirs("nsga3", n_obj, kw.get("ref_dirs"))
        pop_size = int(kw.get("pop_size", len(ref_dirs)))
        return NSGA3(pop_size=pop_size, ref_dirs=ref_dirs)

    elif name == "moead":
        ref_dirs = _build_ref_dirs("moead", n_obj, kw.get("ref_dirs"))
        pop_size = int(kw.get("pop_size", len(ref_dirs)))
        n_neighbors = int(kw.get("n_neighbors", min(15, len(ref_dirs))))
        prob_neighbor_mating = float(kw.get("prob_neighbor_mating", 0.7))
        return MOEAD(ref_dirs=ref_dirs,
                     n_neighbors=n_neighbors,
                     prob_neighbor_mating=prob_neighbor_mating)

    elif name == "agemoea":
        pop_size = int(kw.get("pop_size", 100))
        return AGEMOEA(pop_size=pop_size)

    elif name == "smsemoa":
        pop_size = int(kw.get("pop_size", 100))
        return SMSEMOA(pop_size=pop_size)

    elif name == "cmaes":
        # CMA-ES nur für ein Ziel sinnvoll
        if n_obj != 1:
            raise ValueError("CMA-ES ist ein Single-Objective-Algorithmus (n_obj muss 1 sein).")
        sigma = float(kw.get("sigma", 1.0))
        pop_size = kw.get("pop_size", None)
        return CMAES(sigma=sigma, popsize=pop_size)

    else:
        raise ValueError(f"Unbekannter Optimizer: {name}")


def run_pymoo(problem, optimizer_cfg, seed: int | None = None):
    """
    Baut den gewählten Algorithmus und führt minimize() aus.
    Erwartet:
      - optimizer_cfg.name (str)
      - optimizer_cfg.kwargs (dict) z.B. {"pop_size":..., "n_gen":..., "ref_dirs": {...}}
      - seed (int|None)
    """
    n_obj = int(getattr(problem, "n_obj", 2))

    algo = get_algorithm(optimizer_cfg.name, n_obj, optimizer_cfg.kwargs)
    n_gen = int(optimizer_cfg.kwargs.get("n_gen", 150))

    termination = get_termination("n_gen", n_gen)
    res = minimize(problem, algo, termination, seed=seed, verbose=True)
    return res
