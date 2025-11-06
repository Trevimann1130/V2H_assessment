from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from ..utils import dbg, XSchema

def _lhs(n_samples: int, n_dim: int, rng: np.random.Generator) -> np.ndarray:
    cut = np.linspace(0, 1, n_samples + 1)
    pts = rng.random((n_samples, n_dim)) * (cut[1:] - cut[:-1])[:, None] + cut[:-1, None]
    for j in range(n_dim): rng.shuffle(pts[:, j])
    return pts

def _sobol(n_samples: int, n_dim: int, seed: int) -> np.ndarray:
    try:
        from scipy.stats.qmc import Sobol
    except Exception as e:
        raise RuntimeError("Sampler 'sobol' requires scipy (scipy.stats.qmc.Sobol).") from e
    eng = Sobol(d=n_dim, scramble=True, seed=seed)
    return eng.random(n_samples)

def _halton(n_samples: int, n_dim: int, seed: int) -> np.ndarray:
    try:
        from scipy.stats.qmc import Halton
    except Exception as e:
        raise RuntimeError("Sampler 'halton' requires scipy (scipy.stats.qmc.Halton).") from e
    eng = Halton(d=n_dim, scramble=True, seed=seed)
    return eng.random(n_samples)

def _scale(u: np.ndarray, bounds: Tuple[Tuple[float, float], ...]) -> np.ndarray:
    out = np.empty_like(u, dtype=float)
    for j,(lo,hi) in enumerate(bounds): out[:,j] = lo + (hi-lo)*u[:,j]
    return out

def sample_probes(
    strategy: str, n_total: int, xschema: XSchema, rng_sampler: np.random.Generator,
    include_pareto_from_run: bool, run_root: Path, override_name: str | None = None, override_seed: int | None = None
) -> pd.DataFrame:
    names, d = list(xschema.names), len(xschema.names)
    base = None

    # include Pareto (optional)
    pareto = None
    if include_pareto_from_run:
        p = run_root / "X_pareto.csv"
        if p.exists():
            pareto = pd.read_csv(p)[names]
            dbg(f"Including Pareto points: {len(pareto)}")
        else:
            dbg("WARNING: X_pareto.csv not found (include_pareto_from_run=True).")

    # resolve effective sampler
    eff = strategy
    if strategy == "match_run":
        if not override_name:
            raise ValueError("probes.strategy='match_run' requires sampler_name from settings (override) or run_meta mapping.")
        eff = override_name

    need = n_total - (0 if pareto is None else len(pareto))
    need = max(0, need)

    if eff == "lhs":
        u = _lhs(need, d, rng_sampler)
    elif eff == "random":
        u = rng_sampler.random((need, d))
    elif eff == "sobol":
        if override_seed is None:
            raise ValueError("sobol requires sampler.seed in settings.validation.sampler")
        u = _sobol(need, d, override_seed)
    elif eff == "halton":
        if override_seed is None:
            raise ValueError("halton requires sampler.seed in settings.validation.sampler")
        u = _halton(need, d, override_seed)
    elif eff == "corners":
        grid = np.array(np.meshgrid(*[[0.0,1.0]]*d)).T.reshape(-1,d)
        base = _scale(grid, xschema.bounds)
        if base.shape[0] < n_total:
            reps = int(np.ceil(n_total/base.shape[0]))
            base = np.vstack([base for _ in range(reps)])[:n_total]
        df = pd.DataFrame(base, columns=names)
        return df if pareto is None else pd.concat([pareto, df]).drop_duplicates().reset_index(drop=True)
    else:
        raise ValueError(f"Unknown probes.strategy/eff='{strategy}' resolved to '{eff}'")

    base = _scale(u, xschema.bounds) if need>0 else np.empty((0,d))
    df = pd.DataFrame(base, columns=names)
    if pareto is not None:
        df = pd.concat([pareto, df], axis=0, ignore_index=True).drop_duplicates().reset_index(drop=True)
    if len(df) < 1:
        raise ValueError("No probes selected.")
    return df
