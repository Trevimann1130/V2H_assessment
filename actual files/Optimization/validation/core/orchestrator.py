# Optimization/validation/core/orchestrator.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any

from ..utils import dbg, now_tag
from ..build.settings_bridge import build_context_from_run_id
from ..sampling.strategies import sample_probes
from ..sampling.subsets import pick_fast, pick_gold
from ..engines.teachers import run_fast, run_gold
from ..engines.predictors import run_surrogate
from ..metrics.core import compute
from ..io import reports as R


def _build_params(meta: Dict[str, Any]) -> Dict[str, Any]:
    objectives = meta.get("objectives")
    if not objectives:
        objectives = (meta.get("settings_digest") or {}).get("objectives") or {}
        if not objectives:
            raise ValueError("run_meta.json missing 'objectives' and no fallback in settings_digest.objectives")

    engines = meta.get("engines") or {}

    profiles = meta.get("profiles") or {}
    if not profiles:
        eng = (meta.get("settings_digest") or {}).get("engine", {})
        profiles = {
            "location":           meta.get("location") or eng.get("location"),
            "system_id":          meta.get("scenario_name") or eng.get("system_id"),
            "use_v2h":            bool(eng.get("use_v2h", False)),
            "N_HH":               int(eng.get("N_HH", 1)),
            "N_EV_total":         int(eng.get("N_EV_total", 0)),
            "N_EV_bidirectional": int(eng.get("N_EV_bidirectional", 0)),
            "ec_share_import":    float(eng.get("ec_share_import", 1.0)),
            "ec_share_export":    float(eng.get("ec_share_export", 1.0)),
        }

    surrogate_targets = meta.get("surrogate_targets")

    ec_share = meta.get("ec_share")
    v2h_flag = meta.get("v2h_flag")
    n_households = meta.get("n_households")
    n_evs = meta.get("n_evs")

    scenario_name = meta.get("scenario_name") or (meta.get("settings_digest") or {}).get("engine", {}).get("system_id")
    location = meta.get("location") or (meta.get("settings_digest") or {}).get("engine", {}).get("location")
    if not scenario_name or not location:
        raise ValueError("run_meta.json missing 'scenario_name'/'location' and no fallback available")

    return {
        "scenario":           scenario_name,
        "location":           location,
        "profiles":           profiles,
        "engines":            engines,
        "objectives":         objectives,
        "surrogate_targets":  surrogate_targets,
        "ec_share":           ec_share,
        "v2h_flag":           v2h_flag,
        "n_households":       n_households,
        "n_evs":              n_evs,
    }


def validate() -> Path:
    # 1) Settings/Run-Kontext laden
    vcfg, ctx = build_context_from_run_id()

    # 2) Ausgabepfad für Validation
    project_root = Path(__file__).resolve().parents[3]
    out = (project_root / "Optimization" / "validation" / "results" / ctx.run_meta["run_id"] / now_tag()).resolve()
    (out / "plots").mkdir(parents=True, exist_ok=True)

    # 3) Parameterblock
    params = _build_params(ctx.run_meta)

    # 4) Audit
    audit = {
        "run_id": ctx.run_meta["run_id"],
        "teacher_mode": vcfg.teacher_mode,
        "params_hash": ctx.params_hash,
        "sampler": {
            "mode": vcfg.sampler_mode,
            "name": vcfg.sampler_name,
            "seed": vcfg.sampler_seed,
            "params": vcfg.sampler_params,
        },
        "sampler_from_run": ctx.sampler_from_run,
        "seeds": {"validation": ctx.seeds_validation, "run": ctx.seeds_run},
        "versions": ctx.run_meta.get("engines", {}),
        "xschema": {
            "names": ctx.xschema.names,
            "lower": ctx.xschema.lower,
            "upper": ctx.xschema.upper,
        },
    }
    R.save_audit(out, audit)

    rng_sampler = np.random.default_rng(ctx.seeds_validation["sampler"])
    rng_subset = np.random.default_rng(ctx.seeds_validation["subset_selection"])

    # 5) Effektiver Sampler bei 'match_run'
    eff_name = vcfg.sampler_name
    eff_seed = vcfg.sampler_seed
    if vcfg.probes_strategy == "match_run":
        eff_name = ctx.sampler_from_run.get("name")
        eff_seed = ctx.sampler_from_run.get("seed")
        if not eff_name:
            raise ValueError("probes.strategy='match_run' but run_meta.sampler_signature.name is missing.")

    # 6) Probes ziehen
    X_all = sample_probes(
        strategy=vcfg.probes_strategy,
        n_total=vcfg.probes_n_total,
        xschema=ctx.xschema,
        rng_sampler=rng_sampler,
        include_pareto_from_run=vcfg.include_pareto_from_run,
        run_root=ctx.run_root,
        override_name=eff_name,
        override_seed=eff_seed,
    ).reset_index(drop=True)

    # 7) Subset-Markierungen
    probes = X_all.copy()
    probes["sel_fast"] = 0
    probes["sel_gold"] = 0

    # 8) FAST/GOLD-Subsets
    df_fast = None
    if vcfg.teacher_mode in ("fast", "fast+gold"):
        fs = vcfg.fast_subset.get("strategy", "all")
        fk = vcfg.fast_subset.get("k_total")
        df_fast = pick_fast(X_all, fs, fk, ctx.xschema, rng_subset)
        probes.loc[df_fast.index, "sel_fast"] = 1

    df_gold = None
    if vcfg.teacher_mode in ("gold", "fast+gold"):
        gs = vcfg.gold_subset.get("strategy", "mixed")
        gk = vcfg.gold_subset.get("k_total")
        df_gold = pick_gold(X_all, gs, gk, ctx.xschema, rng_subset)
        probes.loc[df_gold.index, "sel_gold"] = 1

        if vcfg.teacher_mode == "fast+gold" and vcfg.fast_subset.get("strategy") == "match_gold":
            df_fast = df_gold.copy()
            probes["sel_fast"] = probes["sel_gold"]

    R.save_probes(out, probes)

    # 9) Surrogat über alle Probes
    f_s, k_s = run_surrogate(X_all, params, ctx.xschema.names)

    def tidy_frame(name: str, f: pd.DataFrame, k: pd.DataFrame) -> pd.DataFrame:
        f = f.copy(); k = k.copy()
        f.columns = [f"flow_{c}" for c in f.columns]
        k.columns = [f"kpi_{c}" for c in k.columns]
        return pd.concat([f, k], axis=1).assign(model=name)

    tidies = [tidy_frame("surrogate", f_s, k_s)]
    metrics: Dict[str, Any] = {}

    # 10) FAST
    if df_fast is not None and len(df_fast) > 0:
        f_f, k_f = run_fast(df_fast, params, ctx.xschema.names)
        metrics["fast"] = compute(f_s.loc[df_fast.index], k_s.loc[df_fast.index], f_f, k_f)
        R.plot_scatter(out, k_s.loc[df_fast.index], k_f, "FAST")
        R.plot_front(out,   k_s.loc[df_fast.index], k_f, "FAST")
        tidies.append(tidy_frame("fast", f_f, k_f))

    # 11) GOLD
    if df_gold is not None and len(df_gold) > 0:
        f_g, k_g = run_gold(df_gold, params, ctx.xschema.names)
        metrics["gold"] = compute(f_s.loc[df_gold.index], k_s.loc[df_gold.index], f_g, k_g)
        R.plot_scatter(out, k_s.loc[df_gold.index], k_g, "GOLD")
        R.plot_front(out,   k_s.loc[df_gold.index], k_g, "GOLD")
        tidies.append(tidy_frame("gold", f_g, k_g))

    # 12) Sammeln & persistieren
    tidy_all = pd.concat(tidies, axis=0, ignore_index=True)
    R.save_predictions(out, tidy_all)
    R.save_metrics(out, metrics)
    R.save_report_md(
        out,
        {
            "run_id": ctx.run_meta["run_id"],
            "teacher_mode": vcfg.teacher_mode,
            "n_all": int(len(X_all)),
            "n_fast": int(probes["sel_fast"].sum()),
            "n_gold": int(probes["sel_gold"].sum()),
            "params_hash": ctx.params_hash,
        },
        metrics,
    )

    dbg(f"Validation done → {out}")
    return out


if __name__ == "__main__":
    validate()
