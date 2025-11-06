# Optimization/validation/build/settings_bridge.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import sys
import importlib
import importlib.util as _imp_util
import importlib.machinery as _imp_mach
import os

from ..utils import dbg, read_json, make_seed, XSchema


@dataclass
class ValidationConfig:
    run_id: str
    teacher_mode: str
    random_seed: int
    sampler_mode: str
    sampler_name: Optional[str]
    sampler_seed: Optional[int]
    sampler_params: Optional[Dict[str, Any]]
    probes_strategy: str
    probes_n_total: int
    include_pareto_from_run: bool
    fast_subset: Dict[str, Any]
    gold_subset: Dict[str, Any]
    reporting: Dict[str, Any]
    cache: Dict[str, Any]

@dataclass
class BuiltContext:
    runs_root: Path
    run_root: Path
    run_meta: Dict[str, Any]
    params_hash: str
    scenario_signature: str
    xschema: XSchema
    seeds_validation: Dict[str, int]
    seeds_run: Dict[str, int]
    sampler_from_run: Dict[str, Any]

# ----------------------------
# Settings laden (robust)
# ----------------------------

def _load_settings():
    env_mod = os.getenv("SETTINGS_MODULE")
    candidates = [env_mod,
                  "Optimization.framework.Settings.settings",
                  "Optimization.settings",
                  "settings"]
    candidates = [c for c in candidates if c]

    last_err = None
    for mod in candidates:
        try:
            m = importlib.import_module(mod)
            dbg(f"Resolved settings module: {mod}")
            if hasattr(m, "validation") or hasattr(m, "paths"):
                return m
            if hasattr(m, "SETTINGS"):
                return m.SETTINGS
            if hasattr(m, "get_settings"):
                return m.get_settings()
            return m
        except Exception as e:
            last_err = e
            continue

    # Fallback: direkt aus Datei relativ zur Projektwurzel
    here = Path(__file__).resolve()
    proj_root = here.parents[3]
    for rel in [
        Path("Optimization/framework/Settings/settings.py"),
        Path("Optimization/settings.py"),
        Path("settings.py"),
    ]:
        f = (proj_root / rel).resolve()
        if f.exists():
            dbg(f"Loading settings from file: {f}")
            spec = _imp_util.spec_from_loader(
                "settings_file_loader",
                _imp_mach.SourceFileLoader("settings_file_loader", str(f))
            )
            mod = _imp_util.module_from_spec(spec)
            assert spec and spec.loader
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            if hasattr(mod, "validation") or hasattr(mod, "paths"):
                return mod
            if hasattr(mod, "SETTINGS"):
                return mod.SETTINGS
            if hasattr(mod, "get_settings"):
                return mod.get_settings()
            return mod

    raise RuntimeError("Cannot load central settings.") from last_err


# ----------------------------
# Utilities
# ----------------------------

def _req(obj, name: str):
    if not hasattr(obj, name):
        raise ValueError(f"settings.validation.{name} missing.")
    return getattr(obj, name)

def _as_dict(maybe_obj):
    return dict(maybe_obj.__dict__) if hasattr(maybe_obj, "__dict__") else dict(maybe_obj)

def _resolve_run_id(run_id: str, runs_root_path: Path) -> str:
    if str(run_id).lower() != "latest":
        return run_id

    txt = runs_root_path / "_LATEST_RUN.txt"
    jsn = runs_root_path / "_LATEST_RUN.json"

    if txt.exists():
        rid = txt.read_text(encoding="utf-8").strip()
        if rid:
            return rid
    if jsn.exists():
        try:
            obj = read_json(jsn)
            rid = (obj or {}).get("run_id")
            if rid:
                return rid
        except Exception:
            pass

    # Neuester Unterordner
    subdirs = [p.name for p in runs_root_path.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No runs found under {runs_root_path}")
    subdirs.sort()
    return subdirs[-1]

def _resolve_meta_path(run_id: str, runs_root: Path) -> tuple[Path, Path]:
    run_root = (runs_root / run_id).resolve()
    cand1 = run_root / "run_meta.json"
    cand2 = run_root / "meta.json"
    if cand1.exists():
        return run_root, cand1
    if cand2.exists():
        return run_root, cand2
    tried = [str(cand1), str(cand2)]
    raise FileNotFoundError(
        f"Could not locate run metadata for run_id={run_id}. "
        f"Looked under: {tried} (accepts run_meta.json or meta.json)"
    )


# ----------------------------
# Hauptfunktion
# ----------------------------

def build_context_from_run_id() -> Tuple[ValidationConfig, BuiltContext]:
    s = _load_settings()
    v = _req(s, "validation")

    teacher_mode = _req(v, "teacher_mode")
    run_id_input = _req(v, "run_id")
    random_seed  = int(_req(v, "random_seed"))

    sampler      = _req(v, "sampler")
    sampler_mode = getattr(sampler, "mode", None)
    sampler_name = getattr(sampler, "name", None)
    sampler_seed = getattr(sampler, "seed", None)
    sampler_params = getattr(sampler, "params", None)

    probes       = _req(v, "probes")
    probes_strategy = getattr(probes, "strategy", None)
    probes_n_total  = int(getattr(probes, "n_total", 0))
    include_pareto  = bool(getattr(probes, "include_pareto_from_run", False))

    fast_subset = _as_dict(_req(v, "fast_subset"))
    gold_subset = _as_dict(_req(v, "gold_subset"))
    reporting   = _as_dict(_req(v, "reporting"))
    cache       = _as_dict(getattr(v, "cache", {"enabled": True, "force_refresh": False}))

    if teacher_mode not in ("fast", "gold", "fast+gold"):
        raise ValueError("teacher_mode must be 'fast'|'gold'|'fast+gold'")
    if sampler_mode not in ("match_run", "override"):
        raise ValueError("sampler.mode must be 'match_run'|'override'")
    if probes_n_total <= 0:
        raise ValueError("probes.n_total must be > 0")

    # runs_root bestimmen (absolut vs relativ zum Projektroot)
    paths_obj = getattr(s, "paths", None)
    if paths_obj is None or not getattr(paths_obj, "runs_root", None):
        raise ValueError("settings.paths.runs_root missing.")

    raw = str(getattr(paths_obj, "runs_root")).strip()
    project_root = Path(__file__).resolve().parents[3]

    is_abs = Path(raw).is_absolute() or os.path.isabs(raw) or (len(raw) >= 2 and raw[1] == ":")
    runs_root_path = Path(raw) if is_abs else (project_root / raw)
    runs_root_path = runs_root_path.resolve()

    if not runs_root_path.exists():
        raise FileNotFoundError(
            f"settings.paths.runs_root not found:\n  {runs_root_path}\n"
            f"(raw='{raw}', project_root='{project_root}')"
        )

    # run_id ggf. "latest"
    run_id = _resolve_run_id(run_id_input, runs_root_path)

    # Meta laden
    run_root, meta_path = _resolve_meta_path(run_id, runs_root_path)
    run_meta  = read_json(meta_path)

    params_hash = str(run_meta.get("params_hash") or "")
    if not params_hash:
        raise ValueError("run_meta.json missing params_hash")

    scenario_signature = f"{run_meta.get('scenario_name','')}|{run_meta.get('location','')}"
    if "|" not in scenario_signature or scenario_signature.startswith("|"):
        raise ValueError("run_meta.json missing scenario_name/location")

    xschema = XSchema.from_meta(run_meta)

    seeds_validation = {
        "sampler": make_seed(random_seed, "sampler", scenario_signature),
        "subset_selection": make_seed(random_seed, "subset_selection", scenario_signature),
    }
    seeds_run = run_meta.get("seeds_run") or {}
    sampler_from_run = run_meta.get("sampler_signature") or {}

    dbg(f"Loaded run_id={run_id} ({meta_path})")
    dbg(f"Params hash: {params_hash}")
    return (
        ValidationConfig(
            run_id=run_id,
            teacher_mode=teacher_mode,
            random_seed=random_seed,
            sampler_mode=sampler_mode,
            sampler_name=sampler_name,
            sampler_seed=(None if sampler_seed is None else int(sampler_seed)),
            sampler_params=sampler_params,
            probes_strategy=probes_strategy,
            probes_n_total=probes_n_total,
            include_pareto_from_run=include_pareto,
            fast_subset=fast_subset,
            gold_subset=gold_subset,
            reporting=reporting,
            cache=cache
        ),
        BuiltContext(
            runs_root=runs_root_path,
            run_root=run_root,
            run_meta=run_meta,
            params_hash=params_hash,
            scenario_signature=scenario_signature,
            xschema=xschema,
            seeds_validation=seeds_validation,
            seeds_run=seeds_run,
            sampler_from_run=sampler_from_run
        )
    )
