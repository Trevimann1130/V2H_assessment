# Optimization/framework/Settings/settings.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ======================================================================
#   SETTINGS – zentrale, schlanke Konfiguration (ohne Fallbacks)
# ======================================================================
# Namenskonventionen:
#   engine.name   ∈ {"gold", "fast", "surrogate"}
#   sampler.name  ∈ {"lhs", "sobol", "random"}
#   optimizer.name∈ {"nsga2", "nsga3", "moead", "agemoea", "smsemoa"}
#
# Hinweise:
# - Alle numerischen Werte werden so verwendet, wie du sie hier setzt.
# - Keine versteckten Defaults irgendwo im Code.
# - Zielgrößen (surrogate_train.targets) sind LEBENSDAUER-Summen der Flüsse.
# ======================================================================

# ---------- Problem/Design Space ----------
@dataclass
class Bounds:
    # Reihenfolge der Namen muss der Spaltenreihenfolge in X entsprechen!
    names: List[str]                  # z.B. ["pv_kwp", "bess_kwh"]
    lower: List[float]
    upper: List[float]
    steps: Optional[List[float]] = None  # Schrittweite pro Variable (None = beliebig fein)

@dataclass
class Objectives:
    # z.B. ["npc_eur", "pef_pt"], beide minimieren
    names: List[str]
    minimize: List[bool]              # True=minimize, False=maximize

@dataclass
class Constraints:
    # Optional – aktuell leer gelassen; bei Bedarf wie Objectives bestücken
    names: List[str] = field(default_factory=list)
    senses: List[str] = field(default_factory=list)   # "<=" oder ">="
    rhs:   List[float] = field(default_factory=list)

# ---------- Engine-Auswahl & Szenario ----------
@dataclass
class EngineConfig:
    # Engine/Modus: "gold" (langsam, “Ground Truth”), "fast" (vektorisiert),
    #               "surrogate" (RandomForest-Surrogat + post-hoc Objectives)
    name: str
    # System-Topologie: z.B. "PV_BESS_HP_V2H" (mit V2H) oder "PV_BESS_HP_EV" (ohne V2H)
    system_id: str
    # Standortbezeichner, wird in Data.data → Profile & Parameter aufgelöst
    location: str
    # V2H-Schalter (True = EV kann netzdienlich ins Haus entladen)
    use_v2h: bool

    # Energy-Community Parameter
    ec_share_import: float            # 0..1 Anteil der Last, die aus EC gedeckt werden darf
    ec_share_export: float            # 0..1 Anteil der Überschüsse, die in EC abgegeben werden

    # Skalierungsparameter des Szenarios
    N_HH: int                         # Anzahl Haushalte
    N_EV_total: int                   # Anzahl EV gesamt
    N_EV_bidirectional: int           # davon bi-direktional (für V2H relevant)

    # Reproduzierbarkeit
    rng_seed: int

    # Optional: existierendes Surrogat-Artefakt (joblib) laden statt neu trainieren
    surrogate_artifact_path: Optional[str] = None

# ---------- Sampler (Designpunkte für Teacher/Surrogat) ----------
@dataclass
class SamplerConfig:
    # name ∈ {"lhs", "sobol", "random"}
    name: str = "lhs"
    # Anzahl Punkte (Training des Surrogats) – siehe Empfehlungen unten
    n_samples: int = 200
    # Saatwert
    seed: int = 50
    # Zusätzliche Optionen:
    #   LHS:   {"criterion": "maximin", "trials": 5}
    #   SOBOL: {"scramble": True, "skip": 0}
    #   RANDOM: {}
    kwargs: Dict = field(default_factory=dict)

# ---------- Surrogat-Training ----------
@dataclass
class SurrogateTrainConfig:
    model_type: str = "rf"            # heute: "rf" (RandomForestRegressor)
    rf_n_estimators: int = 300        # Bäume; 50–100 (Test), 300–600 (final)
    rf_n_jobs: int = -1               # Parallelisierung
    holdout_frac: float = 0.2         # Anteil für Holdout-Validierung
    # Zentrale Liste der zu lernenden Flüsse (LEBENSDAUER-Summen!):
    targets: List[str] = field(default_factory=lambda: [
        "E_import_grid_kWh",
        "E_export_grid_kWh",
        "E_import_ec_pv_kWh",
        "E_import_ec_ev_kWh",
        "E_bess_throughput_kWh",
        "E_ev_charged_kWh",
        "E_ev_discharged_kWh",
        "E_hp_heat_kWh",
        "E_hp_cool_kWh",
        "E_pv_gen_kWh",
    ])

# ---------- Optimizer (pymoo) ----------
@dataclass
class OptimizerConfig:
    # name ∈ {"nsga2", "nsga3", "moead", "agemoea", "smsemoa"}
    name: str = "nsga2"
    # kwargs je Algorithmus:
    #   nsga2 : {"pop_size": 100, "n_gen": 150}
    #   nsga3 : {"pop_size": 100, "n_gen": 150, "ref_dirs": None}
    #   moead : {"n_partitions": 12}   (ref_dirs werden intern aus partitions erzeugt)
    #   agemoea/smsemoa : {"pop_size": 100, "n_gen": 150}
    kwargs: Dict = field(default_factory=lambda: {"pop_size": 100, "n_gen": 150})
    seed: int = 50
    n_jobs: int = 1                    # (falls der Optimierer es nutzt)

# ---------- Reporting ----------
@dataclass
class ReportingConfig:
    auto_report: bool = True
    write_csv: bool = True
    write_plot: bool = True
    write_summary: bool = True
    plot_max_points: Optional[int] = None
    # Basisordner; darunter: <Location>/<Tag>/surrogate_<ts>/ ...
    output_root: str = "Optimization/run/results"

# ---------- Lauf ----------
@dataclass
class RunConfig:
    # Freies Label, geht in die Ordnerstruktur und Validierungs-Spiegelung ein
    tag: str = "paper_run"

# ---------- Master Settings ----------
@dataclass
class Settings:
    run: RunConfig
    engine: EngineConfig
    bounds: Bounds
    objectives: Objectives
    constraints: Constraints
    sampler: SamplerConfig
    optimizer: OptimizerConfig
    reporting: ReportingConfig
    surrogate_train: SurrogateTrainConfig

# ======================================================================
#  EMPFEHLUNGEN
#  • “Schnell testen” → kleiner Sampler, wenige Bäume, kleine Population/Generationen
#  • “Final/robust”   → größerer Sampler, mehr Bäume, mehr Gen/Pop
# ======================================================================
# QUICK TEST (z. B. CI, Funktionalität):
#   sampler = {"name":"lhs","n_samples": 10, "seed": 10, "kwargs":{"criterion":"maximin"}}
#   surrogate_train = {"rf_n_estimators": 100}
#   optimizer = {"name":"nsga2","kwargs":{"pop_size": 50, "n_gen": 20}}
#
# FULL RUN (bessere Qualität):
#   sampler = {"name":"lhs","n_samples": 200..400, "seed": 42, "kwargs":{"criterion":"maximin","trials":5}}
#   (oder "sobol" mit {"scramble":True})
#   surrogate_train = {"rf_n_estimators": 300..600}
#   optimizer = {"name":"nsga2","kwargs":{"pop_size": 150..300, "n_gen": 150..300}}
#   (nsga3 für ≥3 Ziele sinnvoll; für 2 Ziele bringt nsga2 idR. stabilere Fronten)
# ======================================================================

# ======= Beispiel-Default zum direkten Loslegen =======
def get_settings() -> Settings:
    return Settings(
        run=RunConfig(tag="paper_run"),

        # --- Engine/Szenario ---
        engine=EngineConfig(
            name="surrogate",              # "gold" | "fast" | "surrogate"
            system_id="PV_BESS_HP_V2H",    # "PV_BESS_HP_V2H" oder "PV_BESS_HP_EV"
            location="Vienna",
            use_v2h=False,
            ec_share_import=1.0,
            ec_share_export=1.0,
            N_HH=1,
            N_EV_total=0,
            N_EV_bidirectional=0,
            rng_seed=10,
            surrogate_artifact_path=None,  # <- falls vorhanden, wird geladen statt neu trainiert
        ),

        # --- Designraum ---
        bounds=Bounds(
            names=["pv_kwp", "bess_kwh"],
            lower=[0.0, 0.0],
            upper=[1500.0, 1500.0],        # Tipp: upper dynamisch an N_HH koppeln (z. B. 20 kWp/kWh pro HH)
            steps=[1.0, 1.0],              # Rasterung der Variablen (z. B. kWp/kWh in 1er Schritten)
        ),

        # --- Ziele ---
        objectives=Objectives(
            names=["npc_eur", "pef_pt"],   # zulässig: "npc_eur", "pef_pt", optional "grid_import_kwh"
            minimize=[True, True],
        ),

        # --- (derzeit ohne) Constraints ---
        constraints=Constraints(names=[], senses=[], rhs=[]),

        # --- Sampler (Surrogat-Datensatz) ---
        sampler=SamplerConfig(
            name="lhs",                    # "lhs" | "sobol" | "random"
            n_samples=5,                 # QUICK: 10–50 | FULL: 200–400
            seed=5,
            kwargs={"criterion": "maximin", "trials": 5},  # LHS-Optionen
        ),

        # --- Optimierer (pymoo) ---
        optimizer=OptimizerConfig(
            name="nsga3",                  # "nsga2" | "nsga3" | "moead" | "agemoea" | "smsemoa"
            kwargs={"pop_size": 50, "n_gen": 50},        # QUICK: 50/20 | FULL: 150–300 / 150–300
            seed=50,
            n_jobs=1,
        ),

        # --- Reporting ---
        reporting=ReportingConfig(
            auto_report=True,
            write_csv=True,
            write_plot=True,
            write_summary=True,
            plot_max_points=None,
            output_root="Optimization/run/results",
        ),

        # --- Surrogat-Training ---
        surrogate_train=SurrogateTrainConfig(
            model_type="rf",
            rf_n_estimators=100,           # QUICK: 100 | FULL: 300–600
            rf_n_jobs=-1,
            holdout_frac=0.2,
            # Targets hier EINMAL zentral – keine zweite Kopie unten!
            targets=[
                "E_import_grid_kWh",
                "E_export_grid_kWh",
                "E_import_ec_pv_kWh",
                "E_import_ec_ev_kWh",
                "E_bess_throughput_kWh",
                "E_ev_charged_kWh",
                "E_ev_discharged_kWh",
                "E_hp_heat_kWh",
                "E_hp_cool_kWh",
                "E_pv_gen_kWh",
            ],
        ),
    )

# ======================================================================
#                       VALIDATION (Multi-Fidelity)
# ======================================================================
# Pfade (für Validation-Runs). Wenn du bereits woanders pflegst, einfach hier anpassen.
# ... dein bestehendes Settings (get_settings usw.) bleibt unverändert ...

from dataclasses import dataclass, field
from typing import Optional, Dict, List

# =========================
# VALIDATION (Multi-Fidelity)
# =========================

@dataclass
class _Paths:
    # EIN Kanonischer Stammordner
    runs_root: str = "Optimization/run/results"

@dataclass
class _ValSampler:
    mode: str = "match_run"     # "match_run" | "override"
    name: Optional[str] = None
    seed: Optional[int] = None
    params: Optional[Dict] = None

@dataclass
class _ValProbes:
    strategy: str = "match_run" # "lhs"|"random"|"sobol"|"halton"|"corners"|"match_run"
    n_total: int = 40
    include_pareto_from_run: bool = True

@dataclass
class _ValFastSubset:
    strategy: str = "all"       # "all"|"fixed_k"|"edges"|"pareto_focus"|"diverse_kcenter"|"match_gold"
    k_total: int = 30
    buckets: Optional[Dict] = None
    link_to_gold: bool = True

@dataclass
class _ValGoldSubset:
    strategy: str = "mixed"     # "fixed_k"|"mixed"|"pareto_focus"|"top_error"
    k_total: int = 10
    buckets: Optional[Dict] = None
    time_budget_hint: Optional[float] = None

@dataclass
class _ValReporting:
    include_plots: bool = True

@dataclass
class _ValCache:
    enabled: bool = True
    force_refresh: bool = False

@dataclass
class _Validation:
    run_id: str = "latest"              # "latest" oder fixe run_id
    teacher_mode: str = "fast+gold"     # "fast"|"gold"|"fast+gold"
    random_seed: int = 42

    sampler: _ValSampler = field(default_factory=_ValSampler)
    probes: _ValProbes = field(default_factory=_ValProbes)
    fast_subset: _ValFastSubset = field(default_factory=_ValFastSubset)
    gold_subset: _ValGoldSubset = field(default_factory=_ValGoldSubset)
    reporting: _ValReporting = field(default_factory=_ValReporting)
    cache: _ValCache = field(default_factory=_ValCache)

# Modulweite Objekte:
paths = _Paths()
validation = _Validation()


