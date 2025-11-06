# Optimization/framework/Orchestrator/registry.py

# Wichtig: exakt diese Pfade, gemäß deiner Ordnerstruktur
from Optimization.framework.engines.Gold.gold_engine import GoldEngine
from Optimization.framework.engines.Vectorized_model.fast_engine import FastEngine
from Optimization.framework.engines.Surrogat_model.surrogate_engine import SurrogateEngine

_ENGINE_REG = {
    "gold": GoldEngine,
    "fast": FastEngine,
    "surrogate": SurrogateEngine,
}

def resolve_engine(name: str):
    import inspect
    key = (name or "").lower()
    if key not in _ENGINE_REG:
        raise ValueError(f"[registry] Unknown engine '{name}'. Available: {list(_ENGINE_REG)}")
    Eng = _ENGINE_REG[key]
    print(f"[registry] engine='{key}' → {Eng.__name__} @ {inspect.getsourcefile(Eng)}")
    return Eng