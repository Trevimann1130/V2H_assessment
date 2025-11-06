from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json, time, hashlib
from datetime import datetime

def dbg(msg: str) -> None:
    print(f"[validation] {msg}")

def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def read_json(path: Path | str) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path | str, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def write_text(path: Path | str, text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")

@dataclass
class XSchema:
    names: List[str]
    lower: List[float]
    upper: List[float]

    @staticmethod
    def from_meta(run_meta: Dict[str, Any]) -> "XSchema":
        xs = run_meta.get("x_schema") or {}
        names = xs.get("names") or (run_meta.get("settings_digest") or {}).get("bounds", {}).get("names")
        lower = (xs.get("bounds") or {}).get("lower") or (run_meta.get("settings_digest") or {}).get("bounds", {}).get("lower")
        upper = (xs.get("bounds") or {}).get("upper") or (run_meta.get("settings_digest") or {}).get("bounds", {}).get("upper")
        if not names or lower is None or upper is None:
            raise ValueError("run_meta.json missing x_schema {names,bounds.lower,bounds.upper}")
        return XSchema(names=list(names), lower=list(lower), upper=list(upper))

def make_seed(master_seed: int, purpose: str, scenario_signature: str) -> int:
    h = hashlib.sha1(f"{master_seed}|{purpose}|{scenario_signature}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)
