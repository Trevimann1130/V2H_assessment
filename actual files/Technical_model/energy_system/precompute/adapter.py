# Technical_model/energy_system/precompute/adapter.py
from __future__ import annotations
from typing import Tuple, Dict, Any

# Daten laden (passe Pfad an, falls deine Data-Schicht anders heißt)
from Data.data import get_parameters, load_profiles

# Lokales Jahres-Precompute (liegt im selben Ordner)
from .precompute import prepare_profiles as _prepare


def prepare_profiles_adapter(location: str,
                             data_source: str | None = None
                             ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Dünner Adapter für den Runner:
      - lädt params + Rohprofile
      - ruft das lokale Jahres-Precompute
      - gibt (params, prepared_profiles) zurück
    """
    # 1) Basis-Parameter
    params = dict(get_parameters(location))
    params["location"] = location

    # 2) Roh-Profile (mit/ohne data_source – strikt)
    if data_source is None:
        profiles_raw = load_profiles(location)
    else:
        profiles_raw = load_profiles(location, data_source=data_source)

    # 3) 1-Jahres-Precompute
    prepared = _prepare(
        params=params,
        profiles=profiles_raw,
        do_hp_electricity=True,
        do_coeffs=False,
    )

    return params, prepared
