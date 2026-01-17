from __future__ import annotations

"""
config_paths.py
---------------
Utility centralizzate per:
- individuare la root del progetto e le directory dati/output/config;
- caricare il config JSON;
- costruire i path per CSV/Numpy (mono/segmenti);
- leggere la strategia di moltiplicazione per ordine dal config;
- iterare i segmenti .npy quando richiesti;
- stampare un riepilogo della struttura attesa.

Struttura progetto attesa (relativa alla root):
  /config/config.json
  /data/
      grid.flo_{n}.oct.csv
      /npy/
          /order_{n}/
              floretion_order_{n}_segment_000_indices.npy
              floretion_order_{n}_segment_000_signs.npy
          /order_{n}.segments/
              floretion_order_{n}_segment_000_indices.npy
              floretion_order_{n}_segment_000_signs.npy
              ...
      /centers/
          /order_{n}/
              /both/
                  centers_order_{n}_segment_000.npy
                  centers_order_{n}_segment_000.json
  /output/    (opzionale)
"""

from pathlib import Path
from functools import lru_cache
from typing import Iterator, Tuple, Optional
import os
import json
import re


# ---------------------------------------------------------------------
# Localizzazione root e cartelle principali
# ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def project_root() -> Path:
    """
    Restituisce la root del progetto: due livelli sopra questo file.
    lib/floretion_utils/config_paths.py -> parents[2] == root
    """
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def config_dir() -> Path:
    return project_root() / "config"


@lru_cache(maxsize=None)
def data_dir() -> Path:
    return project_root() / "data"


@lru_cache(maxsize=None)
def output_dir() -> Path:
    return project_root() / "output"


# ---------------------------------------------------------------------
# Config JSON
# ---------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_config() -> dict:
    """
    Carica config/config.json oppure il file puntato da ENV FLORETION_CONFIG.
    Se assente o invalido, restituisce {}.
    """
    env_path = os.environ.get("FLORETION_CONFIG", "").strip()
    cfg_path = Path(env_path) if env_path else (config_dir() / "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------
# Path helper per dati CSV/NPY
# ---------------------------------------------------------------------

def get_grid_csv(order: int) -> Path:
    return data_dir() / f"grid.flo_{int(order)}.oct.csv"


def _npy_base(order: int) -> Path:
    return data_dir() / "npy" / f"order_{int(order)}"


def _npy_segments_dir(order: int) -> Path:
    return data_dir() / "npy" / f"order_{int(order)}.segments"


def _npy_mono_paths(order: int) -> Tuple[Path, Path]:
    """
    Ritorna (indices.npy, signs.npy) per il file monolitico segment_000.
    """
    base = _npy_base(order)
    ind = base / f"floretion_order_{int(order)}_segment_000_indices.npy"
    sgn = base / f"floretion_order_{int(order)}_segment_000_signs.npy"
    return ind, sgn


# ---------------------------------------------------------------------
# Strategie di moltiplicazione da config
# ---------------------------------------------------------------------

def get_mult_strategy(order: int) -> str:
    """
    Restituisce la strategia per l'ordine indicato dal config JSON.
    Esempi: "direct", "npy-mono", "npy-segment-num-64".
    Alias: "npy" -> "npy-mono".
    """
    cfg = load_config()
    key = f"order_{int(order)}_mult_strategy"
    val = str(cfg.get(key, "direct"))
    if val == "npy":
        return "npy-mono"
    return val


def parse_segment_num(strategy: str) -> Optional[int]:
    """
    Estrae K da stringhe del tipo "npy-segment-num-K". Se non combacia, None.
    """
    m = re.match(r"^npy-segment-num-(\d+)$", str(strategy))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------
# Scoperta/iterazione segmenti NPY
# ---------------------------------------------------------------------

def get_npy_strategy(order: int) -> dict:
    """
    Auto-detect dei file .npy disponibili.
    Se esiste directory .segments con coppie *_indices.npy/*_signs.npy -> "segments".
    Altrimenti, se esistono i monolitici segment_000 -> "mono".
    Altrimenti ritorna "mono" con path non esistenti (il chiamante deciderà il fallback).
    """
    segdir = _npy_segments_dir(order)
    if segdir.exists():
        inds = sorted(segdir.glob(f"floretion_order_{int(order)}_segment_*_indices.npy"))
        sgns = sorted(segdir.glob(f"floretion_order_{int(order)}_segment_*_signs.npy"))
        if inds and sgns:
            sgn_map = {p.name.replace("_signs.npy", ""): p for p in sgns}
            pairs = []
            for ip in inds:
                key = ip.name.replace("_indices.npy", "")
                sp = sgn_map.get(key)
                if sp:
                    pairs.append((ip, sp))
            if pairs:
                return {"mode": "segments", "segments": pairs}

    ind, sgn = _npy_mono_paths(order)
    return {"mode": "mono", "mono": (ind, sgn)}


def iter_npy_segments(order: int, limit: Optional[int] = None) -> Iterator[Tuple[Path, Path, None]]:
    """
    Iteratore dei segmenti .npy (indices, signs). Se limit è impostato, restituisce solo i primi `limit` segmenti.
    In assenza di segmenti, restituisce la coppia monolitica (segment_000) una sola volta.
    """
    strat = get_npy_strategy(order)
    if strat["mode"] == "segments":
        pairs = strat["segments"]
        if limit is not None:
            pairs = pairs[:int(limit)]
        for ip, sp in pairs:
            yield ip, sp, None
    else:
        ip, sp = strat["mono"]
        yield ip, sp, None


# ---------------------------------------------------------------------
# Riepilogo struttura attesa (per messaggi di errore)
# ---------------------------------------------------------------------

def expected_layout_summary(order: int) -> str:
    root = project_root()
    grid_csv = get_grid_csv(order)

    mono_ind, mono_sgn = _npy_mono_paths(order)
    seg_dir = _npy_segments_dir(order)
    seg_example_i = seg_dir / f"floretion_order_{int(order)}_segment_000_indices.npy"
    seg_example_s = seg_dir / f"floretion_order_{int(order)}_segment_000_signs.npy"

    centers_npy = data_dir() / "centers" / f"order_{int(order)}" / "both" / f"centers_order_{int(order)}_segment_000.npy"
    centers_json = data_dir() / "centers" / f"order_{int(order)}" / "both" / f"centers_order_{int(order)}_segment_000.json"

    lines = [
        "Floretion – struttura progetto attesa",
        f"Root progetto: {root}",
        "",
        "Obbligatori (per il solo caricamento griglia):",
        f"  - {grid_csv}",
        "",
        "Opzionali (accelerazione moltiplicazione, modalità mono):",
        f"  - {mono_ind}",
        f"  - {mono_sgn}",
        "",
        "Opzionali (accelerazione moltiplicazione, modalità segmenti):",
        f"  - Directory: {seg_dir}",
        "    Contenuti es.:",
        f"    - {seg_example_i}",
        f"    - {seg_example_s}",
        "",
        "Opzionali (centri):",
        f"  - {centers_npy}",
        f"  - {centers_json}",
        "",
        "Nota: tutti i percorsi sono relativi alla root del progetto.",
    ]
    return "\n".join(lines)
