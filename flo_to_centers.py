# flo_to_centers.py
from __future__ import annotations

import os
from multiprocessing import Pool, cpu_count, freeze_support
from pathlib import Path
from typing import Dict, List, Tuple, Literal

import numpy as np
from floretion import Floretion

from lib.floretion_utils.floretion_ops import find_center_base_vectors_only
from lib.floretion_utils.floretion_centers import (
    Mode,
    centers_dir,
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DEFAULT_CORES = max(1, cpu_count())

# Naming (SEMPRE con range finale):
#   centers_order_{n}_segment_{seg:03d}.{START_OCT}-{END_OCT}.{npy|json}
SEG_FMT = "centers_order_{order}_segment_{seg:03d}.{start_oct}-{end_oct}.{ext}"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _oct_zfill(dec_val: int, order: int) -> str:
    """
    Converte in ottale e forza lunghezza = order (padding a sinistra con '0').
    Nota: per base-vectors validi la lunghezza dovrebbe già essere = order, ma
    questo rende il naming stabile e comparabile.
    """
    return format(int(dec_val), "o").rjust(int(order), "0")


def _segment_bounds(order: int, total_segments: int, seg: int) -> Tuple[int, int]:
    """
    Restituisce (start_idx, end_idx) (end esclusivo) sulla griglia ordinata
    di 4**order base-vectors.
    """
    n = 4 ** int(order)
    segs = max(1, int(total_segments))
    per = (n + segs - 1) // segs  # ceil
    start = int(seg) * per
    end = min((int(seg) + 1) * per, n)
    return start, end


def _compute_centers_for_slice(args) -> str:
    order, mode, seg, total_segments, ext = args
    mode = str(mode).strip()

    start, end = _segment_bounds(order, total_segments, seg)
    if start >= end:
        return f"[SKIP] seg={seg:03d} empty"

    unit = Floretion.from_string(f'1{"e"*int(order)}')
    base_all = unit.base_vec_dec_all

    # calcola mapping {base_dec_str : [centers...]}
    out_map: Dict[str, List[int]] = {}
    for idx in range(start, end):
        base_dec = int(base_all[idx])
        f = Floretion(np.array([1.0], dtype=float), np.array([base_dec], dtype=int))
        centers = find_center_base_vectors_only(f, decomposition_type=mode)
        out_map[str(base_dec)] = sorted(map(int, centers))

    # naming con intervallo in ottale (umano) e lunghezza fissa = order
    start_oct = _oct_zfill(int(base_all[start]), order)
    end_oct = _oct_zfill(int(base_all[end - 1]), order)

    outdir = centers_dir(int(order), mode)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / SEG_FMT.format(
        order=int(order),
        seg=int(seg),
        start_oct=start_oct,
        end_oct=end_oct,
        ext=str(ext),
    )

    if out_path.suffix.lower() == ".npy":
        np.save(str(out_path), out_map, allow_pickle=True)
    else:
        out_path.write_text(_json_dumps(out_map), encoding="utf-8")

    return f"[OK] order={order} mode={mode} seg={seg:03d} [{start_oct}..{end_oct}] -> {out_path.name}"


def _json_dumps(d: Dict) -> str:
    import json
    return json.dumps(d, indent=2)


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------

def save_centers_segmented(
    order: int,
    *,
    mode: Mode = "both",
    total_segments: int = 64,
    fmt: Literal["npy", "json"] = "npy",
    cores_per_batch: int = DEFAULT_CORES,
) -> None:
    """
    Salva i centers per un ordine in più file segmentati con nome:
      centers_order_{n}_segment_{XXX}.{START_OCT}-{END_OCT}.{npy|json}

    Tipico:
      - order 7 -> 64 segmenti
      - order 8 -> 256 segmenti
    """
    segs = max(1, int(total_segments))
    tasks = [(int(order), str(mode), seg, segs, fmt) for seg in range(segs)]
    for batch in range(0, len(tasks), max(1, int(cores_per_batch))):
        chunk = tasks[batch:batch + max(1, int(cores_per_batch))]
        with Pool(processes=len(chunk)) as pool:
            for msg in pool.imap_unordered(_compute_centers_for_slice, chunk):
                print(msg)


def save_centers_singlefile(
    order: int,
    *,
    mode: Mode = "both",
    fmt: Literal["npy", "json"] = "npy",
) -> Path:
    """
    Per ordini piccoli (assumiamo 1..6): scrive un UNICO file, ma sempre col suffix range:

      centers_order_{n}_segment_000.{11..11}-{77..77}.{npy|json}

    dove la lunghezza dei blocchi dipende da 'order' (es. order=5 -> 11111 e 77777).
    """
    order = int(order)
    outdir = centers_dir(order, str(mode))
    outdir.mkdir(parents=True, exist_ok=True)

    unit = Floretion.from_string(f'1{"e"*order}')
    base_all = unit.base_vec_dec_all

    mapping: Dict[str, List[int]] = {}
    for dec in map(int, base_all):
        f = Floretion(np.array([1.0], dtype=float), np.array([dec], dtype=int))
        centers = find_center_base_vectors_only(f, decomposition_type=str(mode))
        mapping[str(dec)] = sorted(map(int, centers))

    # range logico completo: 111..111 -> 777..777 (lunghezza = order)
    start_oct = "1" * order
    end_oct = "7" * order

    out = outdir / SEG_FMT.format(
        order=order,
        seg=0,
        start_oct=start_oct,
        end_oct=end_oct,
        ext=fmt,
    )

    if fmt == "npy":
        np.save(str(out), mapping, allow_pickle=True)
    else:
        out.write_text(_json_dumps(mapping), encoding="utf-8")

    print(f"[OK] single file -> {out}")
    return out


# -----------------------------------------------------------------------------
# CLI demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Necessario su Windows quando si usa multiprocessing
    freeze_support()

    # Esempi:
    #   python flo_to_centers.py
    print("[demo] generating sample centers...")

    # Ordini piccoli -> file unico (con range in filename)
    for n in (1, 2, 3):
        save_centers_singlefile(n, mode="both", fmt="npy")

    # Ordine 7 segmentato (64)
    # save_centers_segmented(7, mode="both", total_segments=64, fmt="npy")

    # Ordine 8 segmentato (256)
    # save_centers_segmented(8, mode="both", total_segments=256, fmt="npy")
