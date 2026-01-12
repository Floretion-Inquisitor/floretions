# === centroid distance (versione pct) =========================================

from __future__ import annotations

import math
from typing import Tuple, Literal

import numpy as np

from floretion import Floretion
from lib.floretion_utils.config_paths import data_dir



# cache tabella per ordine
_DIST_CACHE: dict[int, np.ndarray] = {}

def _accum_uvw_from_octal(oct_str: str) -> Tuple[int, int, int]:
    n = len(oct_str)
    s = -1
    U = V = W = 0
    for t, ch in enumerate(oct_str, start=1):
        if ch == '7':
            s = -s
        elif ch == '1':
            U += s * (1 << (n - t))
        elif ch == '2':
            V += s * (1 << (n - t))
        elif ch == '4':
            W += s * (1 << (n - t))
        else:
            raise ValueError(f"Digit ottale non valido: {ch}")
    return U, V, W

def _equilateral_norm2(U: int, V: int, W: int) -> int:
    return U*U + V*V + W*W - (U*V + V*W + W*U)

def _distance_invariants_table(order: int) -> np.ndarray:
    """
    Ritorna array dtype=object con colonne:
      [ base_dec(int), base_oct(str), U(int), V(int), W(int), S(int), dist_norm(float) ]
    dove dist_norm = sqrt(S)/2^(order-1).
    """
    cached = _DIST_CACHE.get(order)
    if cached is not None:
        return cached

    zero = 0 * Floretion.from_string('1' + 'e' * order)
    bases = zero.base_vec_dec_all.astype(int)

    rows = []
    scale = 1 << max(order - 1, 0)  # 2^(order-1)
    for dec in bases:
        oct_str = format(int(dec), 'o')
        U, V, W = _accum_uvw_from_octal(oct_str)
        S = _equilateral_norm2(U, V, W)
        dist = (math.sqrt(float(S)) / scale) if order > 0 else 0.0
        rows.append((int(dec), oct_str, U, V, W, int(S), float(dist)))

    arr = np.array(rows, dtype=object)
    _DIST_CACHE[order] = arr
    return arr

def export_distances_table(order: int, save_csv: bool = False) -> str:
    """
    Salva in data/distances/distances_order_{n}.npy (+ opzionale .csv) le colonne:
      base_dec, base_oct, U, V, W, S_equilateral, dist_norm
    """
    arr = _distance_invariants_table(order)
    out_dir = data_dir() / "distances"
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = (out_dir / f"distances_order_{order}.npy").resolve()
    np.save(str(npy_path), arr)
    if save_csv:
        import csv
        csv_path = str(npy_path).replace(".npy", ".csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["base_dec","base_oct","U","V","W","S_equilateral","dist_norm"])
            w.writerows(arr.tolist())
    return str(npy_path)



def flo_from_centroid_distance(
    *,
    order: int,
    pct: float,
    relation: Literal["equal", "le", "lt", "ge", "gt"] = "equal",
    coeff: float | str = 1.0,
) -> Floretion:
    """
    Seleziona i base-vectors di un dato ordine in base alla distanza dal centro,
    confrontata con una soglia pari a 'pct' percento della distanza massima d_max.

    Parametri
    ---------
    order : int
        Ordine della Floretion (>=1).
    pct : float
        Percentuale di d_max (0..100). 0 -> centro, 100 -> massima distanza possibile.
    relation : {"equal","le","lt","ge","gt"}
        Confronto rispetto alla soglia:
          - equal: prende la/e distanza/e *disponibile/i* più vicina/e alla soglia (usa S intero)
          - le/lt/ge/gt: confronto stretto su S rispetto alla soglia convertita.
    coeff : float | "dist"
        Se numero: coeff costante. Se "dist": coeff = distanza normalizzata di ciascun base-vector.

    Ritorna
    -------
    Floretion con i termini selezionati (zero-floretion se nessun match).
    """
    if order < 0:
        raise ValueError("order deve essere >= 0")
    # clamp pct nella fascia [0,100]
    pct = max(0.0, min(100.0, float(pct)))

    arr = _distance_invariants_table(order)  # [dec, oct, U,V,W, S, dist]
    S_all = arr[:, 5].astype(int)
    S_max = int(S_all.max()) if S_all.size else 0

    # Se d = sqrt(S)/2^(n-1), allora (d / d_max)^2 = S / S_max.
    # Quindi la soglia in termini di S è:
    S_thr_real = (pct / 100.0) ** 2 * float(S_max)

    rel = relation.lower().strip()
    eps = 1e-12

    if rel == "equal":
        # Trova l’S disponibile più vicino a S_thr_real
        S_unique = np.unique(S_all)
        # indice del minimo scarto
        j = int(np.argmin(np.abs(S_unique.astype(float) - S_thr_real)))
        S_target = int(S_unique[j])
        mask = (S_all == S_target)
    elif rel == "le":
        S_thr = math.floor(S_thr_real + eps)
        mask = (S_all <= S_thr)
    elif rel == "lt":
        S_thr = math.ceil(S_thr_real - eps)
        mask = (S_all < S_thr)
    elif rel == "ge":
        S_thr = math.ceil(S_thr_real - eps)
        mask = (S_all >= S_thr)
    elif rel == "gt":
        S_thr = math.floor(S_thr_real + eps)
        mask = (S_all > S_thr)
    else:
        raise ValueError(f"relation non riconosciuta: {relation}")

    sel = arr[mask]
    if sel.size == 0:
        return 0 * Floretion.from_string('1' + 'e' * order)

    base_dec = sel[:, 0].astype(int)

    if isinstance(coeff, (int, float)):
        coeffs = np.full(base_dec.shape[0], float(coeff), dtype=float)
    else:
        if str(coeff).lower() in ("dist", "distance"):
            coeffs = sel[:, 6].astype(float)  # dist_norm
        else:
            raise ValueError(f"coeff non valido: {coeff} (usa numero o 'dist')")

    return Floretion(coeffs, base_dec)



def get_basevec_coords(base_vector_oct: str):
    x = 0.0
    y = 0.0
    distance = 1.0
    sign_distance = -1.0

    for digit in base_vector_oct:
        if digit == "7":
            sign_distance *= -1.0
        else:
            if digit == "4":
                angle = 210
            elif digit == "2":
                angle = 90
            elif digit == "1":
                angle = 330
            else:
                raise ValueError(f"Invalid digit {digit} in base vector.")
            x += math.cos(math.radians(angle)) * distance * sign_distance
            y += math.sin(math.radians(angle)) * distance * sign_distance
        distance /= 2.0

    # per avere la stessa disposizione visuale di Triangleize in Blender:
    return [x, -y]



def get_base_vec_centroid_dist(base_vector_oct: str) -> float:
    """Back-compat: distanza dal centro."""
    x, y = get_basevec_coords(base_vector_oct)
    return math.hypot(x, y)


#for m in range(1,8):
#    export_distances_table(m, save_csv=False)
#    export_distances_table(m, save_csv=True)