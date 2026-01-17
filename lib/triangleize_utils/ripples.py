# lib/triangleize_utils/ripples.py

from __future__ import annotations
from typing import List, Tuple, Literal, Optional
from pathlib import Path
import json
import math
import numpy as np
import cv2

# Import *solo* dove serve per evitare cicli
from floretion import Floretion
#from lib.triangleize_utils.centroid_distance import write_centroid_distance_table
from lib.triangleize_utils.coloring import NegPolicy
# Triangleize lo importiamo *dentro* la funzione di render per evitare cicli


# ---------- utilità distanza (carica o calcola) ----------

def _compute_dist_vec_for_order(order: int) -> np.ndarray:
    """
    Calcola la distanza normalizzata dal centro per *tutti* i base-vector dell'ordine.
    Restituisce un vettore float di lunghezza 4^order, nello *stesso ordine della griglia*.
    """
    unit = Floretion.from_string(f'1{"e"*order}')
    bases_dec = unit.base_vec_dec_all.astype(int)
    bases_oct = [format(int(b), "o").rjust(order, "0") for b in bases_dec]

    # passi 1, 1/2, 1/4, ...  (n step)
    steps = np.array([0.5**i for i in range(order)], dtype=float)

    def dist_one(oct_str: str) -> float:
        x = 0.0
        y = 0.0
        sign = -1.0
        for idx, d in enumerate(oct_str):
            if d == "7":
                sign *= -1.0
            else:
                if d == "4":
                    ang = math.radians(210)
                elif d == "2":
                    ang = math.radians(90)
                elif d == "1":
                    ang = math.radians(330)
                else:
                    raise ValueError(f"Digit ottale non valido: {d}")
                step = steps[idx]
                x += math.cos(ang) * step * sign
                y += math.sin(ang) * step * sign
        return math.hypot(x, y)

    dists = np.fromiter((dist_one(s) for s in bases_oct), dtype=float, count=len(bases_oct))
    dmax = float(dists.max()) if dists.size else 1.0
    return (dists / max(dmax, 1e-12)).astype(float)


def _load_or_build_dist_vec(order: int) -> np.ndarray:
    """
    Prova a caricare la tabella salvata (JSON/NPY) con distanze,
    se non esiste la crea e poi restituisce il vettore normalizzato
    nell'ordine della griglia.
    """
    # Percorso tabella ufficiale
    base = Path("data") / "centroid" / f"order_{order}"
    base.mkdir(parents=True, exist_ok=True)

    json_path = base / f"centroid_table.order_{order}.json"
    npy_path  = base / f"centroid_table.order_{order}.npy"

    # tenta NPY
    if npy_path.exists():
        obj = np.load(npy_path, allow_pickle=True).item()
        # mappa dec->idx per ripristinare l'ordine della griglia
        unit = Floretion.from_string(f'1{"e"*order}')
        idx_map = {int(v): i for i, v in enumerate(unit.base_vec_dec_all)}
        dist_vec = np.zeros(len(unit.base_vec_dec_all), dtype=float)
        for k, rec in obj.items():
            dist_vec[idx_map[int(k)]] = float(rec["dist_norm"])
        return dist_vec

    # tenta JSON
    if json_path.exists():
        import json as _json
        data = _json.loads(json_path.read_text(encoding="utf-8"))
        unit = Floretion.from_string(f'1{"e"*order}')
        idx_map = {int(v): i for i, v in enumerate(unit.base_vec_dec_all)}
        dist_vec = np.zeros(len(unit.base_vec_dec_all), dtype=float)
        for k, rec in data.items():
            dist_vec[idx_map[int(k)]] = float(rec["dist_norm"])
        return dist_vec

    # altrimenti calcola e salva
    dist_vec = _compute_dist_vec_for_order(order)

    # salvataggio comodo (NPY “dict-like”)
    unit = Floretion.from_string(f'1{"e"*order}')
    out = {}
    for i, dec in enumerate(unit.base_vec_dec_all):
        out[int(dec)] = {
            "dist_norm": float(dist_vec[i]),
            "base_oct": format(int(dec), "o").rjust(order, "0"),
            "base_dec": int(dec),
        }
    np.save(npy_path, out)

    # JSON più “leggibile”
    try:
        import json as _json
        json_path.write_text(_json.dumps(out, indent=2), encoding="utf-8")
    except Exception:
        pass

    return dist_vec


# ---------- render con ondine radiali ----------

def render_radial_ripples_sequence(
    width: int,
    height: int,
    *,
    order: int,
    total_frames: int,
    # seed di partenza: se None uso tutti 1.0
    seed: Optional[Floretion] = None,
    # parametri onda: d_norm ∈ [0,1]
    wavelength_pct: float = 10.0,          # lunghezza d'onda in % di d_max -> λ_norm = wavelength_pct/100
    speed_pct_per_frame: float = 1.0,      # velocità della cresta (in % di d_max per frame) -> v_norm
    amplitude: float = 0.8,                # ampiezza della modulazione
    bias: float = 0.2,                     # offset >=0 per non azzerare tutto (es. 0.2)
    phase0: float = 0.0,                   # fase iniziale (radians)
    damping_alpha: float = 0.0,            # smorzamento radiale exp(-alpha * d_norm)
    # combinazione con seed
    mix: Literal["mul", "add"] = "mul",
    # normalizzazione/clamp
    normalize_to: Optional[float] = 2.0,   # None => niente normalizzazione finale
    # colore
    plot_type: Literal["triangle", "triangles", "dot", "dots"] = "triangle",
    color_mode: Literal["legacy", "abs-hsv", "diverging", "gray", "log-hsv", "banded"] = "diverging",
    max_val: Optional[float] = None,
    auto_clip_pct: float = 99.0,
    gamma: float = 0.7,
    sat_dist_weight: float = 0.5,
    neg_policy: NegPolicy = "hue-180",
    band_count: int = 8,
    # output
    output_dirname: str = "rippless",
) -> Tuple[Path, List[Path]]:
    """
    Applica una modulazione sinusoidale radiale ai coefficienti:
      f(d,t) = bias + amplitude * sin( 2π*( d/λ_norm - t*ν ) + phase0 ) * exp(-alpha*d)
    con ν = speed_pct_per_frame / wavelength_pct.

    Se mix == "mul" usa coeff' = coeff * f(d,t), altrimenti "add" fa coeff' = coeff + f(d,t).

    I frame sono salvati in: output/order_{order}/{output_dirname}/frame_XXXX.png
    """
    total_frames = max(int(total_frames), 1)
    λ_norm = max(float(wavelength_pct) / 100.0, 1e-6)
    v_norm = float(speed_pct_per_frame) / 100.0
    nu = v_norm / λ_norm  # cicli per frame

    # seed
    if seed is None:
        zero = Floretion.from_string(f'0{"e"*order}')
        seed = Floretion(np.ones_like(zero.coeff_vec_all, dtype=float),
                         zero.base_vec_dec_all, zero.grid_flo_loaded_data)

    # distanze normalizzate (stesso ordine della griglia del seed)
    dist_vec = _load_or_build_dist_vec(order)  # shape (4^n,)

    out_dir = Path("output") / f"order_{order}" / output_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps({
        "order": order,
        "total_frames": total_frames,
        "wavelength_pct": wavelength_pct,
        "speed_pct_per_frame": speed_pct_per_frame,
        "amplitude": amplitude,
        "bias": bias,
        "phase0": phase0,
        "damping_alpha": damping_alpha,
        "mix": mix,
        "normalize_to": normalize_to,
        "plot_type": plot_type,
        "color_mode": color_mode,
        "max_val": max_val,
        "auto_clip_pct": auto_clip_pct,
        "gamma": gamma,
        "sat_dist_weight": sat_dist_weight,
        "neg_policy": neg_policy,
        "band_count": band_count,
    }, indent=2), encoding="utf-8")

    # import qui per evitare import ciclici
    from triangleize import Triangleize

    frame_paths: List[Path] = []
    X = seed
    for t in range(total_frames):

        phase = 2.0 * math.pi * (-nu * t) + float(phase0)  # -(nu t) -> propagazione verso l’esterno
        arg = 2.0 * math.pi * (dist_vec / λ_norm) + phase
        env = np.exp(-float(damping_alpha) * dist_vec) if damping_alpha > 0 else 1.0
        mod = float(bias) + float(amplitude) * np.sin(arg) * env  # shape (4^n,)

        if mix == "mul":
            coeffs_new = seed.coeff_vec_all.astype(float) * mod
            coeffs_old = seed.coeff_vec_all.astype(float) * np.sin(phase)
        else:
            coeffs_new = seed.coeff_vec_all.astype(float) + mod

        X = Floretion(coeffs_new, seed.base_vec_dec_all, seed.grid_flo_loaded_data)
        #Y =  Floretion(coeffs_old, seed.base_vec_dec_all, seed.grid_flo_loaded_data)

        #axis_ijk = Floretion.get_typical_floretions("sierp_flo", order)
        #X = X*X
        seed = Floretion.rotate_coeffs(seed)
        seed = Floretion.tri(seed)
        X = X*X
        if normalize_to is not None and normalize_to > 0:
            X = Floretion.normalize_coeffs(X, float(normalize_to))

        img = np.zeros((height, width, 3), dtype=np.uint8)
        tri = Triangleize(X, img, plot_type=plot_type)

        if color_mode == "legacy":
            # identico aspetto “storico”
            tri.plot_floretion(color_mode="abs-hsv", max_val=None,
                               auto_clip_pct=100.0, gamma=1.0,
                               sat_dist_weight=1.0, neg_policy="none",
                               band_count=band_count)
        else:
            tri.plot_floretion(color_mode=color_mode, max_val=max_val,
                               auto_clip_pct=auto_clip_pct, gamma=gamma,
                               sat_dist_weight=sat_dist_weight,
                               neg_policy=neg_policy, band_count=band_count)

        fpath = out_dir / f"frame_{t:04d}.png"
        print(f"Processed {fpath} ")

        if not cv2.imwrite(str(fpath), img):
            raise RuntimeError(f"cv2.imwrite fallita: {fpath}")
        frame_paths.append(fpath)


    return out_dir, frame_paths


