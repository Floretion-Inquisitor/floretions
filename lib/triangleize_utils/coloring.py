# lib/triangleize_utils/coloring.py
from __future__ import annotations

from typing import Tuple, Literal, Optional
import math
import numpy as np
import cv2

# Modalità esistenti + nuove (tutte backward-compatible)
ColorMode = Literal[
    "legacy",
    "abs-hsv",
    "diverging",
    "gray",
    "log-hsv",
    "banded",
    # nuove:
    "pastel",
    "pastel-diverging",
    "coolwarm",
    "heat",
    "neon",
    "sat-only",
    "distance-hsv",
    "banded-pastel",
    "ink",
]
NegPolicy = Literal["hue-180", "hue-90", "none"]

__all__ = [
    "ColorMode",
    "NegPolicy",
    "choose_max_val_for_colors",
    "map_color",
]

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _clamp255(x: int) -> int:
    if x < 0:
        return 0
    if x > 255:
        return 255
    return x

def _hsv_to_bgr(h: int, s: int, v: int) -> Tuple[int, int, int]:
    hsv_col = np.array([[[int(h) % 180, _clamp255(int(s)), _clamp255(int(v))]]], dtype=np.uint8)
    bgr_px = cv2.cvtColor(hsv_col, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr_px[0]), int(bgr_px[1]), int(bgr_px[2])

def _apply_gamma01(x01: float, gamma: float) -> float:
    x01 = _clamp01(x01)
    if gamma <= 0:
        return x01
    return x01 ** gamma

def _hue_shift(h: int, deg: float) -> int:
    # OpenCV HSV: hue in [0..179] ≈ [0..360)
    return int((h + int(round(deg / 2.0))) % 180)

def choose_max_val_for_colors(
    coeffs: np.ndarray,
    max_val: Optional[float],
    auto_clip_pct: float
) -> float:
    """
    Se max_val è None: usa il percentile (es. 99°) per evitare saturazione da outlier.
    """
    if max_val is not None and max_val > 0:
        return float(max_val)
    mags = np.abs(coeffs.astype(float))
    clip = float(np.percentile(mags, auto_clip_pct)) if mags.size else 1.0
    return max(clip, 1e-12)

def map_color(
    *,
    coeff: float,
    basevec_at_pct: float,   # ∈ [0,1]
    dist_norm: float,        # ∈ [0,1] (distanza/altezza immagine)
    max_val: float,
    mode: ColorMode = "abs-hsv",
    gamma: float = 0.6,
    sat_dist_weight: float = 0.5,
    neg_policy: NegPolicy = "hue-180",
    band_count: int = 8,
) -> Tuple[Tuple[int, int, int], float]:
    """
    Ritorna (BGR, brightness01). brightness01 ∈ [0..1] (usabile per dimensione/riempimento).

    Modalità:
      - legacy        : identica alla vecchia.
      - abs-hsv       : hue da posizione, sat/val da coeff+dist.
      - log-hsv       : come abs-hsv ma con compressione log.
      - banded        : bande discrete di hue.
      - diverging     : rosso (+) / blu (-).
      - gray          : scala di grigi.
      - pastel        : colori “pastello” (S bassa, V alta), hue da posizione.
      - pastel-diverging: diverging ma pastello (bianco vicino a zero, più sat con |coeff|).
      - coolwarm      : diverging “soft”: vicino a 0 tende al bianco; saturazione cresce con |coeff|.
      - heat          : colormap tipo “heat”: blu->rosso su |coeff| (opzionalmente segnabile via neg_policy).
      - neon          : molto saturo e brillante.
      - sat-only      : V quasi costante, la magnitudine modula soprattutto la saturazione.
      - distance-hsv  : S controllata soprattutto dalla distanza; V dalla magnitudine.
      - banded-pastel : come banded ma in stile pastello.
      - ink           : “inchiostro”: fondo scuro, segni colorati (V più basso).
    """

    # ---------
    # LEGACY: replica esatta del vecchio schema
    # ---------
    if mode == "legacy":
        hue = int(179 * _clamp01(float(basevec_at_pct)))
        if coeff < 0:
            hue = int(hue / 2)

        sat = int(2.0 * 255.0 * _clamp01(float(dist_norm)))
        sat = _clamp255(sat)

        val = int(abs(float(coeff)) * 255.0)
        if val > 255:
            val = 255

        color = _hsv_to_bgr(hue, sat, val)
        brightness01 = 0.25 + 0.75 * (val / 255.0)
        return color, brightness01

    # ---------
    # MODERNE
    # ---------
    mag = abs(float(coeff))
    mag01_raw = 0.0 if max_val <= 0 else (mag / float(max_val))
    mag01 = _apply_gamma01(mag01_raw, gamma)

    tpos = _clamp01(float(basevec_at_pct))
    sat_dist = _clamp01(float(dist_norm))

    # hue “arcobaleno” lungo la griglia
    base_hue = int(179 * tpos)

    # sat/val base (riutilizzate da alcune modalità)
    sat_from_dist = int(255 * (sat_dist ** 1.2) * float(sat_dist_weight))
    s_base = int(200 * mag01)
    sat = _clamp255(s_base + sat_from_dist)
    val = _clamp255(int(255 * mag01))

    # Applica neg_policy SOLO alle modalità che usano hue “spaziale”
    # (quelle divergenti già codificano il segno)
    if coeff < 0 and mode in ("abs-hsv", "log-hsv", "banded", "pastel", "neon", "sat-only", "distance-hsv", "banded-pastel", "ink", "heat"):
        if neg_policy == "hue-180":
            base_hue = _hue_shift(base_hue, 180.0)
        elif neg_policy == "hue-90":
            base_hue = _hue_shift(base_hue, 90.0)

    # ---------
    # MODALITÀ SPECIFICHE
    # ---------
    if mode == "diverging":
        hue = 0 if coeff >= 0 else 120  # 0≈rosso, 120≈blu
        sat = _clamp255(int(255 * mag01))
        val = 230

    elif mode == "gray":
        hue = 0
        sat = 0
        val = _clamp255(int(255 * mag01))

    elif mode == "log-hsv":
        k = 4.0
        num = math.log1p(k * mag)
        den = math.log1p(k * float(max_val) if float(max_val) > 0 else k)
        v01 = 0.0 if den <= 0 else (num / den)
        v01 = _apply_gamma01(v01, gamma)
        hue = base_hue
        val = _clamp255(int(255 * v01))
        sat = _clamp255(int(180 * v01) + sat_from_dist)

    elif mode == "banded":
        band_count = max(int(band_count), 2)
        idx = int(mag01 * band_count)
        idx = min(idx, band_count - 1)
        hue = int((179 * idx) / max(band_count - 1, 1))
        val = 230
        sat = _clamp255(210 + int(45 * float(sat_dist_weight)))

    elif mode == "pastel":
        # Pastello: V alta, S bassa/moderata. Hue da posizione.
        hue = base_hue
        # saturazione: un filo con |coeff| + un filo con distanza, ma resta "soft"
        s = 35 + int(85 * mag01) + int(60 * sat_dist * float(sat_dist_weight))
        sat = _clamp255(min(s, 150))
        # value: sempre alto
        v = 210 + int(45 * mag01)
        val = _clamp255(v)

    elif mode == "banded-pastel":
        band_count = max(int(band_count), 2)
        idx = int(mag01 * band_count)
        idx = min(idx, band_count - 1)
        hue = int((179 * idx) / max(band_count - 1, 1))
        sat = _clamp255(55 + int(70 * mag01))
        sat = min(sat, 150)
        val = _clamp255(225 + int(25 * mag01))

    elif mode == "pastel-diverging":
        # diverging pastello: vicino a 0 -> quasi bianco (sat bassa), con |coeff| aumenta la sat
        # hue fissati ma "ammorbiditi"
        hue = 10 if coeff >= 0 else 115  # rosato / azzurrino
        # sat cresce con |coeff|
        sat = _clamp255(int(30 + 140 * mag01))
        sat = min(sat, 160)
        # value alto; leggero boost lontano dal centro se vuoi più “aria”
        val = _clamp255(int(235 + 15 * (mag01 - 0.5)))

    elif mode == "coolwarm":
        # Divergente soft: bianco a 0, colore più saturo con |coeff|
        hue = 0 if coeff >= 0 else 120
        # sat cresce con |coeff|, ma parte da molto basso
        sat = _clamp255(int(10 + 245 * mag01))
        # value sempre alto; leggermente più basso per magnitudini enormi (effetto “inchiostro”)
        val = _clamp255(int(255 - 35 * (mag01 ** 1.2)))

    elif mode == "heat":
        # Colormap su |coeff|: mag01=0 -> blu (120), mag01=1 -> rosso (0)
        hue = int(120 * (1.0 - mag01))
        sat = 255
        val = _clamp255(int(60 + 195 * mag01))

    elif mode == "neon":
        # Neon: sat alta, val alta (ma con dinamica sulla magnitudine)
        hue = base_hue
        sat = _clamp255(200 + int(55 * (mag01 ** 0.5)))
        val = _clamp255(80 + int(175 * (mag01 ** 0.7)))

    elif mode == "sat-only":
        # V quasi costante e alto; la differenza la fa soprattutto la saturazione
        hue = base_hue
        val = 235
        sat = _clamp255(int(10 + 245 * mag01) + int(40 * sat_dist * float(sat_dist_weight)))

    elif mode == "distance-hsv":
        # distanza -> saturazione, magnitudine -> value
        hue = base_hue
        sat = _clamp255(int(30 + 225 * (sat_dist ** 0.9)))
        val = _clamp255(int(30 + 225 * mag01))

    elif mode == "ink":
        # “inchiostro”: più scuro, sat moderata-alta; magnitudine controlla soprattutto V
        hue = base_hue
        sat = _clamp255(int(60 + 170 * (mag01 ** 0.7)) + int(40 * sat_dist * float(sat_dist_weight)))
        val = _clamp255(int(15 + 165 * (mag01 ** 0.8)))

    else:
        # default: abs-hsv (fallback sicuro)
        hue = base_hue

    color = _hsv_to_bgr(int(hue), int(sat), int(val))
    brightness01 = 0.25 + 0.75 * (float(val) / 255.0)
    return color, brightness01
