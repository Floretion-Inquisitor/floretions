from __future__ import annotations

from typing import Callable, Sequence, Optional

import numpy as np
from floretion import Floretion
from lib.triangleize_utils.triangle_math import clip_coeffs


def _annotate(fn: Callable, factory_name: str, params: dict | None = None, series: np.ndarray | None = None, series_name: str | None = None):
    """
    Attacca metadati serializzabili agli step (callable) per consentire ricostruzione in meta.json.
    - _meta: info su factory e parametri
    - _series: eventuale sequenza numerica (es. factors)
    - _series_name: nome logico della sequenza
    """
    fn._meta = {"factory": factory_name, "params": params or {}}
    if series is not None:
        fn._series = np.asarray(series, dtype=float)
        fn._series_name = series_name or "series"
    return fn


def step_square(x: Floretion, iter_index: Optional[int] = None) -> Floretion:
    """x ↦ x*x"""
    return x * x

# Annotazione statica (utile per meta.json)
step_square._meta = {"factory": "builtin", "name": "step_square", "params": {}}


def step_normalize_const(value: float) -> Callable[[Floretion], Floretion]:
    """
    Factory: crea uno step che normalizza sempre a 'value'.
    """
    v = float(value)

    def _step(x: Floretion, iter_index: Optional[int] = None) -> Floretion:
        return Floretion.normalize_coeffs(x, v)

    return _annotate(_step, "step_normalize_const", params={"value": v})


def make_clip_by_factor(factors: Sequence[float]) -> Callable[[Floretion, int], Floretion]:
    """
    Factory: step che applica clip_coeffs con soglia factors[iter_index] sullo stato corrente.
    """
    factors_arr = np.asarray(factors, dtype=float)

    def _step(x: Floretion, iter_index: Optional[int] = None) -> Floretion:
        idx = 0 if iter_index is None else int(iter_index)
        f = float(factors_arr[idx])
        return clip_coeffs(x, f)

    return _annotate(_step, "make_clip_by_factor",
                     params={"len": int(factors_arr.size), "min": float(factors_arr.min(initial=0.0)), "max": float(factors_arr.max(initial=0.0))},
                     series=factors_arr,
                     series_name="factors")


def make_normalize_by_factor(factors: Sequence[float]) -> Callable[[Floretion, int], Floretion]:
    """
    Factory: step che normalizza a factors[iter_index].
    """
    factors_arr = np.asarray(factors, dtype=float)

    def _step(x: Floretion, iter_index: Optional[int] = None) -> Floretion:
        idx = 0 if iter_index is None else int(iter_index)
        f = float(factors_arr[idx])
        return Floretion.normalize_coeffs(x, f)

    return _annotate(_step, "make_normalize_by_factor",
                     params={"len": int(factors_arr.size), "min": float(factors_arr.min(initial=0.0)), "max": float(factors_arr.max(initial=0.0))},
                     series=factors_arr,
                     series_name="factors")
