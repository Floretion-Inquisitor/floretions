from __future__ import annotations
from typing import Dict

_mask_cache: Dict[int, int] = {}
_oct666_cache: Dict[int, int] = {}
_oct111_cache: Dict[int, int] = {}

def _bitmask(order: int) -> int:
    m = _mask_cache.get(order)
    if m is None:
        m = (1 << (3 * order)) - 1
        _mask_cache[order] = m
    return m

def _oct666(order: int) -> int:
    v = _oct666_cache.get(order)
    if v is None:
        v = int("6" * order, 8)
        _oct666_cache[order] = v
    return v

def _oct111(order: int) -> int:
    v = _oct111_cache.get(order)
    if v is None:
        v = int("1" * order, 8)
        _oct111_cache[order] = v
    return v

def sgn(x: float) -> int:
    return -1 if x < 0 else 1
