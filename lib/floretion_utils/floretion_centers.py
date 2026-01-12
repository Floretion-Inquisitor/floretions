# lib/floretion_utils/floretion_centers.py
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Dict, Iterable, Set, Tuple, Any
from typing import TYPE_CHECKING
import numpy as np
import json
import re

# NOTE: to avoid circular imports, we only import Floretion inside functions
if TYPE_CHECKING:
    from floretion import Floretion  # type: ignore

# --------------------------------------------
# Public API (kept stable for floretion.py)
# --------------------------------------------

Mode = Literal["pos", "neg", "both", "P", "N"]

__all__ = [
    "Mode",
    "centers_dir",
    "load_centers",               # small orders (single file in folder)
    "load_centers_for_base",      # segmented orders 7/8 (or fallback)
    "load_centers_single",        # compat: single base, supports P/N + segments
    "load_centers_map",           # small orders -> dict[str,set[int]], segmented -> None
    "write_centers_to_file",
    "_oct_from_dec",
    "find_center",                # NEW: compute Floretion of centers for a single base vector
    "find_center_base_vectors_only",
]

# --------------------------------------------
# Paths & naming
# --------------------------------------------

_BASE_DIR = Path("./data/centers")

_PAR_MAP = {
    "pos": "pos",
    "positive": "pos",
    "cp": "pos",
    "p": "pos",

    "neg": "neg",
    "negative": "neg",
    "cn": "neg",
    "n": "neg",

    "both": "both",
    "cb": "both",

    # aliases used by UI
    "P": "pos",
    "N": "neg",
}

# Supporta:
# A) centers_order_{n}_segment_{seg:03d}.{START_OCT}-{END_OCT}.{npy|json}
# B) centers_order_{n}_segment_{seg:03d}_{START_OCT}_{END_OCT}.{npy|json}
# C) centers_order_{n}_segment_{seg:03d}.{npy|json}          (single-file, no range)
# D) centers_order_{n}_segment_{seg:03d}.{anything}.{npy}    (se vuoi essere super-tollerante puoi estendere)
_SEGMENT_RE = re.compile(
    r"^centers_order_(\d+)_segment_(\d{3})"
    r"(?:(?:[._])([1247]+)(?:-|_)([1247]+))?"
    r"\.(npy|json)$"
)

def _norm_parity(parity: str) -> str:
    p = str(parity).strip().lower()
    return _PAR_MAP.get(p, p)

def centers_dir(order: int, parity: str) -> Path:
    par = _norm_parity(parity)
    return _BASE_DIR / f"order_{int(order)}" / par

# --------------------------------------------
# Helpers: IO
# --------------------------------------------

def _is_segmented_filename(p: Path) -> bool:
    return _SEGMENT_RE.match(p.name) is not None

def _list_center_files(dirp: Path) -> list[Path]:
    if not dirp.exists():
        return []

    files = []
    for p in dirp.iterdir():
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf not in (".npy", ".json"):
            continue
        # ignora manifest e roba non-centers
        if p.name.lower() == "manifest.json":
            continue
        if not p.name.startswith("centers_order_"):
            continue
        files.append(p)

    return sorted(files, key=lambda x: x.name)


def _load_dict_from_file(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".npy":
        data = np.load(str(path), allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        if not isinstance(data, dict):
            raise ValueError(f"[centers] npy does not contain a dict: {path}")
        return data
    elif path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"[centers] unsupported file extension: {path}")

# --------------------------------------------
# Oct/dec helpers
# --------------------------------------------

def _oct_from_dec(base_dec: int) -> str:
    return format(int(base_dec), "o")

# --------------------------------------------
# Loaders
# --------------------------------------------

def load_centers(order: int, parity: str = "both", storage_type: str = "npy") -> Dict[str, Any]:
    """
    Load the whole centers dict for *small orders* where a single file exists under:
      data/centers/order_{n}/{parity}/centers_order_{n}_{parity}.{npy|json}

    For segmented orders (7/8), we intentionally refuse to load all segments at once.
    """
    par = _norm_parity(parity)
    dirp = centers_dir(order, par)
    files = _list_center_files(dirp)
    if not files:
        raise FileNotFoundError(f"[centers] empty/non-existent dir: {dirp}")

    if any(_is_segmented_filename(p) for p in files):
        raise RuntimeError(
            "[centers] load_centers is for small orders (single file). "
            "For 7/8 use load_centers_for_base(...) (segment-aware)."
        )

    # choose 1 file (prefer npy if both exist)
    npy = [p for p in files if p.suffix.lower() == ".npy"]
    candidates = npy if npy else files
    if len(candidates) != 1:
        raise RuntimeError(
            f"[centers] expected exactly 1 file in {dirp}, found {len(candidates)}: {[c.name for c in candidates]}"
        )

    path = candidates[0]
    return _load_dict_from_file(path)

# --------------------------------------------
# Segment-aware: single base -> centers list
# --------------------------------------------

def _find_segment_file_for_oct(dirp: Path, base_oct: str) -> Optional[Path]:
    """
    Se i segmenti hanno START_OCT/END_OCT, seleziona quello corretto.
    Se non hanno range (es: centers_order_6_segment_000.npy), usa quel file
    (a patto che sia unico).
    """
    files = _list_center_files(dirp)

    seg_files = []
    no_range = []
    ranged = []

    for p in files:
        m = _SEGMENT_RE.match(p.name)
        if not m:
            continue
        seg_files.append(p)
        _ord_s, _seg_id, start_oct, end_oct, _ext = m.groups()
        if start_oct is None or end_oct is None:
            no_range.append(p)
        else:
            ranged.append((p, start_oct, end_oct))

    # Caso 1: abbiamo segmenti con range -> scegli per confronto lessicografico
    for p, start_oct, end_oct in ranged:
        if start_oct <= base_oct <= end_oct:
            return p

    # Caso 2: non c'è match per range ma c'è esattamente un "no-range"
    if len(no_range) == 1 and len(ranged) == 0:
        return no_range[0]

    # Caso 3: come fallback, se c'è esattamente un file segment riconosciuto, usa quello
    if len(seg_files) == 1:
        return seg_files[0]

    return None


def load_centers_for_base(order: int, parity: str, base_dec: int) -> np.ndarray:
    """
    Load the centers for a *single* base vector (decimal code).
    Works for:
      - segmented orders 7/8: picks the correct segment by START_OCT..END_OCT,
      - small orders: falls back to the single file.

    Returns a sorted np.ndarray[int].
    """
    dirp = centers_dir(order, parity)
    files = _list_center_files(dirp)
    if not files:
        raise FileNotFoundError(f"[centers] empty/non-existent dir: {dirp}")

    base_oct = _oct_from_dec(int(base_dec))
    segmented = any(_is_segmented_filename(p) for p in files)

    if segmented:
        segfile = _find_segment_file_for_oct(dirp, base_oct)
        if segfile is None:
            raise FileNotFoundError(
                f"[centers] no segment contains {base_oct} in {dirp}"
            )
        data = _load_dict_from_file(segfile)
        arr = data.get(str(int(base_dec))) or data.get(base_oct)
        if arr is None:
            raise KeyError(f"[centers] key not found in segment {segfile.name}: {base_dec} (oct: {base_oct})")
        return np.asarray(sorted(map(int, arr)), dtype=int)

    # single-file (small orders)
    npy = [p for p in files if p.suffix.lower() == ".npy"]
    candidates = npy if npy else files
    if len(candidates) != 1:
        raise RuntimeError(
            f"[centers] expected exactly 1 file in {dirp}, found {len(candidates)}: {[c.name for c in candidates]}"
        )
    data = _load_dict_from_file(candidates[0])
    arr = data.get(str(int(base_dec))) or data.get(base_oct)
    if arr is None:
        raise KeyError(f"[centers] key not found in {candidates[0].name}: {base_dec} (oct: {base_oct})")
    return np.asarray(sorted(map(int, arr)), dtype=int)

# --------------------------------------------
# Compat: single base -> centers list (supports P/N semantics)
# --------------------------------------------

def load_centers_single(order: int, mode: "Mode", base_dec: int, storage_type: str = "npy") -> np.ndarray:
    """
    Back-compat helper expected by lib.floretion_utils.helpers.parse_special_commands.

    - For mode in {"pos","neg","both"} (and aliases Cp/Cn/Cb): loads precomputed centers from disk.
      Works with both single-file orders and segmented (7/8) orders.
    - For mode in {"P","N"}: computes the set { b | sign(a*b) > 0 } or { b | sign(a*b) < 0 } at runtime
      using Floretion.mult_flo_sign_only. This avoids needing separate precomputed files.

    Returns a sorted np.ndarray[int] of base-vectors (decimal codes).
    """
    m = str(mode).strip()

    # P / N are *sign-based* selection (not commutation-based).
    if m.upper() == "P":
        from floretion import Floretion as _F  # lazy import (avoid cycles)
        all_bases = _F.from_string("1" + "e" * int(order)).base_vec_dec_all.astype(int)
        out = [int(b) for b in all_bases if _F.mult_flo_sign_only(int(base_dec), int(b), int(order)) > 0]
        return np.asarray(out, dtype=int)

    if m.upper() == "N":
        from floretion import Floretion as _F  # lazy import (avoid cycles)
        all_bases = _F.from_string("1" + "e" * int(order)).base_vec_dec_all.astype(int)
        out = [int(b) for b in all_bases if _F.mult_flo_sign_only(int(base_dec), int(b), int(order)) < 0]
        return np.asarray(out, dtype=int)

    # Cp/Cn/Cb + pos/neg/both → file-backed centers
    parity = _norm_parity(m)
    return load_centers_for_base(int(order), parity, int(base_dec))

# --------------------------------------------
# Map for square_fast (small orders only)
# --------------------------------------------

def load_centers_map(order: int, mode: str = "both", storage_type: str = "npy") -> Optional[Dict[str, Set[int]]]:
    """
    For small orders: load the single centers file and convert to dict[str, set[int]].
    For segmented orders (7/8): return None (avoid loading everything).
    """
    dirp = centers_dir(order, mode)
    files = _list_center_files(dirp)
    if not files:
        return None

    if any(_is_segmented_filename(p) for p in files):
        return None

    try:
        data = load_centers(order, mode, storage_type)
    except Exception:
        return None

    out: Dict[str, Set[int]] = {}
    for k, v in data.items():
        out[str(k)] = set(map(int, v))
    return out

# --------------------------------------------
# Writers
# --------------------------------------------

def write_centers_to_file(order: int,
                          parity: str,
                          data: Dict[str, Iterable[int]],
                          *,
                          path: Optional[Path] = None,
                          fmt: Literal["npy", "json"] = "npy") -> Path:
    """
    Write a dict { base_dec(str) : [centers_dec.] } to disk.
    If path is None, writes to: data/centers/order_{n}/{parity}/centers_order_{n}_{parity}.{fmt}
    """
    par = _norm_parity(parity)
    dirp = centers_dir(order, par)
    dirp.mkdir(parents=True, exist_ok=True)

    if path is None:
        fname = f"centers_order_{order}_{par}.{fmt}"


# --- Centers (runtime) for a single base vector -----------------------------

def _single_nonzero_base_info(X: "Floretion") -> tuple[int, float]:
    """
    Return (base_dec, coeff) for a Floretion with exactly one non-zero coefficient.
    Raises if support != 1.
    """
    import numpy as np

    nz = np.nonzero(X.coeff_vec_all)[0]
    if nz.size != 1:
        raise ValueError(f"[centers] expected a single base-vector support, got {nz.size}")
    idx = int(nz[0])
    base_dec = int(X.base_vec_dec_all[idx])
    coeff = float(X.coeff_vec_all[idx])
    return base_dec, coeff


def find_center_base_vectors_only(floretion: "Floretion", decomposition_type: str = "both") -> set[int]:
    """
    Return the *set of center base-vectors (decimals)* for a Floretion that represents
    exactly one base vector (support size == 1). Uses precomputed centers on disk.
    """
    from floretion import Floretion as _F  # lazy import (avoid cycles)

    if not isinstance(floretion, _F):
        raise TypeError("[centers] find_center_base_vectors_only expects a Floretion")

    order = int(floretion.flo_order)
    parity = _norm_parity(decomposition_type)

    base_dec, _ = _single_nonzero_base_info(floretion)
    arr = load_centers_for_base(order, parity, int(base_dec))
    return set(map(int, arr.tolist()))


def find_center(floretion: "Floretion", decomposition_type: str = "both") -> "Floretion":
    """
    Return a Floretion whose support is exactly the set of centers of the single base-vector
    contained in the input Floretion (support size must be 1). Coefficients are all 1.0.
    """
    import numpy as np
    from floretion import Floretion as _F  # lazy import (avoid cycles)

    centers_set = find_center_base_vectors_only(floretion, decomposition_type)
    if not centers_set:
        return _F.from_string(f'0{"e"*int(floretion.flo_order)}')

    coeffs = np.ones(len(centers_set), dtype=float)
    bases = np.array(sorted(centers_set), dtype=int)
    return _F(coeffs, bases)
