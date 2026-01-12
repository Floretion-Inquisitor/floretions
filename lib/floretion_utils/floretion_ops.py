# lib/floretion_utils/floretion_ops.py
from __future__ import annotations

from typing import Optional, Literal, Iterable, Dict, Set, Any
from pathlib import Path
import numpy as np
import json

# Espone le funzioni usate da floretion.py
__all__ = [
    # utils base
    "normalize_coeffs",
    "fractional_coeffs",
    "mirror",
    "split_by_leading_symbol",
    "cyc",
    "tri",
    "strip_flo",
    "grow_flo",
    "proj",
    "proj_strip_grow",
    "rotate_coeffs",
    "apply_sequence",
    # centers / compat
    "find_center_base_vectors_only",
    "find_center",
    "load_centers",
    "write_centers_to_file",
    "get_typical_floretions",
    # mul
    "square_fast",
    "roll"
]

# =============================
#  Operazioni su base vectors
# =============================

def normalize_coeffs(X: "Floretion", max_abs: float = 1.0) -> "Floretion":
    from floretion import Floretion
    v = X.coeff_vec_all.astype(float)
    m = float(np.max(np.abs(v))) if v.size else 1.0
    if m <= 0:
        return Floretion(np.zeros_like(v, dtype=float), X.base_vec_dec_all, X.grid_flo_loaded_data)
    return Floretion((v / m) * float(max_abs), X.base_vec_dec_all, X.grid_flo_loaded_data)

def fractional_coeffs(X: "Floretion", p: int) -> "Floretion":
    from floretion import Floretion
    return Floretion(X.coeff_vec_all / float(p), X.base_vec_dec_all, X.grid_flo_loaded_data)



def roll(floretion: "Floretion", shift: int = 1, block: int | None = None) -> "Floretion":
    """
    Cicla i coefficienti di una Floretion.

    - Se block è None: np.roll su tutto l'array di coeff.
    - Se block divide esattamente len(coeffs): roll per blocchi indipendenti.
    """
    from floretion import Floretion
    coeffs = floretion.coeff_vec_all
    n = coeffs.size

    if block is None:
        newc = np.roll(coeffs, int(shift))
        return Floretion(newc, floretion.base_vec_dec_all, floretion.grid_flo_loaded_data)

    block = int(block)
    if block <= 0 or (n % block) != 0:
        raise ValueError(f"block={block} non valido: deve dividere la lunghezza {n}.")

    newc = np.empty_like(coeffs, dtype=float)
    num_blocks = n // block
    for b in range(num_blocks):
        s = b * block
        e = s + block
        newc[s:e] = np.roll(coeffs[s:e], int(shift))
    return Floretion(newc, floretion.base_vec_dec_all, floretion.grid_flo_loaded_data)


def roll_quarters(floretion: "Floretion", shift_quarters: int = 1) -> "Floretion":
    """
    Rotazione circolare dei 4 quarti (i/j/k/e) assumendo base ordinata in 4 blocchi uguali.
    """
    from floretion import Floretion
    coeffs = floretion.coeff_vec_all
    n = coeffs.size
    if (n % 4) != 0:
        raise ValueError("La base non è multipla di 4; impossibile considerare quarti i/j/k/e.")

    q = n // 4
    blocks = [coeffs[0:q], coeffs[q:2*q], coeffs[2*q:3*q], coeffs[3*q:4*q]]
    k = int(shift_quarters) % 4
    if k == 0:
        return floretion
    blocks_rot = blocks[-k:] + blocks[:-k]
    newc = np.concatenate(blocks_rot, axis=0)
    return Floretion(newc, floretion.base_vec_dec_all, floretion.grid_flo_loaded_data)


def mirror(floretion: "Floretion", axis: str) -> "Floretion":
    """
    Riflette i simboli (i/j/k) rispetto all'asse richiesto:
      axis="I": scambia j<->k
      axis="J": scambia i<->k
      axis="K": scambia i<->j
    """
    from floretion import Floretion
    axis = axis.upper()
    if axis not in ("I", "J", "K"):
        raise ValueError("Axis deve essere 'I', 'J' o 'K'.")

    new_coeffs = np.zeros_like(floretion.coeff_vec_all, dtype=float)
    base_all = floretion.base_vec_dec_all
    idx_by_dec = {int(v): int(i) for i, v in enumerate(base_all)}
    order = int(floretion.flo_order)

    def map_digit(d: int) -> int:
        if d == 7:
            return 7
        if axis == "I":
            return 4 if d == 2 else 2 if d == 4 else 1
        if axis == "J":
            return 4 if d == 1 else 1 if d == 4 else 2
        # axis == "K"
        return 2 if d == 1 else 1 if d == 2 else 4

    for base_dec, coeff in floretion.base_to_nonzero_coeff.items():
        a = int(base_dec)
        mapped = 0
        pos = 0
        for _ in range(order):
            d = (a >> pos) & 7
            mapped |= (map_digit(d) << pos)
            pos += 3
        idx = idx_by_dec.get(mapped)
        if idx is not None:
            new_coeffs[idx] += float(coeff)

    return Floretion(new_coeffs, base_all, floretion.grid_flo_loaded_data)


def split_by_leading_symbol(floretion: "Floretion") -> tuple["Floretion", "Floretion", "Floretion", "Floretion"]:
    """
    Suddivide la floretion in 4 parti (i/j/k/e) assumendo base ordinata in 4 quarti.
    """
    from floretion import Floretion
    n = int(floretion.base_vec_dec_all.size)
    if n % 4 != 0:
        raise ValueError("La lunghezza della base non è multipla di 4: impossibile suddividere in quarti.")
    q = n // 4

    def make_part(start: int, end: int) -> "Floretion":
        part_coeffs = np.zeros_like(floretion.coeff_vec_all, dtype=float)
        part_coeffs[start:end] = floretion.coeff_vec_all[start:end]
        return Floretion(part_coeffs, floretion.base_vec_dec_all, floretion.grid_flo_loaded_data)

    X_i = make_part(0, q)
    X_j = make_part(q, 2 * q)
    X_k = make_part(2 * q, 3 * q)
    X_e = make_part(3 * q, n)
    return (X_i, X_j, X_k, X_e)


def cyc(floretion: "Floretion") -> "Floretion":
    """
    Ciclo dei simboli: i->j, j->k, k->i, e->e (digit mapping 1->2->4->1, 7->7).
    """
    from floretion import Floretion
    order = int(floretion.flo_order)
    base_all = floretion.base_vec_dec_all
    idx_by_dec = {int(v): int(i) for i, v in enumerate(base_all)}
    new_coeffs = np.zeros_like(floretion.coeff_vec_all, dtype=float)

    def map_digit(d: int) -> int:
        if d == 1:
            return 2
        if d == 2:
            return 4
        if d == 4:
            return 1
        if d == 7:
            return 7
        raise ValueError(f"Cifra ottale non valida: {d}")

    for base_dec, coeff in floretion.base_to_nonzero_coeff.items():
        a = int(base_dec)
        mapped = 0
        pos = 0
        for _ in range(order):
            d = (a >> pos) & 7
            mapped |= (map_digit(d) << pos)
            pos += 3
        idx = idx_by_dec.get(mapped)
        if idx is not None:
            new_coeffs[idx] += float(coeff)

    return Floretion(new_coeffs, base_all, floretion.grid_flo_loaded_data)


def tri(floretion: "Floretion") -> "Floretion":
    """
    Proiezione triangolare: (X + cyc(X) + cyc(cyc(X))) / 3
    """
    X = floretion
    X1 = cyc(X)
    X2 = cyc(X1)
    return (1.0 / 3.0) * (X + X1 + X2)


def strip_flo(floretion: "Floretion") -> "Floretion":
    """
    Rimuove il digit ottale più significativo (ordine n -> n-1).
    """
    from floretion import Floretion
    order = int(floretion.flo_order)
    if order <= 1:
        raise ValueError("strip_flo richiede una Floretion di ordine >= 2.")
    new_order = order - 1

    zero_new = 0 * Floretion.from_string('1' + 'e' * new_order)
    idx_by_dec_new = {int(v): int(i) for i, v in enumerate(zero_new.base_vec_dec_all)}
    new_coeffs = np.zeros_like(zero_new.coeff_vec_all, dtype=float)

    for base_dec, coeff in floretion.base_to_nonzero_coeff.items():
        oct_str = format(int(base_dec), "o").rjust(order, "0")
        tail_str = oct_str[1:]
        new_base_dec = int(tail_str, 8)
        idx = idx_by_dec_new.get(new_base_dec)
        if idx is not None:
            new_coeffs[idx] += float(coeff)

    return Floretion(new_coeffs, zero_new.base_vec_dec_all, zero_new.grid_flo_loaded_data)


def grow_flo(floretion: "Floretion") -> "Floretion":
    """
    Aggiunge un digit 'e' (7) in coda (ordine n -> n+1).
    """
    from floretion import Floretion
    order = int(floretion.flo_order)
    new_order = order + 1

    zero_new = 0 * Floretion.from_string('1' + 'e' * new_order)
    idx_by_dec_new = {int(v): int(i) for i, v in enumerate(zero_new.base_vec_dec_all)}
    new_coeffs = np.zeros_like(zero_new.coeff_vec_all, dtype=float)

    for base_dec, coeff in floretion.base_to_nonzero_coeff.items():
        new_base_dec = int(base_dec) * 8 + 7
        idx = idx_by_dec_new.get(new_base_dec)
        if idx is not None:
            new_coeffs[idx] += float(coeff)

    return Floretion(new_coeffs, zero_new.base_vec_dec_all, zero_new.grid_flo_loaded_data)


def proj(floretion: "Floretion") -> "Floretion":
    """
    Proiezione per quarti: Xi*Xi + Xj*Xj + Xk*Xk + Xe*Xe
    """
    Xi, Xj, Xk, Xe = split_by_leading_symbol(floretion)
    return Xi * Xi + Xj * Xj + Xk * Xk + Xe * Xe


def proj_strip_grow(floretion: "Floretion", m: int = 1) -> "Floretion":
    """
    Itera: proj -> strip -> grow, m volte.
    """
    X = floretion
    for _ in range(int(m)):
        X = proj(X)
        X = strip_flo(X)
        X = grow_flo(X)
    return X


def rotate_coeffs(X: "Floretion", shift: int = 1) -> "Floretion":
    from floretion import Floretion
    v = np.roll(X.coeff_vec_all.astype(float), int(shift))
    return Floretion(v, X.base_vec_dec_all, X.grid_flo_loaded_data)


def roll(x, shift):
    """
    Back-compat API expected by render_image.py.
    - If x is a Floretion: circularly rotate the coefficient vector by `shift`
      (uses rotate_coeffs to keep semantics consistent).
    - If x is an array-like: return numpy.roll(x, shift).
    """
    import numpy as np
    try:
        from floretion import Floretion as _F  # lazy import to avoid cycles
    except Exception:
        _F = None

    s = int(shift)
    if _F is not None and isinstance(x, _F):
        # keep the official behavior for Floretion instances
        return rotate_coeffs(x, s)
    # fall back to array rolling
    return np.roll(np.asarray(x), s)


def apply_sequence(floretion: "Floretion", methods, iter_index: int | None = None) -> "Floretion":
    """
    Applica una sequenza di operazioni.
    `methods` può essere:
      - stringa tipo "rot,tri" oppure "proj_strip_grow"
      - lista/tupla di stringhe
    """
    import re
    X = floretion

    if methods is None:
        return X

    if isinstance(methods, str):
        tokens = [t for t in re.split(r"[,\s]+", methods.strip()) if t]
    else:
        tokens = list(methods)

    for t in tokens:
        name = str(t).strip().lower()
        if name in ("rot", "rotate"):
            X = rotate_coeffs(X, shift=1)
        elif name in ("tri",):
            X = tri(X)
        elif name in ("cyc",):
            X = cyc(X)
        elif name in ("proj",):
            X = proj(X)
        elif name in ("proj_strip_grow", "psg"):
            X = proj_strip_grow(X, m=1)
        elif name in ("square",):
            sq = X * X
            try:
                X = normalize_coeffs(sq, 2.0)
            except Exception:
                X = sq
        else:
            raise ValueError(f"apply_sequence: metodo non riconosciuto: {t}")

    return X


# -----------------------------------------------------------------------------
# Centri: base vectors only
# -----------------------------------------------------------------------------

def find_center_base_vectors_only(floretion: "Floretion", decomposition_type: str = "both") -> Set[int]:
    """
    Accetta una Floretion con UN SOLO base vector non-zero e ritorna
    l'insieme dei base vectors (in decimale) che:
      - 'pos'  : commutano (ab == ba)
      - 'neg'  : anti-commutano (ab == -ba)
      - 'both' : unione
      - 'P'    : s_ab > 0
      - 'N'    : s_ab < 0
    """
    from floretion import Floretion  # import locale: evita cicli
    dec = floretion.base_vec_dec_all[np.nonzero(floretion.coeff_vec_all)[0]]
    if dec.size != 1:
        raise ValueError("find_center_base_vectors_only: serve un solo base vector non-zero.")
    a = int(dec[0])
    n = int(floretion.flo_order)

    # Griglia completa dell'ordine
    all_bases = Floretion.from_string(f'1{"e"*n}').base_vec_dec_all
    res: Set[int] = set()

    mode = str(decomposition_type).lower()
    for b in map(int, all_bases):
        s_ab = Floretion.mult_flo_sign_only(a, b, n)
        s_ba = Floretion.mult_flo_sign_only(b, a, n)
        if mode in ("pos", "positive", "cp"):
            if s_ab == s_ba:
                res.add(b)
        elif mode in ("neg", "negative", "cn"):
            if s_ab == -s_ba:
                res.add(b)
        elif mode in ("both", "cb"):
            # per definizione “both” è l’unione (qui sempre vero su base vectors)
            res.add(b)
        elif mode == "p":
            if s_ab > 0:
                res.add(b)
        elif mode == "n":
            if s_ab < 0:
                res.add(b)
        else:
            res.add(b)
    return res

def find_center(X: "Floretion", decomposition_type: str = "both") -> np.ndarray:
    """
    Compat: ritorna un array ordinato (decimali) dei centers del base vector di X.
    """
    centers = find_center_base_vectors_only(X, decomposition_type=decomposition_type)
    return np.asarray(sorted(list(centers)), dtype=int)

# -----------------------------------------------------------------------------
# Loader/saver compat per centers (thin wrapper verso floretion_centers)
# -----------------------------------------------------------------------------

def write_centers_to_file(order: int, parity: str, data: Dict[str, Iterable[int]],
                          *, path: Optional[Path] = None, fmt: Literal["npy","json"] = "npy") -> Path:
    """
    Compat: delega al nuovo modulo floretion_centers.
    """
    from lib.floretion_utils.floretion_centers import write_centers_to_file as _write
    return _write(order, parity, data, path=path, fmt=fmt)

def load_centers(order: int, parity: str = "both", storage_type: str = "npy") -> Dict[str, Any]:
    """
    Compat storica (usata solo per ordini piccoli):
      - se c’è un **unico file** (npy/json) sotto data/centers/order_{n}/{parity}/,
        ritorna l’intero dict { base_dec_str : [centers_dec...] }
      - se esistono **file segmentati** (nuovo schema con intervallo START_OCT_END_OCT),
        solleva un’eccezione (troppo grande per caricarli tutti).
    """
    from lib.floretion_utils.floretion_centers import centers_dir as _centers_dir
    dirp = _centers_dir(order, parity)
    if not dirp.exists():
        raise FileNotFoundError(f"[centers] directory non trovata: {dirp}")

    files = sorted([p for p in dirp.iterdir() if p.is_file() and p.suffix.lower() in (".npy",".json")])
    # se ci sono file segmentati (pattern con intervallo) evitiamo di caricare tutto
    has_range = any("segment_" in p.name and p.stem.count("_") >= 4 for p in files)
    if has_range:
        raise RuntimeError("[centers] load_centers è pensato per ordini piccoli (file unico). "
                           "Per 7/8 usa i loader puntuali (es. parse_special_commands / load_centers_single).")

    if not files:
        raise FileNotFoundError(f"[centers] nessun file in {dirp}")

    # preferisci npy
    npy = [p for p in files if p.suffix.lower() == ".npy"]
    path = npy[0] if npy else files[0]

    if path.suffix.lower() == ".npy":
        data = np.load(str(path), allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype == object:
            data = data.item()
        if not isinstance(data, dict):
            raise ValueError(f"[centers] npy non contiene un dict: {path}")
        # normalizza a list[int]
        return {str(k): list(map(int, v)) for k, v in data.items()}
    else:
        return json.loads(path.read_text(encoding="utf-8"))

# -----------------------------------------------------------------------------
# Tipici per UI (minimo set, compat con endpoint /typical_floretions)
# -----------------------------------------------------------------------------

def _mk_typical(name: str, s: str) -> Dict[str, str]:
    # summary == full per semplicità/robustezza
    return {"name": name, "summary": s, "full": s}

def get_typical_floretions(order: int) -> Dict[str, Any]:
    """
    Restituisce un payload semplice:
      {
        "unit": "e...e",
        "x": [ {name, summary, full}, ... ],
        "y": [ ... ]
      }
    Dove summary==full per evitare ambiguità lato client.
    """
    if order <= 0:
        order = 1
    i = "i" * order
    j = "j" * order
    k = "k" * order
    e = "e" * order

    # qualche pattern aggiuntivo per non essere “scarni”
    ij = ("ij" * ((order + 1) // 2))[:order]
    ik = ("ik" * ((order + 1) // 2))[:order]
    jk = ("jk" * ((order + 1) // 2))[:order]

    base = [
        _mk_typical(f"{order}× i-axis", i),
        _mk_typical(f"{order}× j-axis", j),
        _mk_typical(f"{order}× k-axis", k),
        _mk_typical(f"{order}× unit",  e),
        _mk_typical(f"{order}× ij pattern", ij),
        _mk_typical(f"{order}× ik pattern", ik),
        _mk_typical(f"{order}× jk pattern", jk),
    ]
    return {"unit": e, "x": base, "y": base}

# -----------------------------------------------------------------------------
# Squaring veloce con filtro centers opzionale per ordini piccoli
# -----------------------------------------------------------------------------

def square_fast(X: "Floretion",
                use_centers: bool = True,
                centers_mode: str = "both",
                centers_storage: str = "npy") -> "Floretion":
    """
    X*X con somma simmetrica (a,b)+(b,a) e cancellazione quando s_ab + s_ba = 0.
    Per evitare import circolari, il loader dei centers è importato localmente.
    """
    from floretion import Floretion

    order = int(X.flo_order)
    nz_idx = np.nonzero(X.coeff_vec_all)[0]
    if nz_idx.size == 0:
        return 0 * X

    bases = X.base_vec_dec_all[nz_idx].astype(int)
    coeffs = X.coeff_vec_all[nz_idx].astype(float)

    index_by_val = {int(v): int(i) for i, v in enumerate(X.base_vec_dec_all)}

    centers_map: Optional[Dict[str, Set[int]]] = None
    if use_centers:
        # Import locale: evita cicli; per ordini alti preferisci disattivare dal config.
        try:
            from lib.floretion_utils.floretion_centers import load_centers_map
            centers_map = load_centers_map(order, mode=centers_mode, storage_type=centers_storage)
        except Exception:
            centers_map = None  # fallback

    z = np.zeros_like(X.coeff_vec_all, dtype=float)
    m = int(len(bases))

    for i in range(m):
        a = int(bases[i])
        ca = float(coeffs[i])
        allowed_b = None
        if centers_map is not None:
            allowed_b = centers_map.get(str(a), None)

        for j in range(i, m):
            b = int(bases[j])
            if allowed_b is not None and b not in allowed_b:
                continue
            cb = float(coeffs[j])
            z_abs = Floretion.mult_flo_base_absolute_value(a, b, order)
            idx = index_by_val.get(int(z_abs))
            if idx is None:
                continue
            if i == j:
                s = Floretion.mult_flo_sign_only(a, a, order)
                contrib = (ca * cb) * float(s)
            else:
                s_ab = Floretion.mult_flo_sign_only(a, b, order)
                s_ba = Floretion.mult_flo_sign_only(b, a, order)
                ssum = s_ab + s_ba
                if ssum == 0:
                    continue
                contrib = (ca * cb) * float(ssum)
            z[idx] += contrib

    return Floretion(z, X.base_vec_dec_all, X.grid_flo_loaded_data)
