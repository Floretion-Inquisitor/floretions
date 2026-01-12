"""
Operazioni radix‑4 con API in **ottale** (1=i, 2=j, 4=k, 7=e), senza uso di lettere
all'interno (eccetto per bootstrap tabella teste con from_string).

Funzioni esposte:
- P(X, d): selettore testa d∈{1,2,4,7}, ritorna Floretion (ordine n)
  + P1, P2, P4, P7
- S(X): strip della testa → Floretion di ordine n−1
- pref(X, d): reinserisce d come testa → Floretion di ordine n
  + pref_1, pref_2, pref_4, pref_7
- mul_radix4_once(X, Y): una riduzione n→n−1; i sottoprodotti X_d * Y_{d'}
  sono calcolati con **direct** (indipendente dal config) sull'ordine n−1.

Nota: usa split_by_leading_symbol / strip_flo / tabella (σ,r) all'ordine 1.
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import time
from functools import lru_cache

from floretion import Floretion
from lib.floretion_utils.floretion_ops import split_by_leading_symbol
from lib.floretion_utils.floretion_ops import strip_flo as S  # usa funzione esistente

# --- mapping ottale↔lettera solo dove serve (bootstrap) ---
D2L = {1: "i", 2: "j", 4: "k", 7: "e"}
DIGITS = (1, 2, 4, 7)





# ------------------------------------------------------------
# P(X,d): selettore per testa d∈{1,2,4,7} (ordine n)
# ------------------------------------------------------------

def P(X: Floretion, d: int) -> Floretion:
    """Seleziona i termini di X con cifra di testa d (ottale). Ritorna ordine n."""
    Xi, Xj, Xk, Xe = split_by_leading_symbol(X)
    if d == 1:
        return Xi
    if d == 2:
        return Xj
    if d == 4:
        return Xk
    if d == 7:
        return Xe
    raise ValueError("d deve essere in {1,2,4,7}")

# shortcut
#P1 = lambda X: P(X, 1)
#P2 = lambda X: P(X, 2)
#P4 = lambda X: P(X, 4)
#P7 = lambda X: P(X, 7)

# ------------------------------------------------------------
# pref(X,d): reinserisce la testa d (ottale) → ordine +1
# (implementazione in ottale, niente lettere)
# ------------------------------------------------------------

def pref(X: Floretion, d: int) -> Floretion:
    if d not in DIGITS:
        raise ValueError("d deve essere in {1,2,4,7}")
    # nuovo ordine
    new_order = int(X.flo_order) + 1
    # floretion zero di ordine new_order
    zero_new = 0 * Floretion.from_string('1' + 'e' * new_order)
    idx_by_dec_new = {int(v): int(i) for i, v in enumerate(zero_new.base_vec_dec_all)}
    new_coeffs = np.zeros_like(zero_new.coeff_vec_all, dtype=float)

    for base_dec, coeff in X.base_to_nonzero_coeff.items():
        tail_oct = format(int(base_dec), 'o')
        head_plus_tail = str(d) + tail_oct
        new_base_dec = int(head_plus_tail, 8)
        idx = idx_by_dec_new.get(new_base_dec)
        if idx is not None:
            new_coeffs[idx] += float(coeff)

    return Floretion(new_coeffs, zero_new.base_vec_dec_all, zero_new.grid_flo_loaded_data)


@lru_cache(maxsize=None)
def _zero_grid(order: int):
    Z = 0 * Floretion.from_string('1' + 'e' * order)
    idx = {int(v): int(i) for i, v in enumerate(Z.base_vec_dec_all)}
    return Z.base_vec_dec_all, idx, Z.grid_flo_loaded_data

def pref_cached(X: Floretion, d: int) -> Floretion:
    new_order = int(X.flo_order) + 1
    base_new, idx_new, grid_data = _zero_grid(new_order)
    new_coeffs = np.zeros(len(base_new), dtype=float)
    for base_dec, coeff in X.base_to_nonzero_coeff.items():
        # prepend d in ottale
        new_base_dec = int(str(d) + format(int(base_dec), 'o'), 8)
        j = idx_new.get(new_base_dec)
        if j is not None:
            new_coeffs[j] += float(coeff)
    return Floretion(new_coeffs, base_new, grid_data)



# shortcut
#pref_1 = lambda X: pref(X, 1)
#pref_2 = lambda X: pref(X, 2)
#pref_4 = lambda X: pref(X, 4)
#pref_7 = lambda X: pref(X, 7)

# ------------------------------------------------------------
# Tabella (σ, r) per teste in **ottale** costruita a runtime (ordine 1)
# ------------------------------------------------------------

def _build_head_rule_table() -> Dict[Tuple[int, int], Tuple[int, int]]:
    # Mappa ottale→decimale per basi a 1 cifra
    dec_map: Dict[int, int] = {}
    for d in DIGITS:
        Fd = Floretion.from_string(f"1{D2L[d]}")
        k = next(iter(Fd.base_to_nonzero_coeff.keys()))
        dec_map[d] = int(k)
    inv = {v: k for k, v in dec_map.items()}  # dec→ottale

    table: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for a in DIGITS:
        for b in DIGITS:
            z_abs = Floretion.mult_flo_base_absolute_value(dec_map[a], dec_map[b], 1)
            s = Floretion.mult_flo_sign_only(dec_map[a], dec_map[b], 1)
            r_digit = int(inv[int(z_abs)])
            table[(a, b)] = (int(s), r_digit)
    return table

_HEAD_RULE_NUM = _build_head_rule_table()

# ------------------------------------------------------------
# Direct multiplication indipendente dal config (uso su ordine n−1)
# ------------------------------------------------------------

def mul_direct_floretion(A: Floretion, B: Floretion) -> Floretion:
    """Moltiplicazione diretta (sparse‑friendly) senza consultare la strategia config.
    Copia dell'algoritmo di __mul__.mul_direct, adattata come funzione esterna.
    """
    assert A.flo_order == B.flo_order
    order = int(A.flo_order)

    # Riduci il numero di z candidati (come in compute_possible_vecs)
    possible_base_vecs = A.compute_possible_vecs(
        A.base_to_nonzero_coeff, B.base_to_nonzero_coeff, order, A.base_vec_dec_all
    )
    z_base_vecs = np.array(A.base_vec_dec_all)
    z_coeffs = np.zeros(len(A.base_vec_dec_all), dtype=float)

    # cache y non‑zero
    for z in possible_base_vecs:
        acc = 0.0
        for y_bv, coeff_y in B.base_to_nonzero_coeff.items():
            if coeff_y == 0:
                continue
            x_match = Floretion.mult_flo_base_absolute_value(z, y_bv, order)
            idx_x = A.base_to_grid_index.get(x_match)
            if idx_x is None:
                continue
            coeff_x = A.coeff_vec_all[idx_x]
            if coeff_x == 0:
                continue
            acc += coeff_x * coeff_y * Floretion.mult_flo_sign_only(x_match, y_bv, order)
        idx_z = int(np.where(A.base_vec_dec_all == int(z))[0][0])
        z_coeffs[idx_z] = acc
    return Floretion(z_coeffs, z_base_vecs, A.grid_flo_loaded_data)

# ------------------------------------------------------------
# Una riduzione n→n−1 con direct su sottoprodotti
# ------------------------------------------------------------

def mul_radix4_once(X: Floretion, Y: Floretion) -> Floretion:
    """Esegue una sola riduzione d'ordine n→n−1 con API ottali.
    X*Y = Σ_{d1,d2} σ(d1,d2) · pref( S(P(X,d1)) ⊛ S(P(Y,d2)) , r(d1,d2) )
    dove ⊛ = mul_direct_floretion (direct) sull'ordine n−1.
    """
    assert X.flo_order == Y.flo_order, "Ordini diversi: non moltiplicabili"
    n = int(X.flo_order)
    if n <= 1:
        # niente riduzione possibile
        return X * Y

    # Costruisci le code (ordine n−1) solo se i blocchi non sono vuoti
    tailsX: Dict[int, Floretion] = {}
    tailsY: Dict[int, Floretion] = {}
    for d in DIGITS:
        PX = P(X, d)
        if int(np.count_nonzero(PX.coeff_vec_all)) > 0:
            tailsX[d] = S(PX)
        PY = P(Y, d)
        if int(np.count_nonzero(PY.coeff_vec_all)) > 0:
            tailsY[d] = S(PY)

    if not tailsX or not tailsY:
        # uno dei due è nullo → prodotto nullo
        return 0 * X

    out = 0 * X  # ordine n

    for d1, TX in tailsX.items():
        for d2, TY in tailsY.items():
            sigma, r_digit = _HEAD_RULE_NUM[(d1, d2)]
            T = TX*TY # mul_direct_floretion(TX, TY)  # direct su ordine n−1
            Tp = pref_cached(T, r_digit)
            out.coeff_vec_all += float(sigma) * Tp.coeff_vec_all

    return out




def floretion_all_ones(order: int) -> Floretion:
    unit = Floretion.from_string('1' + 'e' * order)
    coeffs = np.ones(4 ** order, dtype=float)
    return Floretion(coeffs, unit.base_vec_dec_all, unit.grid_flo_loaded_data)




def run_once(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    print(f"{label}: {t1 - t0:.3f}s")
    return out, (t1 - t0)




def main():
    n = 8
    print(f"Order n={n} | size={4**n}")


    X = floretion_all_ones(n)
    Y = floretion_all_ones(n)


    # Warm‑up (carica eventuali cache/griglie)
    _ = X * (0 * Y)


    Z_std, t_std = run_once("std __mul__(X,Y)", lambda: X * Y)
    Z_r4, t_r4 = run_once("radix mul_radix4_once(X,Y)", lambda: mul_radix4_once(X, Y))


    equal = np.allclose(Z_std.coeff_vec_all, Z_r4.coeff_vec_all)
    print("equal results:", equal)


    # Anche caso square (X*X)
    Zs_std, ts_std = run_once("std __mul__(X,X)", lambda: X * X)
    Zs_r4, ts_r4 = run_once("radix mul_radix4_once(X,X)", lambda: mul_radix4_once(X, X))
    equal_sq = np.allclose(Zs_std.coeff_vec_all, Zs_r4.coeff_vec_all)
    print("equal results (square):", equal_sq)


    print("--- summary (seconds) ---")
    print(f"std(X,Y) : {t_std:.3f}")
    print(f"radix(X,Y) : {t_r4:.3f}")
    print(f"std(X,X) : {ts_std:.3f}")
    print(f"radix(X,X) : {ts_r4:.3f}")






if __name__ == "__main__":
    main()

# ------------------------------------------------------------
# Piccolo self‑test (commenta in produzione)
# ------------------------------------------------------------
if __name__ == "__wain__":
    X = Floretion.from_string("jjj+jje+jej+jee+ejj+eje+eej+eee")
    Y = Floretion.from_string("iii")
    Z1 = mul_radix4_once(X, Y)
    Zs = X * Y
    print("Z1:", Z1.as_floretion_notation())
    print("Zs:", Zs.as_floretion_notation())
    print("match:", np.allclose(Z1.coeff_vec_all, Zs.coeff_vec_all))
