import numpy as np
from typing import Sequence, List, Dict, Any, Tuple, Optional

from floretion import Floretion
from .config_paths import load_config

# ---------------------------------------------------------------------------
# Maschere 0/1 di lunghezza 16 per l'ordine 2 (definizioni STATICHE)
# ---------------------------------------------------------------------------

VES      = (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
TES      = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1)
JES      = (0,0,0,1,0,0,0,1,0,0,0,1,1,1,1,0)
JESLEFT  = (0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0)
JESRIGHT = (0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0)
LES      = (1,1,1,0,1,1,1,0,1,1,1,0,0,0,0,0)
LESTES   = (1,1,1,0,1,1,1,0,1,1,1,0,0,0,0,1)

# Mappa nome -> maschera; per aggiungere nuove maschere basta:
#  1) definire la tupla sopra;
#  2) aggiungere qui la entry corrispondente; il resto è guidato dal config.
_MASKS_BY_NAME: Dict[str, Sequence[int]] = {
    "VES": VES,
    "TES": TES,
    "JES": JES,
    "JESLEFT": JESLEFT,
    "JESRIGHT": JESRIGHT,
    "LES": LES,
    "LESTES": LESTES,
}

# ---------------------------------------------------------------------------
# Config globale
# ---------------------------------------------------------------------------

_cfg: Dict[str, Any] = load_config()

# Ampiezza default per la ricerca dei coefficienti di ricorrenza:
# cerchiamo c_j in [-MAX_RANGE_REC, +MAX_RANGE_REC].
MAX_RANGE_REC: int = int(_cfg.get("max_range_recurrences", 6))

# Ordine massimo della ricorrenza (di solito 4).
MAX_ORDER_REC: int = int(_cfg.get("max_order_recurrence", 4))

# Numero minimo di termini per provare a riconoscere una ricorrenza.
MIN_TERMS_REC: int = int(_cfg.get("min_terms_recurrence", max(10, 2 * MAX_RANGE_REC)))

# Config delle sequenze (definite in config.json -> "integer_sequences").
_INT_SEQ_CFG: Dict[str, Any] = _cfg.get("integer_sequences", {}) or {}

# Definizioni di fallback, se nel config non c'è la sezione integer_sequences.
_DEFAULT_INT_SEQS_ORDER_2: List[Dict[str, Any]] = [
    {"label": "VES",      "kind": "mask", "mask": "VES",      "sign": None},
    {"label": "VESPOS",   "kind": "mask", "mask": "VES",      "sign": "pos"},
    {"label": "VESNEG",   "kind": "mask", "mask": "VES",      "sign": "neg"},

    {"label": "JES",      "kind": "mask", "mask": "JES",      "sign": None},
    {"label": "JESPOS",   "kind": "mask", "mask": "JES",      "sign": "pos"},
    {"label": "JESNEG",   "kind": "mask", "mask": "JES",      "sign": "neg"},

    {"label": "JESLEFT",  "kind": "mask", "mask": "JESLEFT",  "sign": None},
    {"label": "JESRIGHT", "kind": "mask", "mask": "JESRIGHT", "sign": None},

    {"label": "LES",      "kind": "mask", "mask": "LES",      "sign": None},
    {"label": "TES",      "kind": "mask", "mask": "TES",      "sign": None},
    {"label": "LESTES",   "kind": "mask", "mask": "LESTES",   "sign": None},
]

_DEFAULT_INT_SEQS_ORDER_3: List[Dict[str, Any]] = [
    {"label": "VES",    "kind": "mode", "mode": "ves",    "sign": None},
    {"label": "VESPOS", "kind": "mode", "mode": "ves",    "sign": "pos"},
    {"label": "VESNEG", "kind": "mode", "mode": "ves",    "sign": "neg"},

    {"label": "JES",    "kind": "mode", "mode": "jes",    "sign": None},
    {"label": "JESPOS", "kind": "mode", "mode": "jes",    "sign": "pos"},
    {"label": "JESNEG", "kind": "mode", "mode": "jes",    "sign": "neg"},

    {"label": "LES",    "kind": "mode", "mode": "les",    "sign": None},
    {"label": "TES",    "kind": "mode", "mode": "tes",    "sign": None},
    {"label": "LESTES", "kind": "mode", "mode": "lestes", "sign": None},
]


def _get_sequence_specs_for_order(order: int) -> List[Dict[str, Any]]:
    """
    Legge dal config la lista delle sequenze da calcolare per un dato ordine.

    Ritorna sempre una lista di dict con chiavi tipiche:
      - label: stringa
      - kind : "mask" oppure "mode"
      - mask : nome della maschera (se kind == "mask")
      - mode : nome del mode (se kind == "mode")
      - sign : None / "pos" / "neg"

    Se il config non contiene una sezione valida per l'ordine richiesto,
    si usano le definizioni di fallback.
    """
    key = f"order_{int(order)}"
    section = _INT_SEQ_CFG.get(key)

    if isinstance(section, list):
        cleaned: List[Dict[str, Any]] = []
        for item in section:
            if isinstance(item, dict) and "label" in item:
                cleaned.append(item)
        if cleaned:
            return cleaned

    # Fallback
    if order == 2:
        return list(_DEFAULT_INT_SEQS_ORDER_2)
    if order == 3:
        return list(_DEFAULT_INT_SEQS_ORDER_3)
    return []


# ---------------------------------------------------------------------------
# Helper: scelta del fattore di scala p in {1,2,4}
# ---------------------------------------------------------------------------

def _pick_scale_p(seq_float: Sequence[float], eps: float = 1e-9) -> int:
    """
    Sceglie il più piccolo p in {1,2,4} tale che p * seq_float sia (numericamente)
    una sequenza di interi. Se non esiste, restituisce 0.
    """
    candidates = (1, 2, 4)
    for p in candidates:
        all_int_like = True
        for v in seq_float:
            s = p * float(v)
            if abs(round(s) - s) > eps:
                all_int_like = False
                break
        if all_int_like:
            return p
    return 0


# ---------------------------------------------------------------------------
# Ricerca di ricorrenze lineari a coefficienti interi (brute-force)
# ---------------------------------------------------------------------------

def _fit_linear_recurrence_bruteforce(
    seq: Sequence[int],
    max_order: int,
    max_range: int,
) -> Optional[tuple[int, list[int]]]:
    """
    Cerca una ricorrenza lineare a coefficienti interi per la sequenza 'seq':

        a_n = c1*a_{n-1} + c2*a_{n-2} + ... + ck*a_{n-k}

    con:
      - 1 <= k <= max_order
      - ciascun c_j in [-max_range, +max_range]
      - (c1,...,ck) non tutti nulli.

    Strategia (semplice brute-force):

      - si provano prima gli ordini bassi (k=1,2,3,...);
      - per ogni k si esplorano tutte le combinazioni in
        [-max_range..max_range]^k via backtracking;
      - la prima combinazione che soddisfa TUTTI i termini viene accettata.

    Ritorna:
      - (k, [c1,...,ck]) se trova una ricorrenza;
      - None se non ne trova nessuna con i vincoli dati.
    """
    n = len(seq)
    # euristica di sicurezza: con pochissimi termini è facile avere falsi positivi
    if n < max_order + 4:
        return None

    # Proviamo ordini 1..max_order
    for k in range(1, max_order + 1):
        if n <= k:
            # troppo corta anche solo per controllare questo ordine
            break

        coeffs = [0] * k  # buffer mutabile per il backtracking

        def backtrack(pos: int) -> Optional[list[int]]:
            """
            Backtracking su coeffs[pos].
            Quando pos == k, tutti i coefficienti sono assegnati.
            """
            if pos == k:
                # scarta il caso banale “tutti zero”
                if all(c == 0 for c in coeffs):
                    return None

                # verifica la ricorrenza su tutti i termini disponibili
                for idx in range(k, n):
                    s = 0
                    for j in range(1, k + 1):
                        s += coeffs[j - 1] * seq[idx - j]
                    if s != seq[idx]:
                        return None
                # se arriviamo qui, funziona per tutti i termini
                return list(coeffs)

            # esplora tutti i valori possibili per coeffs[pos]
            for c in range(-max_range, max_range + 1):
                coeffs[pos] = c
                res = backtrack(pos + 1)
                if res is not None:
                    return res

            return None

        found = backtrack(0)
        if found is not None:
            return k, found

    return None


def _detect_recurrence_for_int_sequence(
    seq: Sequence[int],
    *,
    max_order: int = MAX_ORDER_REC,
    max_range: int = MAX_RANGE_REC,
    min_terms: int = MIN_TERMS_REC,
) -> Optional[Dict[str, Any]]:
    """
    Wrapper di comodità:

      - controlla che la sequenza sia abbastanza lunga (>= min_terms);
      - prova a trovare una ricorrenza lineare di ordine 1..max_order
        con coefficienti in [-max_range, +max_range];
      - restituisce un piccolo dizionario di metadati oppure None.

    La sequenza 'seq' DEVE essere già di interi (come quella che otteniamo
    da flo_seq / flo_2_seq quando p>0).
    """
    n = len(seq)
    if n < min_terms:
        # troppo pochi termini: meglio non “indovinare” ricorrenze
        return None

    res = _fit_linear_recurrence_bruteforce(seq, max_order=max_order, max_range=max_range)
    if res is None:
        return None

    order, coeffs = res
    return {
        "order": order,
        "coeffs": coeffs,
        "used_terms": n,
        "max_range": max_range,
    }


def _format_recurrence_en(order: int, coeffs: Sequence[int]) -> str:
    """
    Formatta una ricorrenza lineare in inglese, ad es.:

        a_n = 3·a_{n-1} - 2·a_{n-2} + a_{n-3}

    Testo in inglese perché viene mostrato nel box delle sequenze sul sito.
    """
    parts: list[str] = []
    for j, c in enumerate(coeffs, start=1):
        if c == 0:
            continue

        term = f"a_(n-{j})"

        if not parts:
            # primo termine: gestiamo il segno direttamente
            if c == 1:
                parts.append(term)
            elif c == -1:
                parts.append(f"-{term}")
            else:
                parts.append(f"{c}·{term}")
        else:
            if c > 0:
                if c == 1:
                    parts.append(f"+ {term}")
                else:
                    parts.append(f"+ {c}·{term}")
            else:
                cc = -c
                if cc == 1:
                    parts.append(f"- {term}")
                else:
                    parts.append(f"- {cc}·{term}")

    if not parts:
        rhs = "0"
    else:
        rhs = " ".join(parts)

    return f"a_n = {rhs}"


# ---------------------------------------------------------------------------
# Sequenze per ordine 2 con maschere esplicite (lunghezza 16)
# ---------------------------------------------------------------------------

def flo_2_seq(
    mask016: Sequence[int],
    floretion: Floretion,
    num_terms: int,
    cyc: bool = False,
    sign: str | None = None,
) -> List[object]:
    """
    Calcola S_k come somma dei coefficienti selezionati (mask 0/1 di lunghezza 16)
    di X^k, con X = floretion (ordine 2).

    Restituisce [p, seq] con p in {0,1,2,4}:
      - p in {1,2,4} => p*S_k sono (numericamente) interi, seq è una lista di INT
      - p == 0       => nessuna delle scalature {1,2,4} rende p*S_k intera;
                        seq è una lista di FLOAT (gli S_k stessi).

    Vincoli:
      - floretion.flo_order deve essere 2
      - num_terms >= 10 (meglio ancora >= MIN_TERMS_REC)
      - mask016 deve avere lunghezza 16
    """
    if floretion.flo_order != 2:
        raise ValueError(f"Floretion {floretion.as_floretion_notation()} is not of order 2.")
    if num_terms < 10:
        raise ValueError(
            "num_terms must be >= 10 to determine a reliable scaling factor p (1, 2, or 4)."
        )

    if sign not in (None, "pos", "neg"):
        raise ValueError('sign must be one of {None, "pos", "neg"}')

    w = np.asarray(mask016, dtype=float)
    if w.shape != (16,):
        raise ValueError("Mask must have length 16 for order-2 floretions.")

    seq_float: List[float] = []
    X = Floretion.from_string("ee")  # unità per ordine 2

    for _ in range(num_terms):
        X = X * floretion
        if cyc:
            X = Floretion.cyc(X)

        coeffs = np.asarray(X.coeff_vec_all, dtype=float)
        if coeffs.shape != (16,):
            raise RuntimeError(f"Unexpected coeff length {coeffs.shape}, expected (16,)")

        mask_bool = (w != 0)

        if sign == "pos":
            mask_bool &= (coeffs > 0)
        elif sign == "neg":
            mask_bool &= (coeffs < 0)

        w_eff = np.zeros_like(w, dtype=float)
        w_eff[mask_bool] = w[mask_bool]

        seq_float.append(float(np.dot(coeffs, w_eff)))

    p = _pick_scale_p(seq_float, eps=1e-9)

    if p > 0:
        seq_int = [int(round(p * v)) for v in seq_float]
        return [p, seq_int]
    else:
        seq_float_rounded = [float(round(v, 12)) for v in seq_float]
        return [0, seq_float_rounded]


# ---------------------------------------------------------------------------
# Moltiplicazione esplicita ordine 2 (formula storica, sanity check)
# ---------------------------------------------------------------------------

def mult_direct_2nd_order_only(X: Floretion, Y: Floretion) -> Floretion:
    """
    Moltiplicazione per l'ordine 2 usando la formula storica 16×16.

    X e Y devono essere floretions di ordine 2 con lo stesso base_vec_dec_all.
    """
    if X.flo_order != 2 or Y.flo_order != 2:
        raise ValueError("mult_direct_2nd_order_only requires floretions of order 2.")

    if not np.array_equal(X.base_vec_dec_all, Y.base_vec_dec_all):
        raise ValueError("X and Y must share the same base_vec_dec_all.")

    base_dec = X.base_vec_dec_all
    if base_dec.size != 16:
        raise ValueError(f"For order 2, expected 16 base vectors, found {base_dec.size}.")

    dec_to_idx = {int(v): int(i) for i, v in enumerate(base_dec)}

    def get_coeffs(flo: Floretion) -> dict[int, float]:
        c: dict[int, float] = {}
        for dec, idx in dec_to_idx.items():
            val = float(flo.coeff_vec_all[idx])
            if abs(val) > np.finfo(float).eps:
                c[int(dec)] = val
        return c

    cX = get_coeffs(X)
    cY = get_coeffs(Y)

    def cx(dec: int) -> float:
        return cX.get(int(dec), 0.0)

    def cy(dec: int) -> float:
        return cY.get(int(dec), 0.0)

    # X coeffs
    P = cx(63)  # ee
    A = cx(15)  # ie
    B = cx(23)  # je
    C = cx(39)  # ke
    D = cx(57)  # ei
    G = cx(9)   # ii
    L = cx(17)  # ji
    N = cx(33)  # ki
    E = cx(58)  # ej
    J = cx(10)  # ij
    H = cx(18)  # jj
    O = cx(34)  # kj
    F = cx(60)  # ek
    K = cx(12)  # ik
    M = cx(20)  # jk
    I = cx(36)  # kk

    # Y coeffs
    p = cy(63)
    a = cy(15)
    b = cy(23)
    c = cy(39)
    d = cy(57)
    g = cy(9)
    l = cy(17)
    n = cy(33)
    e_ = cy(58)
    j_ = cy(10)
    h = cy(18)
    o = cy(34)
    f_ = cy(60)
    k_ = cy(12)
    m = cy(20)
    i_ = cy(36)

    # Formula storica
    P_out = (
        P*p - A*a - B*b - C*c
        - D*d + G*g + L*l + N*n
        - E*e_ + J*j_ + H*h + O*o
        - F*f_ + K*k_ + M*m + I*i_
    )

    A_out = (
        P*a + A*p + B*c - C*b
        - D*g - G*d - L*n + N*l
        - E*j_ - J*e_ - H*o + O*h
        - F*k_ - K*f_ - M*i_ + I*m
    )

    B_out = (
        P*b - A*c + B*p + C*a
        - D*l + G*n - L*d - N*g
        - E*h + J*o - H*e_ - O*j_
        - F*m + K*i_ - M*f_ - I*k_
    )

    C_out = (
        P*c + A*b - B*a + C*p
        - D*n - G*l + L*g - N*d
        - E*o - J*h + H*j_ - O*e_
        - F*i_ - K*m + M*k_ - I*f_
    )

    D_out = (
        P*d - A*g - B*l - C*n
        + D*p - G*a - L*b - N*c
        + E*f_ - J*k_ - H*m - O*i_
        - F*e_ + K*j_ + M*h + I*o
    )

    G_out = (
        P*g + A*d + B*n - C*l
        + D*a + G*p + L*c - N*b
        + E*k_ + J*f_ + H*i_ - O*m
        - F*j_ - K*e_ - M*o + I*h
    )

    L_out = (
        P*l - A*n + B*d + C*g
        + D*b - G*c + L*p + N*a
        + E*m - J*i_ + H*f_ + O*k_
        - F*h + K*o - M*e_ - I*j_
    )

    N_out = (
        P*n + A*l - B*g + C*d
        + D*c + G*b - L*a + N*p
        + E*i_ + J*m - H*k_ + O*f_
        - F*o - K*h + M*j_ - I*e_
    )

    E_out = (
        P*e_ - A*j_ - B*h - C*o
        - D*f_ + G*k_ + L*m + N*i_
        + E*p - J*a - H*b - O*c
        + F*d - K*g - M*l - I*n
    )

    J_out = (
        P*j_ + A*e_ + B*o - C*h
        - D*k_ - G*f_ - L*i_ + N*m
        + E*a + J*p + H*c - O*b
        + F*g + K*d + M*n - I*l
    )

    H_out = (
        P*h - A*o + B*e_ + C*j_
        - D*m + G*i_ - L*f_ - N*k_
        + E*b - J*c + H*p + O*a
        + F*l - K*n + M*d + I*g
    )

    O_out = (
        P*o + A*h - B*j_ + C*e_
        - D*i_ - G*m + L*k_ - N*f_
        + E*c + J*b - H*a + O*p
        + F*n + K*l - M*g + I*d
    )

    F_out = (
        P*f_ - A*k_ - B*m - C*i_
        + D*e_ - G*j_ - L*h - N*o
        - E*d + J*g + H*l + O*n
        + F*p - K*a - M*b - I*c
    )

    K_out = (
        P*k_ + A*f_ + B*i_ - C*m
        + D*j_ + G*e_ + L*o - N*h
        - E*g - J*d - H*n + O*l
        + F*a + K*p + M*c - I*b
    )

    M_out = (
        P*m - A*i_ + B*f_ + C*k_
        + D*h - G*o + L*e_ + N*j_
        - E*l + J*n - H*d - O*g
        + F*b - K*c + M*p + I*a
    )

    I_out = (
        P*i_ + A*m - B*k_ + C*f_
        + D*o + G*h - L*j_ + N*e_
        - E*n - J*l + H*g - O*d
        + F*c + K*b - M*a + I*p
    )

    coeffs_out_by_dec = {
        63: P_out,
        57: D_out,
        58: E_out,
        60: F_out,
        15: A_out,
        23: B_out,
        39: C_out,
        9:  G_out,
        18: H_out,
        36: I_out,
        10: J_out,
        12: K_out,
        17: L_out,
        20: M_out,
        33: N_out,
        34: O_out,
    }

    z_coeffs = np.zeros_like(X.coeff_vec_all, dtype=float)
    for dec, idx in dec_to_idx.items():
        z_coeffs[idx] = coeffs_out_by_dec.get(int(dec), 0.0)

    return Floretion(z_coeffs, base_dec, X.grid_flo_loaded_data)


# ---------------------------------------------------------------------------
# Pesi generici (VES / JES / LES / LESTES / TES) per QUALSIASI ordine
# ---------------------------------------------------------------------------

def _build_mode_weights(floretion: Floretion, mode: str) -> np.ndarray:
    """
    Costruisce il vettore dei pesi w a seconda del mode:

      - "ves":     somma di TUTTI i coefficienti
      - "jes":     somma dei coeff con b*b = -e
      - "les":     somma dei coeff con b*b = +e, ESCLUDENDO l'identità
      - "lestes":  come "les" ma INCLUDENDO l'identità
      - "tes":     solo coefficiente dell'identità

    Qui b*b è calcolato via mult_flo_base_only e confrontato con ±e.
    """
    mode = mode.lower()
    if mode not in {"ves", "jes", "les", "lestes", "tes"}:
        raise ValueError('mode must be one of {"ves","jes","les","lestes","tes"}')

    base_all = floretion.base_vec_dec_all
    n = len(base_all)
    order = int(floretion.flo_order)

    w = np.zeros(n, dtype=float)

    # indice dell'identità: per convenzione è l'ultimo
    idx_e = n - 1
    dec_e = int(base_all[idx_e])

    if mode == "ves":
        w[:] = 1.0
        return w

    # Classifica b*b rispetto a ±e
    sq_class = np.zeros(n, dtype=int)
    for i, v in enumerate(base_all):
        b_dec = int(v)
        prod = Floretion.mult_flo_base_only(b_dec, b_dec, order)
        if prod == dec_e:
            sq_class[i] = 1
        elif prod == -dec_e:
            sq_class[i] = -1
        else:
            sq_class[i] = 0

    if mode == "jes":
        w[sq_class == -1] = 1.0
    elif mode == "les":
        mask = (sq_class == 1)
        mask[idx_e] = False
        w[mask] = 1.0
    elif mode == "lestes":
        w[sq_class == 1] = 1.0
    elif mode == "tes":
        w[idx_e] = 1.0

    return w


# ---------------------------------------------------------------------------
# Sequenza generica da potenze X^k (qualsiasi ordine)
# ---------------------------------------------------------------------------

def flo_seq(
    floretion: Floretion,
    num_terms: int,
    mode: str = "ves",
    cyc: bool = False,
    sign: str | None = None,
) -> List[object]:
    """
    Sequenza generica S_k ottenuta dalle potenze di una floretion X, per QUALSIASI ordine.

    Consideriamo X^k (opzionalmente con 'cyc' ad ogni passo) e definiamo S_k come:

      - mode = "ves"    : somma di TUTTI i coefficienti
      - mode = "jes"    : somma dei coeff con b*b = -1
      - mode = "les"    : somma dei coeff con b*b = +1, ESCLUSA identità
      - mode = "lestes" : somma dei coeff con b*b = +1, IDENTITÀ inclusa
      - mode = "tes"    : coefficiente dell'identità (ultimo base vector)

    Filtro sign:
      - sign = None  : usa tutti i coefficienti selezionati
      - sign = "pos" : solo coefficienti POSITIVI nelle posizioni selezionate
      - sign = "neg" : solo coefficienti NEGATIVI nelle posizioni selezionate

    Restituisce [p, seq] con p in {0,1,2,4} (stessa convenzione di flo_2_seq).
    """
    if num_terms < 10:
        raise ValueError(
            "num_terms must be >= 10 to determine a reliable scaling factor p (1, 2, or 4)."
        )

    if sign not in (None, "pos", "neg"):
        raise ValueError('sign must be one of {None, "pos", "neg"}')

    base_all = floretion.base_vec_dec_all
    n = len(base_all)

    w = _build_mode_weights(floretion, mode)
    if w.shape != (n,):
        raise RuntimeError(f"_build_mode_weights returned shape {w.shape}, expected ({n},)")

    seq_float: List[float] = []

    order = int(floretion.flo_order)
    X = Floretion.from_string("e" * order)  # unità di ordine 'order'

    for _ in range(num_terms):
        X = X * floretion
        if cyc:
            X = Floretion.cyc(X)

        coeffs = np.asarray(X.coeff_vec_all, dtype=float)
        if coeffs.shape != (n,):
            raise RuntimeError(f"Unexpected coeff length {coeffs.shape}, expected ({n},)")

        mask_bool = (w != 0.0)

        if sign == "pos":
            mask_bool &= (coeffs > 0)
        elif sign == "neg":
            mask_bool &= (coeffs < 0)

        w_eff = np.zeros_like(w, dtype=float)
        w_eff[mask_bool] = w[mask_bool]

        seq_float.append(float(np.dot(coeffs, w_eff)))

    p = _pick_scale_p(seq_float, eps=1e-9)

    if p > 0:
        seq_int = [int(round(p * v)) for v in seq_float]
        return [p, seq_int]
    else:
        seq_float_rounded = [float(round(v, 12)) for v in seq_float]
        return [0, seq_float_rounded]


# ---------------------------------------------------------------------------
# Helper per clamp del numero di termini (usato dal backend / config)
# ---------------------------------------------------------------------------

def clamp_num_terms(raw: Any, minimum: int = MIN_TERMS_REC, maximum: int = 25) -> int:
    """
    Clamp di num_terms all'intervallo [minimum, maximum].
    Se la conversione fallisce, restituisce minimum.

    Di default il minimo è MIN_TERMS_REC, così da avere abbastanza
    termini per una ricerca di ricorrenze sensata.
    """
    try:
        n = int(raw)
    except Exception:
        return minimum
    return max(minimum, min(maximum, n))


# ---------------------------------------------------------------------------
# Analisi delle sequenze intere per ordini 2–3 (con ricorrenze)
# ---------------------------------------------------------------------------

def analyze_integer_sequences(
    Z: Floretion,
    num_terms: int = 12,
    cyc: bool = False,
) -> Dict[str, Any] | None:
    """
    Analizza le sequenze “classiche” per ordini 2 e 3 usando le
    specifiche lette dal config (integer_sequences.order_2/3).

    Ordine 2:
      - kind="mask": usa flo_2_seq con la maschera nominata (VES, JES, ...).
      - opzionalmente si possono avere anche kind="mode" per usare flo_seq.

    Ordine 3:
      - kind="mode": usa flo_seq con mode simbolici ("ves", "jes", ...).

    Per ogni sequenza:
      - si calcola [p, seq], con p in {0,1,2,4};
      - se p == 0, la sequenza viene scartata (non è intera a scala 1,2,4);
      - se p in {1,2,4}, si prova a “indovinare” una ricorrenza lineare
        con _detect_recurrence_for_int_sequence;
      - se viene trovata, si aggiunge un campo "recurrence" al dict della
        sequenza, contenente anche la forma formattata "a_n = ...".

    Ritorna un dict:

      {
        "order": int,
        "num_terms": int,
        "cyc": bool,
        "sequences": [
           {
             "label": str,
             "kind": "mask"|"mode",
             "sign": str|None,
             "p": int,
             "seq": list[int],
             "recurrence": {
               "order": int,
               "coeffs": list[int],
               "used_terms": int,
               "max_range": int,
               "text": str
             } | None
           },
           ...
        ],
        "identities": { ... },
        "recurrence_summary": {
          "orders_present": [ ... ],
          "min_order": int,
          "max_order": int,
          "max_range": int,
          "max_order_tried": int,
          "min_terms": int,
          "num_sequences": int,
          "num_with_recurrence": int
        } | None
      }

    Se non ci sono sequenze (o se l'ordine non è 2 o 3) ritorna None.
    """
    order = int(Z.flo_order)
    if order not in (2, 3):
        return None

    if num_terms < 10:
        raise ValueError("num_terms must be >= 10 for integer sequence detection.")

    # Legge la lista di "spec" per l'ordine richiesto (config-driven).
    specs = _get_sequence_specs_for_order(order)
    if not specs:
        return None

    sequences: list[dict[str, Any]] = []
    rec_orders: set[int] = set()

    for spec in specs:
        label = spec.get("label") or "?"
        kind = (spec.get("kind") or "mode").lower()
        sign = spec.get("sign", None)
        if sign not in (None, "pos", "neg"):
            # valore di sign non riconosciuto -> saltiamo questa voce
            continue

        try:
            if order == 2 and kind == "mask":
                # sequenza basata su maschera statica 0/1 di lunghezza 16
                mask_name = spec.get("mask")
                if not isinstance(mask_name, str):
                    continue
                mask = _MASKS_BY_NAME.get(mask_name.upper())
                if mask is None:
                    # maschera non definita in questo file
                    continue
                p, seq = flo_2_seq(mask, Z, num_terms, cyc=cyc, sign=sign)
            else:
                # fallback / ordine 3: sequenza dinamica via mode simbolico
                mode = spec.get("mode")
                if not isinstance(mode, str):
                    continue
                p, seq = flo_seq(Z, num_terms, mode=mode, cyc=cyc, sign=sign)
        except Exception:
            # in caso di errori per una singola sequenza, non blocchiamo tutto
            continue

        if not isinstance(p, int):
            try:
                p = int(p)
            except Exception:
                continue

        if p == 0:
            # nessuna scalatura in {1,2,4} rende la sequenza intera: saltiamo
            continue

        seq_int = list(seq)
        entry: Dict[str, Any] = {
            "label": label,
            "kind": kind,
            "sign": sign,
            "p": int(p),
            "seq": seq_int,
        }

        # Proviamo a trovare una ricorrenza lineare sulla sequenza intera.
        rec_info = _detect_recurrence_for_int_sequence(
            seq_int,
            max_order=MAX_ORDER_REC,
            max_range=MAX_RANGE_REC,
            min_terms=MIN_TERMS_REC,
        )
        if rec_info is not None:
            rec_info = dict(rec_info)
            rec_info["text"] = _format_recurrence_en(rec_info["order"], rec_info["coeffs"])
            entry["recurrence"] = rec_info
            rec_orders.add(int(rec_info["order"]))
        else:
            entry["recurrence"] = None

        sequences.append(entry)

    if not sequences:
        return None

    # Identità simboliche (come prima; il renderer decide quali mostrare)
    identities = {
        "ves_split": ["VES", "JES", "LES", "TES"],
        "ves_sign":  ["VES", "VESPOS", "VESNEG"],
        "jes_sign":  ["JES", "JESPOS", "JESNEG"],
        "jes_lr":    ["JES", "JESLEFT", "JESRIGHT"],
    }

    summary: Optional[Dict[str, Any]] = None
    if rec_orders:
        orders_sorted = sorted(rec_orders)
        summary = {
            "orders_present": orders_sorted,
            "min_order": orders_sorted[0],
            "max_order": orders_sorted[-1],
            "max_range": MAX_RANGE_REC,
            "max_order_tried": MAX_ORDER_REC,
            "min_terms": MIN_TERMS_REC,
            "num_sequences": len(sequences),
            "num_with_recurrence": sum(1 for s in sequences if s.get("recurrence") is not None),
        }

    return {
        "order": order,
        "num_terms": int(num_terms),
        "cyc": bool(cyc),
        "sequences": sequences,
        "identities": identities,
        "recurrence_summary": summary,
    }


# ---------------------------------------------------------------------------
# Rendering “umano” delle sequenze intere + ricorrenze
# ---------------------------------------------------------------------------

def render_integer_sequences(info: Dict[str, Any] | None) -> str:
    """
    Converte il risultato di analyze_integer_sequences(...) in una
    stringa testuale per il pannello delle integer sequences.

    - Non mostra sequenze con p == 0 (già filtrate da analyze_integer_sequences).
    - Le identità sono vere per definizione (nessun "checked on first terms").
    - Ogni identità viene mostrata solo se tutte le etichette coinvolte
      sono effettivamente presenti nelle sequenze calcolate.
    - Se per una sequenza viene trovata una ricorrenza lineare, la mostra
      subito sotto la lista dei termini.
    """
    if not info:
        return ""

    seqs: List[Dict[str, Any]] = info.get("sequences") or []
    if not seqs:
        return ""

    order = info.get("order", "?")
    num_terms = info.get("num_terms", "?")
    cyc = bool(info.get("cyc", False))

    lines: List[str] = []
    lines.append(
        f"Integer sequences from Z = X*Y (order {order}, terms = {num_terms}, cyc = {'on' if cyc else 'off'})."
    )

    # Eventuale riepilogo delle ricorrenze
    summary = info.get("recurrence_summary") or {}
    orders_present = summary.get("orders_present") or []
    if orders_present:
        orders_str = ", ".join(str(o) for o in orders_present)
        max_range = summary.get("max_range", MAX_RANGE_REC)
        min_terms = summary.get("min_terms", MIN_TERMS_REC)
        lines.append(
            f"Detected linear recurrences on integer sequences (orders = {orders_str}), "
            f"coeffs in [-{max_range}, +{max_range}] based on at least {min_terms} terms."
        )

    lines.append("")

    # Etichette presenti, per decidere quali identità ha senso mostrare
    present_labels = {s["label"] for s in seqs if "label" in s}

    # Determina quanti termini mostrare
    def _get_max_show(seq: List[int]) -> int:
        try:
            requested = int(num_terms)
            if requested <= 0:
                return len(seq)
            return min(requested, len(seq))
        except Exception:
            return len(seq)

    # Mostra ogni sequenza
    for s in seqs:
        label = s.get("label", "?")
        p = int(s.get("p", 0))
        seq = s.get("seq") or []

        max_show = _get_max_show(seq)
        head = ", ".join(str(v) for v in seq[:max_show])

        if len(seq) > max_show:
            head += ", ..."

        if p == 1:
            lines.append(f"{label}: [{head}]")
        else:
            lines.append(f"{label} (p = {p}): [{head}]")

        rec = s.get("recurrence")
        if rec:
            r_order = rec.get("order", "?")
            r_text = rec.get("text") or ""
            if r_text:
                lines.append(f"  Recurrence (order {r_order}): {r_text}")

    # Aggiungi le identità simboliche (solo se tutti i membri sono presenti)
    lines.append("")
    lines.append("Identities:")

    if {"VES", "JES", "LES", "TES"}.issubset(present_labels):
        lines.append("  VES   = JES + LES + TES")

    if {"VES", "VESPOS", "VESNEG"}.issubset(present_labels):
        lines.append("  VES   = VESPOS + VESNEG")

    if {"JES", "JESPOS", "JESNEG"}.issubset(present_labels):
        lines.append("  JES   = JESPOS + JESNEG")

    if {"JES", "JESLEFT", "JESRIGHT"}.issubset(present_labels):
        lines.append("  JES   = JESLEFT + JESRIGHT")

    return "\n".join(lines)
