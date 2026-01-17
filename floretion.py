from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence
import re
import logging

import numpy as np
import pandas as pd

# Logger di modulo
logger = logging.getLogger(__name__)

# Utility di bit e path (lib esterna)
from lib.floretion_utils.bitops import _bitmask, _oct666, _oct111, sgn
from lib.floretion_utils.config_paths import (
    get_grid_csv,
    data_dir,
    get_mult_strategy,
    iter_npy_segments,
    parse_segment_num,
    expected_layout_summary,
)

# Operazioni esterne (lib): wrapper statici sotto
from lib.floretion_utils.floretion_ops import (
    normalize_coeffs as _normalize_coeffs,
    fractional_coeffs as _fractional_coeffs,
    mirror as _mirror,
    split_by_leading_symbol as _split_by_leading_symbol,
    cyc as _cyc,
    tri as _tri,
    strip_flo as _strip_flo,
    grow_flo as _grow_flo,
    proj as _proj,
    proj_strip_grow as _proj_strip_grow,
    get_typical_floretions as _get_typical_floretions,
    rotate_coeffs as _rotate_coeffs,
    apply_sequence as _apply_sequence,
    square_fast as _square_fast_alias,   # resta qui
)



# Sostituisci la parte “centers” in cima a floretion.py

# floretion.py — sostituisci SOLO la parte che importava/esponeva i centers
from lib.floretion_utils.floretion_centers import (
    write_centers_to_file as _write_centers_to_file,
    load_centers as _load_centers,
    load_centers_map as _load_centers_map,
    load_centers_for_base as _load_centers_for_base,
    find_center as _find_center,
    find_center_base_vectors_only as _find_center_base_vectors_only,
)


class Floretion:
    """
    Rappresenta una Floretion con base vectors (in decimale) che in ottale usano solo cifre {1,2,4,7}.
    La moltiplicazione dei vettori base segue regole bitwise (XNOR/AND) equivalenti al caso quaternioni per ordine 1.
    """

    num_processes: int = 2  # placeholder, non usato qui

    def __init__(
        self,
        coeffs_of_base_vecs: Sequence[float],
        base_vecs: Sequence[int],
        grid_flo_loaded_data: pd.DataFrame | None = None,
        format_type: str = "dec",
    ) -> None:
        self.format_type = format_type
        temp_base_vec_dec = np.array(
            [int(str(bv), 8) if format_type == "oct" else int(bv) for bv in base_vecs]
        )

        # coefficienti non nulli
        self.base_to_nonzero_coeff: Dict[int, float] = {
            int(bv): float(coeff)
            for bv, coeff in zip(temp_base_vec_dec, coeffs_of_base_vecs)
            if abs(float(coeff)) > np.finfo(float).eps
        }

        # ordine (limite max fisso per ora)
        self.max_order: int = 10
        self.flo_order: int = self.find_flo_order(temp_base_vec_dec, self.max_order)
        self.load_floretion_data(grid_flo_loaded_data)

    # ---------------------------------------------------------------------
    # Caricamento dati griglia
    # ---------------------------------------------------------------------
    def load_floretion_data(self, grid_flo_loaded_data: pd.DataFrame | None) -> None:
        file_path = get_grid_csv(self.flo_order)
        if grid_flo_loaded_data is None:
            try:
                self.grid_flo_loaded_data = pd.read_csv(file_path, dtype={"oct": str})
            except FileNotFoundError as e:
                logger.error("=== ERRORE: file griglia non trovato ===")
                logger.error("Tentato: %s", file_path)
                logger.error(expected_layout_summary(self.flo_order))
                logger.error("========================================")
                raise FileNotFoundError(
                    f"Griglia mancante per ordine {self.flo_order}: {file_path}. "
                    f"Consulta la struttura attesa nel log."
                ) from e
        else:
            self.grid_flo_loaded_data = grid_flo_loaded_data

        self.base_vec_dec_all = self.grid_flo_loaded_data["floretion"].to_numpy()
        self.coeff_vec_all = np.zeros_like(self.base_vec_dec_all, dtype=float)
        self.map_coeffs_to_base_vectors()



    def map_coeffs_to_base_vectors(self) -> None:
        """Mappa i coefficienti non nulli sugli indici della griglia.

        Crea:
        - self.index_by_val: mappa TUTTI i base vectors della griglia → indice (0..N-1)
        - self.base_to_grid_index: solo i base vectors con coefficiente non nullo in questa istanza.
        """
        # mappa globale: ogni valore di base_vec_dec_all → indice
        self.index_by_val: Dict[int, int] = {
            int(v): int(i) for i, v in enumerate(self.base_vec_dec_all)
        }

        # mappa solo per i vettori effettivamente presenti (coeff ≠ 0)
        self.base_to_grid_index: Dict[int, int] = {}

        for base_vec, coeff in self.base_to_nonzero_coeff.items():
            idx = self.index_by_val.get(int(base_vec))
            if idx is None:
                raise ValueError(
                    f"floretion->map_coeffs_to_base_vectors->index for base_vec {base_vec} not found!"
                )
            self.base_to_grid_index[base_vec] = idx
            self.coeff_vec_all[idx] = coeff

    # ---------------------------------------------------------------------
    # Costruzione da stringa
    # ---------------------------------------------------------------------
    @classmethod
    def from_string(cls, flo_string: str, format_type: str = "dec") -> "Floretion":
        flo_string = flo_string.replace(" ", "")
        if not all(c in "0123456789ijke.+-()" for c in flo_string):
            raise ValueError("Carattere non valido. Esempio ordine 3: 3.5iii + jjk - 0.5eee")
        if any(s in flo_string for s in ("++", "+-", "-+", "--")):
            raise ValueError("Segni consecutivi non validi.")

        terms_str = re.findall(r"[\+\-]?[0-9]*\.?[0-9]*[ijke]+", flo_string)
        coeffs: List[float] = []
        base_vecs: List[int] = []

        for term in terms_str:
            m = re.match(r"([\+\-]?[0-9]*\.?[0-9]*)?([ijke]+)", term)
            if not m:
                raise ValueError(f"Termine non valido: {term}")
            coeff_str, base_vec_str = m.groups()
            if coeff_str in (None, "", "+"):
                coeff = 1.0
            elif coeff_str == "-":
                coeff = -1.0
            else:
                coeff = float(coeff_str)

            base_val = 0
            for ch in base_vec_str:
                base_val = (base_val << 3) | (1 if ch == "i" else 2 if ch == "j" else 4 if ch == "k" else 7)
            coeffs.append(coeff)
            base_vecs.append(base_val)

        return cls(coeffs_of_base_vecs=np.array(coeffs), base_vecs=np.array(base_vecs), format_type=format_type)

    # ---------------------------------------------------------------------
    # Proprietà ordine
    # ---------------------------------------------------------------------
    def find_flo_order(self, temp_base_vec_dec: np.ndarray, max_order: int) -> int:
        common_order = -1
        for base_element in temp_base_vec_dec:
            flo_order = 0
            found_order = False
            while flo_order <= max_order and not found_order:
                flo_order += 1
                if base_element <= (8**flo_order) - 1:
                    found_order = True
            if common_order == -1:
                common_order = flo_order
            elif common_order != flo_order:
                raise ValueError("Tutti i base vectors devono avere lo stesso ordine")
        return common_order

    # ---------------------------------------------------------------------
    # Operatori
    # ---------------------------------------------------------------------
    def __pow__(self, exponent: int) -> "Floretion":
        if exponent < 0:
            raise ValueError("L'esponente deve essere >= 0")
        if exponent == 0:
            return Floretion.from_string("e" * self.flo_order)
        result = self
        for _ in range(exponent - 1):
            result = result * self
        return result

    def __add__(self, other: "Floretion") -> "Floretion":
        new_coeffs = self.coeff_vec_all + other.coeff_vec_all
        return Floretion(new_coeffs, self.base_vec_dec_all, self.grid_flo_loaded_data)

    def __sub__(self, other: "Floretion") -> "Floretion":
        new_coeffs = self.coeff_vec_all - other.coeff_vec_all
        return Floretion(new_coeffs, self.base_vec_dec_all, self.grid_flo_loaded_data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Floretion):
            return False
        return np.array_equal(self.coeff_vec_all, other.coeff_vec_all)

    # ---------------------------------------------------------------------
    # Algebra su base vectors (bitwise)
    # ---------------------------------------------------------------------
    @staticmethod
    def mult_flo_base_absolute_value(a_base_val: int, b_base_val: int, flo_order: int) -> int:
        """Valore assoluto del prodotto di due base vectors (XNOR a 3-bit per cifra)."""
        bitmask = _bitmask(flo_order)
        a = abs(int(a_base_val))
        b = abs(int(b_base_val))
        return bitmask & (~(a ^ b))

    @staticmethod
    def mult_flo_sign_only(a_base_val: int, b_base_val: int, flo_order: int) -> int:
        """Solo il segno del prodotto di due base vectors (±1)."""
        bitmask = _bitmask(flo_order)
        oct666 = _oct666(flo_order)
        oct111 = _oct111(flo_order)

        pre = sgn(a_base_val) * sgn(b_base_val)
        a = abs(int(a_base_val))
        b = abs(int(b_base_val))

        a_cyc = ((a << 1) & oct666) | ((a >> 2) & oct111)
        cyc_sign = 1 if (((a_cyc & b) & bitmask).bit_count() & 1) == 1 else -1
        ord_sign = 1 if (_bitmask(flo_order).bit_count() & 1) == 1 else -1
        return pre * cyc_sign * ord_sign

    @staticmethod
    def mult_flo_base_only(a_base_val: int, b_base_val: int, flo_order: int) -> int:
        """Prodotto completo tra due base vectors (segno * valore assoluto)."""
        bitmask = _bitmask(flo_order)
        oct666 = _oct666(flo_order)
        oct111 = _oct111(flo_order)
        pre = sgn(a_base_val) * sgn(b_base_val)
        a = abs(int(a_base_val))
        b = abs(int(b_base_val))
        a_cyc = ((a << 1) & oct666) | ((a >> 2) & oct111)
        cyc_sign = 1 if (((a_cyc & b) & bitmask).bit_count() & 1) == 1 else -1
        ord_sign = 1 if (_bitmask(flo_order).bit_count() & 1) == 1 else -1
        abs_val = bitmask & (~(a ^ b))
        return abs_val * (pre * cyc_sign * ord_sign)

    # ---------------------------------------------------------------------
    # Possibili vettori Z (ottimizzazione sparsità)
    # ---------------------------------------------------------------------
    @staticmethod
    def compute_possible_vecs(
        x_base_vecs: Mapping[int, float] | Iterable[int],
        y_base_vecs: Mapping[int, float] | Iterable[int],
        flo_order: int,
        base_vecs_all: Sequence[int],
    ) -> set[int]:
        """
        Heuristica per ridurre i candidati z: cerca z tali che esista x con |x*z| in Y.
        Accetta dict (come base_to_nonzero_coeff) o iterabili di interi.
        """
        if isinstance(x_base_vecs, Mapping):
            x_keys = list(x_base_vecs.keys())
        else:
            x_keys = list(x_base_vecs)
        if isinstance(y_base_vecs, Mapping):
            y_keys = set(y_base_vecs.keys())
        else:
            y_keys = set(y_base_vecs)



        # Se X è molto denso, non vale la pena filtrare
        if len(x_keys) > len(base_vecs_all) // 2:
            return set(map(int, base_vecs_all))

        possible: set[int] = set()
        for z in base_vecs_all:
            z = int(z)
            for x in x_keys:
                product_abs = Floretion.mult_flo_base_absolute_value(int(x), z, flo_order)
                if product_abs in y_keys:
                    possible.add(z)
                    break
        return possible

    def mul_direct_legacy(self, other: "Floretion") -> "Floretion":
        possible_base_vecs = self.compute_possible_vecs(
            self.base_to_nonzero_coeff,
            other.base_to_nonzero_coeff,
            self.flo_order,
            self.base_vec_dec_all
        )

        z_base_vecs = []
        z_coeffs = []
        print(possible_base_vecs)
        for z in possible_base_vecs:
            coeff_z = 0.0
            for base_vec_y, coeff_y in other.base_to_nonzero_coeff.items():
                check_if_in_base_vec_x = Floretion.mult_flo_base_absolute_value(
                    z, base_vec_y, self.flo_order
                )
                if check_if_in_base_vec_x in self.base_to_nonzero_coeff.keys():
                    index_x = self.base_to_grid_index[check_if_in_base_vec_x]
                    coeff_x = self.coeff_vec_all[index_x]
                    coeff_z += coeff_x * coeff_y * Floretion.mult_flo_sign_only(
                        check_if_in_base_vec_x, base_vec_y, self.flo_order
                    )

            z_coeffs.append(coeff_z)
            z_base_vecs.append(z)

        return Floretion(z_coeffs, z_base_vecs, self.grid_flo_loaded_data)


    # ---------------------------------------------------------------------
    # Moltiplicazione (fast-path + rispetto strategia config + segmenti)
    # ---------------------------------------------------------------------
    def __mul__(self, other: "Floretion | float | int") -> "Floretion":
        # scalare
        if isinstance(other, (int, float)):
            return Floretion(
                self.coeff_vec_all * float(other),
                self.base_vec_dec_all,
                self.grid_flo_loaded_data,
            )

        assert self.flo_order == other.flo_order, "Ordini diversi: gruppi distinti, non moltiplicabili."

        import warnings
        from lib.floretion_utils.config_paths import load_config

        cfg = load_config()
        n = int(self.flo_order)

        def log(msg: str) -> None:
            if cfg.get("log_mul_strategy", False):
                logger.info("[mul] %s", msg)

        # ---- direct (sparse-friendly) -----------------------------------
        # ---- direct (sparse-friendly) -----------------------------------
        def mul_direct() -> "Floretion":
            """
            Moltiplicazione diretta sfruttando solo i termini con coefficiente non nullo.
            Usa:
            - self.base_to_nonzero_coeff per i contributi da self
            - other.base_to_nonzero_coeff per i contributi da other
            - self.index_by_val per mappare QUALSIASI z della griglia al suo indice.
            """
            possible_base_vecs = self.compute_possible_vecs(
                self.base_to_nonzero_coeff,
                other.base_to_nonzero_coeff,
                self.flo_order,
                self.base_vec_dec_all,
            )
            z_base_vecs = np.array(self.base_vec_dec_all)
            z_coeffs = np.zeros(len(self.base_vec_dec_all), dtype=float)

            for z in possible_base_vecs:
                z_int = int(z)
                acc = 0.0
                for y_bv, coeff_y in other.base_to_nonzero_coeff.items():
                    if coeff_y == 0:
                        continue
                    x_match = Floretion.mult_flo_base_absolute_value(z_int, y_bv, self.flo_order)
                    idx_x = self.base_to_grid_index.get(x_match)
                    if idx_x is None:
                        continue
                    coeff_x = self.coeff_vec_all[idx_x]
                    if coeff_x == 0:
                        continue
                    acc += coeff_x * coeff_y * Floretion.mult_flo_sign_only(
                        x_match, y_bv, self.flo_order
                    )

                # QUI il punto chiave: usiamo la mappa completa della griglia
                idx_z = self.index_by_val.get(z_int)
                if idx_z is None:
                    raise RuntimeError(
                        f"mul_direct: base vector z={z_int} non trovato in index_by_val "
                        f"(order={self.flo_order})"
                    )
                z_coeffs[idx_z] = acc

            return Floretion(z_coeffs, z_base_vecs, self.grid_flo_loaded_data)

        # ---- direct + square_fast (solo quando abbiamo scelto direct) ---
        def mul_direct_maybe_squarefast() -> "Floretion":
            """
            Usa square_fast *solo* se abbiamo già deciso per la via direct
            (niente NPY/segmenti per questo prodotto).
            """
            if cfg.get("enable_square_fast", True) and self == other:
                if cfg.get("log_square_fast", False):
                    logger.info("[square_fast] x==y → using square_fast (direct kernel)")
                return _square_fast_alias(
                    self,
                    use_centers=cfg.get("square_fast_use_centers", True),
                    centers_mode=cfg.get("square_fast_centers_mode", "both"),
                    centers_storage=cfg.get("square_fast_centers_storage", "npy"),
                )
            return mul_direct()

        # ---- metriche e fast-path globali --------------------------------
        nnz_x = int(np.count_nonzero(self.coeff_vec_all))
        nnz_y = int(np.count_nonzero(other.coeff_vec_all))
        prod_nnz = nnz_x * nnz_y
        instant_thresh = int(cfg.get("nnz_product_instant_threshold", 4096))

        # Warning solo se davvero può essere lungo (molti termini da considerare)
        if n > 7 and prod_nnz > instant_thresh:
            warnings.warn(
                f"[Floretion] multiplication at order {n} may take a long time "
                f"if there are many terms (nnz_x*nnz_y={prod_nnz} > {instant_thresh})!",
                RuntimeWarning,
            )

        # Fast path: prodotto nnz piccolo → direct/square_fast
        if prod_nnz <= instant_thresh:
            log(f"small nnz product (prod={prod_nnz} ≤ {instant_thresh}) → direct/square_fast")
            return mul_direct_maybe_squarefast()

        # ---- soglie classiche 4^(n-2), 4^(n-1) (configurabili) ----------
        def pow4(k: int) -> int:
            return 4 ** max(int(k), 0)

        single_off = int(cfg.get("direct_single_pow_offset", 2))  # regola 2: 4^(n-2)
        both_off = int(cfg.get("direct_both_pow_offset", 1))  # regola 3: 4^(n-1)
        t_single = pow4(n - single_off)  # 4^(n-2)
        t_both = pow4(n - both_off)  # 4^(n-1)
        enable_direct_both_rule = bool(cfg.get("enable_direct_both_rule", False))

        # ---- strategia da config per l'ordine ----------------------------
        strategy = get_mult_strategy(self.flo_order)  # "direct" | "npy-mono" | "npy-segment-num-K"
        if n > 5:
            log(f"order={n} strategy={strategy} #nonzero_x={nnz_x} #nonzero_y={nnz_y} prod={prod_nnz}")

        # ============================ n ≤ 8 ===============================
        if n <= 8:
            # 1) direct forzato da config
            if strategy == "direct":
                if n > 1:
                    log("config forces direct")
                return mul_direct_maybe_squarefast()

            # 2) preferisci i segmenti/mono se richiesti da config; usa direct solo se MOLTO sparso
            if strategy == "npy-mono" or parse_segment_num(strategy) is not None:
                # se uno è davvero sparso → direct è meglio
                if (nnz_x < t_single) or (nnz_y < t_single):
                    log(f"very sparse (nnz < 4^(n-2)={t_single}) → direct/square_fast")
                    return mul_direct_maybe_squarefast()

                # altrimenti usa NPY
                logger.info("Multiplication strategy %s", strategy)

                # ---- npy-mono ----
                if strategy == "npy-mono":
                    base = data_dir() / "npy" / f"order_{self.flo_order}"
                    ind_path = base / f"floretion_order_{self.flo_order}_segment_000_indices.npy"
                    sgn_path = base / f"floretion_order_{self.flo_order}_segment_000_signs.npy"
                    if not (ind_path.exists() and sgn_path.exists()):
                        log("npy-mono files missing → fallback to direct")
                        return mul_direct_maybe_squarefast()

                    indices_matrix = np.load(ind_path, mmap_mode="r")
                    signs_matrix = np.load(sgn_path, mmap_mode="r")
                    z_coeffs = np.zeros(len(self.base_vec_dec_all), dtype=float)

                    y_nz = np.flatnonzero(other.coeff_vec_all)
                    other_nz = other.coeff_vec_all[y_nz]

                    for z_index in range(indices_matrix.shape[0]):
                        row_idx = indices_matrix[z_index]
                        row_sgn = signs_matrix[z_index]
                        x_reordered_sel = self.coeff_vec_all[row_idx[y_nz]] * row_sgn[y_nz]
                        z_coeffs[z_index] = float(np.dot(x_reordered_sel, other_nz))

                    log("used npy-mono")
                    return Floretion(z_coeffs, self.base_vec_dec_all, self.grid_flo_loaded_data)

                # ---- npy-segments K ----
                segnum = parse_segment_num(strategy)
                logger.info("Using npy segments (strategy=%s, segnum=%s)", strategy, segnum)

                test_iter = list(iter_npy_segments(self.flo_order, limit=1))
                logger.debug("First segment probe for order %d: %s", self.flo_order, test_iter)

                # Se la strategia segmenti è configurata ma non esistono segmenti, è un errore
                if not test_iter:
                    msg = (
                        f"npy-segment strategy {strategy} configured for order {self.flo_order}, "
                        f"ma nessun file di segmento .npy trovato. "
                        f"Genera i file di segmenti o modifica order_{self.flo_order}_mult_strategy."
                    )
                    logger.error(msg)
                    logger.error(expected_layout_summary(self.flo_order))
                    raise FileNotFoundError(msg)

                z_coeffs = np.zeros(len(self.base_vec_dec_all), dtype=float)
                y_nz = np.flatnonzero(other.coeff_vec_all)
                other_nz = other.coeff_vec_all[y_nz]
                z_offset = 0

                for ind_path, sgn_path, _ in iter_npy_segments(self.flo_order, limit=segnum):
                    if not (ind_path.exists() and sgn_path.exists()):
                        msg = (
                            f"Missing .npy segment files for order {self.flo_order}: "
                            f"{ind_path} or {sgn_path} does not exist "
                            f"(strategy {strategy}, segnum={segnum})."
                        )
                        logger.error(msg)
                        raise FileNotFoundError(msg)

                    logger.debug("Loading segment indices from %s", ind_path)
                    indices_chunk = np.load(ind_path, mmap_mode="r")
                    signs_chunk = np.load(sgn_path, mmap_mode="r")
                    rows = indices_chunk.shape[0]

                    for local_z in range(rows):
                        row_idx = indices_chunk[local_z]
                        row_sgn = signs_chunk[local_z]
                        x_reordered_sel = self.coeff_vec_all[row_idx[y_nz]] * row_sgn[y_nz]
                        z_coeffs[z_offset + local_z] = float(np.dot(x_reordered_sel, other_nz))
                    z_offset += rows

                if z_offset == 0:
                    msg = (
                        f"No segment rows were loaded for order {self.flo_order} "
                        f"(strategy {strategy}, segnum={segnum})."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

                log(f"used npy-segment-num-{segnum}")
                return Floretion(z_coeffs, self.base_vec_dec_all, self.grid_flo_loaded_data)

            # 3) (opzionale) regola both < 4^(n-1) → direct  [disabilitata di default]
            if enable_direct_both_rule and (nnz_x < t_both) and (nnz_y < t_both):
                log(f"nnz < 4^(n-1) (both) → direct (t_both={t_both})")
                return mul_direct_maybe_squarefast()

            # 4) fallback: direct
            log("fallback to direct")
            return mul_direct_maybe_squarefast()

        # ======================= 8 < n ≤ 10 ===============================
        if 8 < n <= 10:
            # Nota: la fast-path per prod_nnz piccolo è già stata gestita sopra.
            # Qui siamo in regime di prodotto grande.

            # richiede entrambi molto sparsi per direct
            if (nnz_x < t_single) and (nnz_y < t_single):
                log(f"high-order direct: both < 4^(n-2) (t_single={t_single})")
                return mul_direct_maybe_squarefast()

            msg = (
                f"Exceeded maximum number of base vectors for direct computation at order {n} "
                f"(nnz_x={nnz_x}, nnz_y={nnz_y}, prod_nnz={prod_nnz})."
            )
            logger.error(msg)
            raise RuntimeError(msg)

        # =========================== n > 10 ================================
        # fallback prudente
        log("n>10 → fallback to direct")
        return mul_direct_maybe_squarefast()

    # ---------------------------------------------------------------------
    # Utility istanza
    # ---------------------------------------------------------------------
    def __rmul__(self, scalar: float | int) -> "Floretion":
        if isinstance(scalar, (int, float)):
            return Floretion(
                self.coeff_vec_all * float(scalar),
                self.base_vec_dec_all,
                self.grid_flo_loaded_data,
            )
        return NotImplemented

    def as_floretion_notation(self) -> str:
        terms: List[str] = []
        for coeff, base_vec in zip(self.coeff_vec_all, self.base_vec_dec_all):
            if coeff == 0:
                continue
            coeff_val = float(coeff)
            sign = "+" if coeff_val > 0 else "-"
            mag = abs(coeff_val)
            coeff_str = sign if np.isclose(mag, 1.0) else f"{sign}{mag:.4f}".rstrip("0").rstrip(".")

            base = int(base_vec)
            s = ""
            while base > 0:
                digit = base & 7
                s = (
                    ("i" if digit == 1 else "j" if digit == 2 else "k" if digit == 4 else "e")
                    + s
                )
                base >>= 3
            terms.append(f"{coeff_str}{s}")

        if not terms:
            return "_0_"

        res = " ".join(terms)
        res = res.replace(" + -", " - ").replace(" + +", " + ").lstrip()
        if res.startswith("+"):
            res = res[1:]
        if res.startswith("+-"):
            res = "-" + res[2:]
        res = res.replace("--", "-")
        return res.strip()

    def sum_of_squares(self) -> float:
        return float(np.dot(self.coeff_vec_all, self.coeff_vec_all))

    def abs(self) -> float:
        return float(np.sqrt(self.sum_of_squares()))

    # ---------------------------------------------------------------------
    # Statiche leggere + wrapper verso lib.ops
    # ---------------------------------------------------------------------
    @staticmethod
    def decimal_to_octal(decimal: int) -> str:
        return format(int(decimal), "o")

    @staticmethod
    def get_basevec_index(base_vec, type: str = "dec") -> int:
        if type not in {"dec", "oct"}:
            raise ValueError('type deve essere "dec" o "oct"')
        if type == "dec":
            base_dec = int(base_vec)
            oct_str = format(base_dec, "o")
        else:
            oct_str = str(base_vec)
            base_dec = int(oct_str, 8)

        order = len(oct_str)
        file_path = get_grid_csv(order)
        try:
            df = pd.read_csv(file_path, dtype={"oct": str})
        except FileNotFoundError as e:
            logger.error("=== ERRORE: file griglia non trovato (get_basevec_index) ===")
            logger.error("Tentato: %s", file_path)
            logger.error(expected_layout_summary(order))
            logger.error("============================================================")
            raise FileNotFoundError(
                f"Griglia mancante per ordine {order}: {file_path}. "
                f"Consulta la struttura attesa nel log."
            ) from e

        rows = df.index[df["floretion"].astype(int) == int(base_dec)].tolist()
        if not rows:
            raise ValueError(f"Base vector {base_dec} (oct {oct_str}) non presente in {file_path.name}")
        return int(rows[0])

    # Wrapper statici
    @staticmethod
    def normalize_coeffs(floretion: "Floretion", max_abs_value: float = 2.0) -> "Floretion":
        return _normalize_coeffs(floretion, max_abs_value)

    @staticmethod
    def fractional_coeffs(floretion: "Floretion", max_abs_value: float = 2.0) -> "Floretion":
        return _fractional_coeffs(floretion, max_abs_value)

    @staticmethod
    def mirror(floretion: "Floretion", axis: str) -> "Floretion":
        return _mirror(floretion, axis)

    @staticmethod
    def split_by_leading_symbol(floretion: "Floretion") -> tuple["Floretion", "Floretion", "Floretion", "Floretion"]:
        return _split_by_leading_symbol(floretion)

    @staticmethod
    def cyc(floretion: "Floretion") -> "Floretion":
        return _cyc(floretion)

    @staticmethod
    def tri(floretion: "Floretion") -> "Floretion":
        return _tri(floretion)

    @staticmethod
    def strip_flo(floretion: "Floretion") -> "Floretion":
        return _strip_flo(floretion)

    @staticmethod
    def grow_flo(floretion: "Floretion") -> "Floretion":
        return _grow_flo(floretion)

    @staticmethod
    def proj(floretion: "Floretion") -> "Floretion":
        return _proj(floretion)

    @staticmethod
    def proj_strip_grow(floretion: "Floretion", m: int = 1) -> "Floretion":
        return _proj_strip_grow(floretion, m)

    @staticmethod
    def write_centers_to_file(flo_order: int, decomposition_type: str = "pos"):
        return _write_centers_to_file(flo_order, decomposition_type)



    @staticmethod
    def get_typical_floretions(name: str, order: int):
        return _get_typical_floretions(name, order)


    @staticmethod
    def load_centers(
            flo_order: int,
            decomposition_type: str = "both",
            storage_type: str = "npy",
            **kwargs,
    ):
        """
        Compat:
          - l’add-on passa decomposition_type="pos|neg|both"
          - vecchio codice poteva passare mode="pos|neg|both"
        """
        # alias legacy
        parity = kwargs.get("mode", decomposition_type)
        return _load_centers(
            int(flo_order),
            parity=parity,
            storage_type=storage_type,
        )

    @staticmethod
    def write_centers_to_file(flo_order: int,
                              mode: str = "both",
                              storage_type: str = "npy") -> None:
        """
        Facciata per scrivere/rigenerare i centers su disco (se usi un job ad hoc).
        """
        _write_centers_to_file(
            order=flo_order,
            mode=mode,
            storage_type=storage_type,
        )


    @staticmethod
    def rotate_coeffs(floretion: "Floretion", shift: int = 1) -> "Floretion":
        return _rotate_coeffs(floretion, shift)

    @staticmethod
    def apply_sequence(floretion: "Floretion", methods, iter_index: int | None = None) -> "Floretion":
        return _apply_sequence(floretion, methods, iter_index)

    @staticmethod
    def from_legacy_order2(flo_legacy: str) -> "Floretion":
        from lib.floretion_utils.helpers import convert_legacy_order2_to_new

        """Shortcut: converte legacy e costruisce la Floretion nuova."""
        s = convert_legacy_order2_to_new(flo_legacy)
        return s


if __name__ == "__main__":

    Em =  Floretion.from_string("ee")

    Qua = Floretion.from_string("ie + je ")
    X = Em * Qua
    print(X.as_floretion_notation())
    exit(-1)

    # Esempi / test manuali: usa logging invece di print
    logging.basicConfig(level=logging.INFO)

    yo = "- 2'i + 2'j - 'k + i' + j' - k' + 2'ii' - 'jj' - 2'kk' + 'ij' + 'ik' + 'ji' + 'jk' - 2'kj' + 2e"
    yo = ".5'i + .5'ii' + .5'ij' + .5'ike'"
    yo_new = Floretion.from_legacy_order2(yo)
    logger.info("yo_new (converted legacy order2): %s", yo_new)
    yo_flo = Floretion.from_string(yo_new)
    logger.info("yo_flo as floretion notation: %s", yo_flo.as_floretion_notation())
    exit(-1)

    flo_x = Floretion([1, 0.5, -0.2], [11, 21, 41], format_type="oct")
    logger.info("normalized: %s", flo_x.normalize_coeffs().as_floretion_notation())

    x = Floretion.from_string("iijiije+2eeijijj")
    y = Floretion.from_string("1.5ikkiiii + ijekkkk")
    z = x * y
    logger.info("z %s", z.as_floretion_notation())

    base = Floretion.get_typical_floretions("uniform_flo", 2)
    logger.info("uniform_flo order 2: %s", base.as_floretion_notation())
    exit(-1)

    yo_array = np.array([1, 1, 2]) == np.array([1, 0, 1])
    logger.info("yo_array mask: %s", yo_array)
    logger.info("masked: %s", np.array([1, 1, 1])[yo_array])
    exit(-1)

    from lib.floretion_utils.floretion_ops import write_centers_to_file_npy, square_fast

    write_centers_to_file_npy(2, "neg")
    write_centers_to_file_npy(3, "neg")
    write_centers_to_file_npy(4, "neg")
    write_centers_to_file_npy(5, "neg")
    write_centers_to_file_npy(6, "neg")
    # write_centers_to_file_npy(7, "pos")

    exit(-1)

    x = Floretion.from_string("iijiije+2eeijijj")
    y = Floretion.from_string("1.5ikkiiii")
    z = x * y
    logger.info("z %s", z.as_floretion_notation())
    exit(-1)

    for order in [4, 5, 6]:
        base = Floretion.get_typical_floretions("uniform_flo", order)
        rng = np.random.default_rng(0)
        mask = rng.random(base.coeff_vec_all.size) < 0.1
        coeffs = rng.standard_normal(base.coeff_vec_all.size) * mask
        X = Floretion(coeffs, base.base_vec_dec_all, base.grid_flo_loaded_data)
        logger.info("start ref (order %d)", order)
        ref = X * X
        logger.info("end ref")
        logger.info("start fast")
        fast = square_fast(X, use_centers=False)
        logger.info("end fast")
        err = np.max(np.abs(ref.coeff_vec_all - fast.coeff_vec_all))
        logger.info("order %d, max abs err: %s", order, err)

    exit(-1)

    x = Floretion.from_string("iiiiijii+2ejijijji")
    y = Floretion.from_string("1.5ijkiijiiikkiijkjk")
    # z = x * y
    logger.info("x: %s", x.as_floretion_notation())
    logger.info("y: %s", y.as_floretion_notation())
    logger.info("z: %s ...", z.as_floretion_notation()[:120])
    logger.info("index(4684): %s", Floretion.get_basevec_index(4684, "dec"))
    logger.info("index(11114): %s", Floretion.get_basevec_index("11114", "oct"))
