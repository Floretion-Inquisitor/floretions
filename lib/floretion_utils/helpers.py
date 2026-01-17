# lib/floretion_utils/helpers.py
from __future__ import annotations

import re
import numpy as np
from typing import Tuple, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from floretion import Floretion

from lib.floretion_utils.floretion_centers import (
    load_centers_single,   # auto npy/json + range aware
    Mode,
)

__all__ = ["parse_special_commands"]

def parse_special_commands(input_str: str, storage_type: str = "npy") -> Floretion:
    """
    Interpreta comandi speciali che restituiscono una Floretion costruita da insiemi di 'centers'.

      Cp(<base>)  -> center(positive) del base vector <base>
      Cn(<base>)  -> center(negative)
      Cb(<base>)  -> center(both)
      P(<base>)   -> { b | s_ab > 0 }
      N(<base>)   -> { b | s_ab < 0 }

    <base> è una stringa su {i,j,k,e} (es: "ikekji").
    Se non è un comando speciale, interpreta come floretion-notation standard.
    """
    from floretion import Floretion  # import locale: evita import circolari

    command_match = re.match(r"(Cp|Cn|Cb|P|N)\(([ij ke]+)\)", input_str.replace(" ", ""))
    if command_match:
        command, base_vec = command_match.groups()

        f_base = Floretion.from_string(base_vec)
        order = f_base.flo_order
        base_oct_str = (
            base_vec.replace("i", "1")
                    .replace("j", "2")
                    .replace("k", "4")
                    .replace("e", "7")
        )
        base_dec = int(base_oct_str, 8)

        cmd_to_mode: dict[str, Mode] = {"Cp": "pos", "Cn": "neg", "Cb": "both", "P": "P", "N": "N"}
        mode: Mode = cmd_to_mode.get(command, "both")

        centers_dec = load_centers_single(int(order), mode, int(base_dec), storage_type=storage_type)

        # coeff=1 su tutti i centers
        coeffs = np.ones(len(centers_dec), dtype=float)
        return Floretion(coeffs, np.asarray(centers_dec, dtype=int))

    # fallback: normale parser floretion
    return Floretion.from_string(input_str)


def convert_legacy_order2_to_new(flo_legacy: str, *, strict: bool = True) -> str:
    """
    Converte la notazione legacy (ordine 2) con apici in notazione attuale.
    Esempi:
      "'i"  -> "ei"
      "i'"  -> "ie"
      "'ij'"-> "ij"
      "2e"  -> "2ee"
    Se strict=True, segnala simboli singoli ambigui (i/j/k non affiancati).
    """
    s = flo_legacy

    # 0) togli spazi facoltativi: from_string li ignora comunque
    s = s.replace(" ", "")

    # 1) coppie con apici: 'ab', ab', 'ab'
    #    (nel legacy servivano solo da “delimitatori”: rimuovili)
    s = re.sub(r"'([ijk]{2})'", r"\1", s)
    s = re.sub(r"'([ijk]{2})",  r"\1", s)
    s = re.sub(r"([ijk]{2})'",  r"\1", s)

    # 2) simboli singoli con apice a dx/sx
    #    i' -> ie,  'i -> ei  (idem per j,k)
    s = re.sub(r"([ijk])'", r"\1e", s)
    s = re.sub(r"'([ijk])", r"e\1", s)

    # 3) unità: e, 'e, e', 'e' -> ee  (quando 'e' è token isolato)
    s = re.sub(r"(?<![ijke])'?e'?(?![ijke])", "ee", s)

    # 4) apici residui (se ce ne fossero ancora per casi strani): rimuovi
    s = s.replace("'", "")

    # 5) facoltativo: in strict, rifiuta simboli singoli ambigui (i/j/k isolati)
    if strict:
        ambigui = re.findall(r"(?<![ijke])[ijk](?![ijke])", s)
        if ambigui:
            uniq = ", ".join(sorted(set(ambigui)))
            raise ValueError(
                f"Notazione legacy ambigua per simbolo singolo: {uniq}. "
                "Nel legacy usa 'i per ei oppure i' per ie (idem j,k), "
                "oppure passa strict=False per forzare."
            )

    # 6) sanity check: solo caratteri ammessi da Floretion.from_string
    #    (che poi farà il parsing dei termini)
    if not all(c in "0123456789ijke.+-()" for c in s):
        raise ValueError("Carattere non valido dopo la conversione legacy→nuovo.")

    return s