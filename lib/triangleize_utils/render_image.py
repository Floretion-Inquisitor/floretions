from __future__ import annotations

"""
render_image_annotated.py
------------------------------------------------------------------
Utilità per renderizzare batch di immagini a partire da una Floretion.
Fornisce:
  - ImageRenderer: incapsula la renderizzazione di un singolo frame 2D
    usando Triangleize;
  - save_frames_images: esegue una pipeline iterativa di trasformazioni
    (pre_methods / post_methods) e salva i frame come PNG (o altro ext);
  - save_frames_simple: scorciatoia minimale;
  - save_rolled_images: genera immagini ciclizzando i coeffs (roll).

In più, salva un meta.json ricco e alcuni asset ricostruibili
(coeffs iniziali, base_vecs e, se presenti, eventuali serie numeriche
legate a step callable).
"""

import json
from pathlib import Path
from typing import Sequence, Optional, Callable, Union, Any, List, Literal, Tuple, Dict

import numpy as np
import cv2

from floretion import Floretion

from lib.floretion_utils.config_paths import output_dir
from lib.floretion_utils.floretion_ops import apply_sequence, roll  # apply_sequence applica string/tuple/callable sequenze
from lib.triangleize_utils.centroid_distance import flo_from_centroid_distance

import gc
from .coloring import (
    ColorMode,
)


def ensure_dir(p: Path) -> None:
    """Crea la directory (ricorsivamente) se non esiste già."""
    p.mkdir(parents=True, exist_ok=True)


def _serialize_methods(methods: Sequence[Union[str, tuple[str, dict], Callable[..., Floretion]]],
                       assets_dir: Path,
                       prefix: str) -> list[dict]:
    """
    Trasforma una sequenza di "step" (pre/post) in una rappresentazione JSON-serializzabile.

    Ogni elemento della sequenza può essere:
      - stringa "nome"           -> {"type":"static", "name":"nome", "kwargs":{}}
      - tupla ("nome", {kwargs}) -> {"type":"static", "name":"nome", "kwargs":{...}}
      - callable                 -> {"type":"callable", "name":..., "meta":..., "series_file":...}

    Convenzioni per i callable (facoltative ma utili):
      - se possiedono attributo dict `_meta`, viene copiato nel record;
      - se possiedono attributi `_series` (array-like) e `_series_name` (str),
        l'array viene salvato come .npy in assets/ e nel record viene inserito
        un riferimento relativo ("series_file") e info riassuntive (min/max/len).

    Parametri
    ---------
    methods : Sequence[Union[str, tuple[str, dict], Callable[..., Floretion]]]
        La pipeline di step. L'interpretazione concreta avviene in `apply_sequence`.
    assets_dir : Path
        Directory assets (creata se serve) in cui salvare eventuali serie.
    prefix : str
        Prefisso usato per nominare file di serie (es. "pre_series_0_...").

    Ritorna
    -------
    list[dict]
        Lista di descrittori JSON-compatibili.
    """
    out: list[dict] = []
    ensure_dir(assets_dir)
    series_counter = 0

    for m in methods:
        if isinstance(m, str):
            out.append({"type": "static", "name": m, "kwargs": {}})
        elif isinstance(m, tuple) and len(m) == 2 and isinstance(m[0], str) and isinstance(m[1], dict):
            out.append({"type": "static", "name": m[0], "kwargs": m[1]})
        elif callable(m):
            name = getattr(m, "__name__", "callable")
            meta = getattr(m, "_meta", None)
            rec: dict[str, Any] = {"type": "callable", "name": name}
            if isinstance(meta, dict):
                rec["meta"] = meta
            # Serie numerica associata allo step (es. fattori tempo-varianti)
            series = getattr(m, "_series", None)
            if series is not None:
                series = np.asarray(series, dtype=float)
                series_name = getattr(m, "_series_name", "series")
                series_file = assets_dir / f"{prefix}_series_{series_counter}_{series_name}.npy"
                np.save(series_file, series)
                rec["series_name"] = str(series_name)
                rec["series_file"] = str(series_file.relative_to(assets_dir.parent))
                rec.setdefault("meta", {}).setdefault("params", {})["series_len"] = int(series.size)
                rec["series_min"] = float(series.min(initial=0.0))
                rec["series_max"] = float(series.max(initial=0.0))
                series_counter += 1
            out.append(rec)
        else:
            # Fallback per tipi imprevisti
            out.append({"type": "unknown", "repr": repr(m)})
    return out


class ImageRenderer:
    """
    Incapsula la renderizzazione 2D di una Floretion tramite Triangleize.

    Parametri
    ---------
    width, height : int
        Dimensioni del frame (pixel).
    plot_type : str
        Passato a Triangleize: 'triangle'/'dot' (alias ammessi).

    Metodi principali
    -----------------
    - render_frame(X, title): crea un'immagine BGR con la resa di X.
    - save_frames_images(...): vedi docstring sotto (pipeline & salvataggi).
    """

    def __init__(self, width: int, height: int, plot_type: str = "triangle"):
        self.width = int(width)
        self.height = int(height)
        self.plot_type = str(plot_type)

    def render_frame(self, floretion_obj: Floretion, title: Optional[str] = None) -> np.ndarray:
        """
        Renderizza un singolo frame BGR (uint8) di dimensioni (height, width).
        Usa Triangleize(...).plot_floretion(title).
        """
        # Import locale per evitare import ciclici:
        # triangleize -> triangleize_ops -> render_image -> triangleize
        from triangleize import Triangleize
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        Triangleize(floretion_obj, img, plot_type=self.plot_type).plot_floretion(title=title)
        return img

    def save_frames_images(
            self,
            floretion_start: Floretion,
            num_iterations: int,
            rel_dir: str,
            file_prefix: str = "frame",
            ext: str = "png",
            normalize_each: Optional[float] = None,
            pre_methods: Sequence[Union[str, tuple[str, dict], Callable[..., Floretion]]] = (),
            multiply_with: Optional[Floretion] = None,
            post_methods: Sequence[Union[str, tuple[str, dict], Callable[..., Floretion]]] = (),
            title_fmt: Optional[str] = None,
            metadata_name: str = "meta.json"
    ) -> Path:
        """
        Esegue una pipeline iterativa e salva N immagini in output/<rel_dir>.

        Ad ogni iterazione t:
          1) X_t = apply_sequence(X_{t-1}, pre_methods, iter_index=t)
             - ogni elemento di pre_methods può essere:
               * "nome"              -> step registrato (interpretato in apply_sequence)
               * ("nome", {kwargs})  -> step con parametri
               * callable(X_t, **?)  -> funzione che ritorna una nuova Floretion
               (apply_sequence è responsabile della "risoluzione" degli step)
          2) Se multiply_with è fornito: X_t = X_t * multiply_with
          3) X_t = apply_sequence(X_t, post_methods, iter_index=t)
          4) Se normalize_each non è None: X_t = Floretion.normalize_coeffs(X_t, normalize_each)
          5) img_t = render_frame(X_t, title=title_fmt.format(t=t) se fornito)
          6) Salva img_t come <file_prefix>_<t:05d>.<ext>

        Output aggiuntivi:
          - output/<rel_dir>/assets/start_coeffs.npy : coeff iniziali (ricostruibili)
          - output/<rel_dir>/assets/base_vecs.npy    : base vectors (decimali)
          - output/<rel_dir>/<metadata_name>         : meta.json con dettagli
            * dimensioni frame, plot_type, numero iterazioni, ecc.
            * normalizzazione per frame (se presente)
            * descrittori di pre_methods/post_methods (vedi _serialize_methods)
            * riferimento a multiply_with (coeff salvati come .npy, se presente)

        Parametri
        ---------
        floretion_start : Floretion
            Stato iniziale X_0.
        num_iterations : int
            Numero di frame/iterazioni da generare.
        rel_dir : str
            Sottocartella dentro output_dir() dove scrivere i risultati.
        file_prefix : str
            Prefisso del nome file dei frame (default "frame").
        ext : str
            Estensione immagine ("png", "jpg", ...). Non cambia il formato dei dati (BGR 8-bit).
        normalize_each : Optional[float]
            Se impostato, normalizza i coefficienti a questo valore ogni iterazione (dopo post_methods).
        pre_methods, post_methods : Sequence[Union[str, (str, dict), callable]]
            Pipeline di step (vedi sopra). apply_sequence riceve iter_index=t per permettere step tempo-varianti.
        multiply_with : Optional[Floretion]
            Se fornita, moltiplica X_t per questa floretion dopo i pre_methods.
        title_fmt : Optional[str]
            Formattazione del titolo: es. "t={t:03d}". Se errata/non formattabile, viene usata come stringa fissa.
        metadata_name : str
            Nome del file meta (default "meta.json").

        Ritorna
        -------
        Path
            Percorso assoluto della cartella di output.
        """
        out_base = output_dir() / rel_dir
        ensure_dir(out_base)
        assets_dir = out_base / "assets"
        ensure_dir(assets_dir)

        # Snapshot iniziale ricostruibile
        start_coeffs = np.asarray(floretion_start.coeff_vec_all, dtype=float)
        base_vecs = np.asarray(floretion_start.base_vec_dec_all, dtype=int)
        np.save(assets_dir / "start_coeffs.npy", start_coeffs)
        np.save(assets_dir / "base_vecs.npy", base_vecs)

        mult_ref: Optional[str] = None
        if multiply_with is not None:
            np.save(assets_dir / "multiply_with_coeffs.npy", np.asarray(multiply_with.coeff_vec_all, dtype=float))
            mult_ref = "assets/multiply_with_coeffs.npy"

        pre_desc = _serialize_methods(pre_methods, assets_dir, prefix="pre")
        post_desc = _serialize_methods(post_methods, assets_dir, prefix="post")

        # Meta arricchito (utile per auditing/riproducibilità)
        meta = {
            "width": self.width,
            "height": self.height,
            "plot_type": self.plot_type,
            "num_iterations": int(num_iterations),
            "file_prefix": str(file_prefix),
            "ext": str(ext),
            "normalize_each": float(normalize_each) if normalize_each is not None else None,
            "title_fmt": title_fmt,
            "start": {
                "order": int(floretion_start.flo_order),
                "start_coeffs_file": "assets/start_coeffs.npy",
                "base_vecs_file": "assets/base_vecs.npy",
                "nonzero_count": int(np.count_nonzero(start_coeffs)),
                "max_abs_coeff": float(np.max(np.abs(start_coeffs))) if start_coeffs.size else 0.0
            },
            "multiply_with": mult_ref,
            "pre_methods": pre_desc,
            "post_methods": post_desc
        }
        (out_base / metadata_name).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Evoluzione dinamica + salvataggio dei frame
        X = floretion_start
        for t in range(int(num_iterations)):
            X = apply_sequence(X, pre_methods, iter_index=t)
            multiply_with = apply_sequence(multiply_with, post_methods, iter_index=t)
            if multiply_with is not None:
                X = X * multiply_with

            if normalize_each is not None:
                X = Floretion.normalize_coeffs(X, float(normalize_each))

            title = None
            if title_fmt:
                try:
                    title = str(title_fmt).format(t=t)
                except Exception:
                    # Se il formato non ha {t}, usa la stringa così com'è
                    title = str(title_fmt)

            img = self.render_frame(X, title=title)
            fname = f"{file_prefix}_{t:05d}.{ext}"
            cv2.imwrite(str(out_base / fname), img)
            print(f"Wrote image {str(out_base / fname)}")

        return out_base


def save_frames_images(
        floretion_start: Floretion,
        num_iterations: int,
        rel_dir: str,
        width: int,
        height: int,
        plot_type: str = "triangle",
        file_prefix: str = "frame",
        ext: str = "png",
        normalize_each: Optional[float] = None,
        pre_methods: Sequence[Union[str, tuple[str, dict], Callable[..., Floretion]]] = (),
        multiply_with: Optional[Floretion] = None,
        post_methods: Sequence[Union[str, tuple[str, dict], Callable[..., Floretion]]] = (),
        title_fmt: Optional[str] = None,
        metadata_name: str = "meta.json"
) -> Path:
    """
    Wrapper comodo: crea un ImageRenderer con (width, height, plot_type) e
    delega a ImageRenderer.save_frames_images(...). Restituisce la cartella di output.
    """
    renderer = ImageRenderer(width=width, height=height, plot_type=plot_type)
    return renderer.save_frames_images(
        floretion_start=floretion_start,
        num_iterations=num_iterations,
        rel_dir=rel_dir,
        file_prefix=file_prefix,
        ext=ext,
        normalize_each=normalize_each,
        pre_methods=pre_methods,
        multiply_with=multiply_with,
        post_methods=post_methods,
        title_fmt=title_fmt,
        metadata_name=metadata_name,
    )


def save_frames_simple(
        floretion_start: Floretion,
        rel_dir: str,
        width: int,
        height: int,
        iterations: int,
        normalize_each: float = 1.0,
        plot_type: str = "triangle"
) -> Path:
    """
    Scorciatoia minimale: genera 'iterations' frame applicando solo la normalizzazione costante.
    Utile per smoke test/preview.
    """
    return save_frames_images(
        floretion_start=floretion_start,
        num_iterations=int(iterations),
        rel_dir=rel_dir,
        width=int(width),
        height=int(height),
        plot_type=plot_type,
        file_prefix="frame",
        ext="png",
        normalize_each=float(normalize_each),
        pre_methods=(),
        multiply_with=None,
        post_methods=(),
        title_fmt=None,
    )


def save_rolled_images(
        floretion_base: Floretion,
        rel_dir: str,
        width: int,
        height: int,
        plot_type: str = "triangle",
        num_rolls: int | None = None,
        shift: int = 1,
        block: int | None = None,
        file_prefix: str = "roll",
        ext: str = "png",
        normalize_each: float | None = None,
        title_fmt: str | None = None,
        metadata_name: str = "meta.roll.json"
) -> Path:
    """
    Genera num_rolls immagini ciclizzando i coefficienti di 'floretion_base'.

    roll(X, shift, block):
      - se block è None: roll globale del vettore di coeffs (shift posizioni).
      - se block è un intero > 0: esegue la rotazione in blocchi di lunghezza 'block'
        (utile per conservare strutture di ordine fissato).

    Per ogni r in [0 .. num_rolls-1]:
      - X_r = roll(X0, shift=r*shift, block=block)
      - se normalize_each è dato: normalizza i coeffs di X_r
      - render_frame(X_r, title=title_fmt.format(r=r) se fornito)
      - salva come <file_prefix>_<r:05d>.<ext>

    Scrive anche un meta minimale in output/<rel_dir>/<metadata_name>.
    """
    out_base = output_dir() / rel_dir
    ensure_dir(out_base)

    if num_rolls == None:
        num_rolls = 4 ** floretion_base.flo_order

    meta = {
        "mode": "roll",
        "num_rolls": int(num_rolls),
        "shift": int(shift),
        "block": int(block) if block is not None else None,
        "normalize_each": float(normalize_each) if normalize_each is not None else None,
        "plot_type": plot_type,
        "width": int(width),
        "height": int(height),
        "file_prefix": file_prefix,
        "ext": ext,
        "base_order": int(floretion_base.flo_order),
        "base_nonzero": int(np.count_nonzero(floretion_base.coeff_vec_all)),
    }
    (out_base / metadata_name).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    renderer = ImageRenderer(width=int(width), height=int(height), plot_type=plot_type)

    X0 = floretion_base
    Xr = X0
    for r in range(int(num_rolls)):

        Xr = roll(Xr, shift=int(r) * int(shift), block=block)
        #Xr = Floretion.rotate_coeffs(Xr)
        Xr = Floretion.tri(Xr)
        Xr = Xr*Xr

        #Xr = Xr + X0
        if normalize_each is not None:
            Xr = Floretion.normalize_coeffs(Xr, float(normalize_each))
        img = renderer.render_frame(Xr, title=(title_fmt.format(r=r) if title_fmt else None))
        fname = f"{file_prefix}_{r:05d}.{ext}"
        cv2.imwrite(str(out_base / fname), img)
        print(f"Saved {r} of {int(num_rolls)} images")

    return out_base




def render_centroid_distance_sequence(
    width: int,
    height: int,
    *,
    order: int,
    pct_begin: float,
    pct_end: float,
    total_frames: int,
    relation: Literal["equal", "le", "lt", "ge", "gt"] = "equal",
    coeff: float | str = "dist",
    output_dirname: str = "centroid_sweep",
    # opzioni colore (ora può essere stringa o lista)
    color_mode: Union[ColorMode, List[ColorMode]] = "diverging",
    max_val: Optional[float] = None,
    auto_clip_pct: float = 99.0,
    gamma: float = 0.7,
    sat_dist_weight: float = 0.5,
    neg_policy: Literal["none","hue-180","separate-palettes"] = "hue-180",
    band_count: int = 8,
    # operazioni sul floretion
    do_square: bool = False,
    normalize_to: Optional[float] = 2.0,
    # performance
    gc_every: int = 25,            # fai gc ogni N frame
    reuse_img: bool = True,        # riusa lo stesso buffer
) -> Tuple[Path, Dict[str, List[Path]]]:
    """
    Genera una sequenza di immagini variando la soglia percentuale di distanza dal centro.

    Salvataggio:
      output/order_{order}/{output_dirname}/{palette}/frame_XXXX.png

    Parametri principali
    --------------------
    width, height : dimensioni immagine (px)
    order         : ordine n della Floretion
    pct_begin     : percentuale iniziale di d_max (0..100)
    pct_end       : percentuale finale di d_max (0..100)
    total_frames  : numero di frame (>=1)
    relation      : "equal","le","lt","ge","gt"
    coeff         : numero -> coeff costante; "dist" -> coeff = distanza normalizzata
    color_mode    : singola palette oppure lista di palette tra:
                    "legacy","abs-hsv","diverging","gray","log-hsv","banded"
    max_val       : saturazione V; se None usa percentile auto_clip_pct
    auto_clip_pct : percentile per max automatico
    gamma         : curva V (non usata in "legacy")
    sat_dist_weight: influenza della distanza nella saturazione (non in "legacy")
    neg_policy    : gestione coeff negativi nelle modalità non-legacy
    band_count    : bande per "banded"
    do_square     : se True calcola X = X*X per ogni frame
    normalize_to  : se non None normalizza i coeff al massimo dato (es. 2.0)

    Ritorna
    -------
    (base_out_dir, frames_by_mode) con frames_by_mode[palette] = lista Path frame
    """
    # Import locale per evitare import ciclici:
    # triangleize -> triangleize_ops -> render_image -> triangleize
    from triangleize import Triangleize
    # normalizza input
    pct_begin = max(0.0, min(100.0, float(pct_begin)))
    pct_end   = max(0.0, min(100.0, float(pct_end)))
    total_frames = max(int(total_frames), 1)

    # normalizza color_mode a lista
    if isinstance(color_mode, str):
        modes: List[ColorMode] = [color_mode]  # type: ignore[list-item]
    else:
        modes = list(color_mode)

    base_out_dir = Path("output") / f"order_{order}" / output_dirname
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # meta globale
    meta = {
        "width": width, "height": height, "order": order,
        "pct_begin": pct_begin, "pct_end": pct_end, "total_frames": total_frames,
        "relation": relation, "coeff": coeff,
        "color_modes": modes, "max_val": max_val,
        "auto_clip_pct": auto_clip_pct, "gamma": gamma,
        "sat_dist_weight": sat_dist_weight, "neg_policy": neg_policy,
        "band_count": band_count, "do_square": do_square, "normalize_to": normalize_to
    }
    (base_out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # crea sottocartelle e meta per palette
    for cm in modes:
        (base_out_dir / cm).mkdir(parents=True, exist_ok=True)
        sub_meta = dict(meta)
        sub_meta["color_mode"] = cm
        (base_out_dir / cm / "meta.json").write_text(json.dumps(sub_meta, indent=2), encoding="utf-8")

    # buffer immagine riusato
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Triangleize persistente: seed con stessa griglia
    seed = Floretion.from_string(f'0{"e"*order}')
    tri = Triangleize(seed, img, plot_type="triangle")
    sierp = Floretion.get_typical_floretions("sierp_flo", order)
    frames_by_mode: Dict[str, List[Path]] = {cm: [] for cm in modes}
    denom = (total_frames - 1) if total_frames > 1 else 1

    # PNG più rapido per ridurre RAM/CPU
    png_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    X = sierp
    #flo_from_centroid_distance(order=order,
    #                               pct=pct_end,
    #                               relation=relation,
    #                               coeff=coeff)
    for fidx in range(total_frames):
        t = fidx / denom
        pct = pct_begin + (pct_end - pct_begin) * t

        # costruisci X (una volta sola)
        Y = flo_from_centroid_distance(order=order, pct=pct, relation=relation, coeff=coeff)
        Y2 = flo_from_centroid_distance(order=order, pct=100-pct, relation=relation, coeff=coeff)
        print(f"Current pct in render_centroid_distance {pct}")

        X = X + Y2*Y
        if do_square:
            X = X * X
        X = X - Y2*Y
        if normalize_to is not None and normalize_to > 0:
            X = Floretion.normalize_coeffs(X, float(normalize_to))

        # aggiorna Triangleize con X
        tri.floretion = X
        tri.coeffs = X.coeff_vec_all

        # render per ogni palette
        for cm in modes:
            if reuse_img:
                img.fill(0)

            tri.img = img  # (se volessi render multipli in parallelo, qui useresti copie)
            tri.plot_floretion(
                color_mode=cm,
                max_val=max_val,
                auto_clip_pct=auto_clip_pct,
                gamma=gamma,
                sat_dist_weight=sat_dist_weight,
                neg_policy=neg_policy,
                band_count=band_count,
            )

            fpath = base_out_dir / cm / f"frame_{fidx:04d}.png"
            print(f"Wrote {fpath}")
            ok = cv2.imwrite(str(fpath), img, png_params)
            if not ok:
                raise RuntimeError(f"cv2.imwrite fallita: {fpath}")
            frames_by_mode[cm].append(fpath)



    return base_out_dir, frames_by_mode



if __name__ == "__main__":
    pass