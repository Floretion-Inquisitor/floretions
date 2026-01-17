# create_data_dirs.py
from __future__ import annotations

import argparse
import os
import time
import threading
import re
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

Fmt = Literal["npy", "json"]
Mode = Literal["pos", "neg", "both"]

# Centers: single file per 1..6, segmentati per 7/8
CENTERS_EXPECTED_SEGMENTS: Dict[int, int] = {
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 64,
    8: 256,
}
DEFAULT_MODES: List[Mode] = ["pos", "neg", "both"]

# accetta:
# centers_order_2_segment_000.11-77.npy
# centers_order_2_segment_000.11_77.npy
# centers_order_2_segment_000.11.77.npy
# centers_order_2_segment_000_11_77.npy
def _seg_regex_for(order: int, fmt: Fmt) -> re.Pattern:
    o = int(order)
    ext = re.escape(fmt)
    return re.compile(
        rf"^centers_order_{o}_segment_(\d{{3}})"
        rf"(?:[._])([1247]{{{o}}})(?:[-._])([1247]{{{o}}})\.{ext}$"
    )

def _repo_root() -> Path:
    return Path(__file__).resolve().parent

def _data_root(repo_root: Path) -> Path:
    return repo_root / "data"

def _centers_dir(repo_root: Path, order: int, mode: str) -> Path:
    return _data_root(repo_root) / "centers" / f"order_{int(order)}" / str(mode)

def _ensure_dirs(repo_root: Path) -> None:
    # crea cartelle (ma NON crea file)
    for order in range(1, 9):
        for mode in DEFAULT_MODES:
            _centers_dir(repo_root, order, mode).mkdir(parents=True, exist_ok=True)

def _heartbeat(stop_evt: threading.Event, label: str, every_sec: int) -> None:
    t0 = time.monotonic()
    while not stop_evt.wait(max(1, int(every_sec))):
        dt = time.monotonic() - t0
        print(f"[{label}] still working... elapsed {dt:,.0f}s", flush=True)

def _parse_modes(s: str) -> List[Mode]:
    modes_in = [m.strip() for m in s.split(",") if m.strip()]
    modes: List[Mode] = [m for m in modes_in if m in ("pos", "neg", "both")]  # type: ignore
    return modes or DEFAULT_MODES

def _orders_list(order_arg: int) -> List[int]:
    if int(order_arg) == 0:
        return sorted(CENTERS_EXPECTED_SEGMENTS.keys())
    return [int(order_arg)]

def _list_matching_range_files(folder: Path, order: int, fmt: Fmt) -> List[Path]:
    """
    Ritorna solo i file che matchano il formato con range.
    I legacy senza range NON contano.
    """
    rx = _seg_regex_for(order, fmt)
    if not folder.exists():
        return []
    out = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if rx.match(p.name):
            out.append(p)
    return sorted(out, key=lambda x: x.name)

def _test_centers_for_order(
    repo_root: Path,
    order: int,
    modes: List[Mode],
    fmt: Fmt,
    verbose: bool,
) -> Tuple[bool, str]:
    order = int(order)
    expected = CENTERS_EXPECTED_SEGMENTS.get(order, 0)
    if expected <= 0:
        return True, f"Order {order} not configured (skipped)."

    missing_any = False
    per_mode_msgs: List[str] = []

    for mode in modes:
        folder = _centers_dir(repo_root, order, mode)

        # files validi (con range)
        files = _list_matching_range_files(folder, order, fmt)
        found = len(files)

        if verbose:
            print(f"[check] order={order} mode={mode} dir={folder}", flush=True)

        if expected == 1:
            # Per ordini 1..6: richiedi segment_000 con range completo 111.. -> 777..
            start_req = "1" * order
            end_req = "7" * order
            rx = _seg_regex_for(order, fmt)

            ok = False
            for p in files:
                m = rx.match(p.name)
                if not m:
                    continue
                seg, start_oct, end_oct = m.groups()
                if seg == "000" and start_oct == start_req and end_oct == end_req:
                    ok = True
                    break

            if not ok:
                missing_any = True
                per_mode_msgs.append(f"{mode} MISSING (want seg=000 {start_req}-{end_req})")
                if verbose:
                    legacy = sorted(folder.glob(f"centers_order_{order}_segment_*.{fmt}")) if folder.exists() else []
                    if legacy:
                        print(f"[check] legacy/other files present: {[p.name for p in legacy]}", flush=True)
            else:
                per_mode_msgs.append(f"{mode} OK")
        else:
            # Ordini segmentati: vogliamo seg 000..expected-1
            rx = _seg_regex_for(order, fmt)
            segs_found: List[int] = []
            for p in files:
                m = rx.match(p.name)
                if m:
                    segs_found.append(int(m.group(1)))

            seg_set = set(segs_found)
            want_set = set(range(expected))
            missing = sorted(want_set - seg_set)

            if missing:
                missing_any = True
                per_mode_msgs.append(f"{mode} MISSING {len(missing)}/{expected} (e.g. {missing[:8]})")
            else:
                per_mode_msgs.append(f"{mode} OK {expected}/{expected}")

            if verbose and folder.exists():
                print(f"[check] matched-range files: {len(files)}", flush=True)

    if not missing_any:
        return True, f"Found all expected center files of order {order}"
    return False, f"Missing center files for order {order} ({', '.join(per_mode_msgs)})"

def _generate_centers_for_order(
    repo_root: Path,
    *,
    order: int,
    modes: List[Mode],
    fmt: Fmt,
    cores: int,
    force: bool,
    heartbeat_sec: int,
) -> None:
    order = int(order)
    expected = CENTERS_EXPECTED_SEGMENTS.get(order)
    if not expected:
        raise ValueError(f"Order {order} non supportato per generation centers.")

    # molti moduli usano ./data/...
    os.chdir(str(repo_root))

    from flo_to_centers import save_centers_singlefile, save_centers_segmented  # type: ignore

    label = f"centers order={order}"
    stop_evt = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(stop_evt, label, heartbeat_sec), daemon=True)

    print(f"[centers] START order={order} fmt={fmt} modes={modes} cores={cores} force={force}", flush=True)
    hb.start()

    try:
        for mode in modes:
            outdir = _centers_dir(repo_root, order, mode)
            outdir.mkdir(parents=True, exist_ok=True)

            # decide se generare: se già “OK” e non force -> skip
            ok, _msg = _test_centers_for_order(repo_root, order, [mode], fmt, verbose=False)
            if ok and not force:
                print(f"[centers] SKIP order={order} mode={mode}: already OK", flush=True)
                continue

            if order <= 6:
                print(f"[centers] GENERATE singlefile order={order} mode={mode} -> {outdir}", flush=True)
                save_centers_singlefile(order, mode=mode, fmt=fmt)
            else:
                print(f"[centers] GENERATE segmented order={order} mode={mode} segments={expected} -> {outdir}", flush=True)
                save_centers_segmented(
                    order,
                    mode=mode,
                    total_segments=int(expected),
                    fmt=fmt,
                    cores_per_batch=max(1, int(cores)),
                )
    finally:
        stop_evt.set()
        time.sleep(0.05)
        print(f"[centers] DONE order={order}", flush=True)

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Test/generate centers with strict filename checks + progress logging."
    )
    ap.add_argument("--root", type=str, default=None, help="Repo root (default: directory di questo script).")
    ap.add_argument("--test-centers", action="store_true", help="Test presenza dei file centers attesi (STRICT).")
    ap.add_argument("--generate-centers", action="store_true", help="Genera i file centers (se mancanti).")
    ap.add_argument("--order", type=int, default=0, help="Ordine specifico (1..8). 0 = tutti.")
    ap.add_argument("--modes", type=str, default="pos,neg,both", help="Modes (comma-separated): pos,neg,both")
    ap.add_argument("--fmt", type=str, default="npy", choices=["npy", "json"], help="Formato centers.")
    ap.add_argument("--cores", type=int, default=max(1, os.cpu_count() or 1), help="Processi per batch.")
    ap.add_argument("--force", action="store_true", help="Rigenera anche se i file esistono già.")
    ap.add_argument("--heartbeat-sec", type=int, default=15, help="Ogni quanti secondi stampare heartbeat.")
    ap.add_argument("--verbose", action="store_true", help="Stampa dettagli dei path/file controllati.")
    return ap.parse_args()

def main() -> None:
    args = _parse_args()
    repo_root = Path(args.root).expanduser().resolve() if args.root else _repo_root()
    fmt: Fmt = args.fmt  # type: ignore
    modes = _parse_modes(args.modes)
    orders = _orders_list(int(args.order))

    print(f"[root] {repo_root}", flush=True)
    # NOTA: creare directory non influenza il test dei file, ma evita errori di path
    _ensure_dirs(repo_root)

    if args.test_centers:
        all_ok = True
        for o in orders:
            ok, msg = _test_centers_for_order(repo_root, o, modes, fmt, verbose=bool(args.verbose))
            print(msg, flush=True)
            all_ok = all_ok and ok
        raise SystemExit(0 if all_ok else 2)

    if args.generate_centers:
        # report prima
        for o in orders:
            ok, msg = _test_centers_for_order(repo_root, o, modes, fmt, verbose=bool(args.verbose))
            print(msg, flush=True)

        for o in orders:
            _generate_centers_for_order(
                repo_root,
                order=o,
                modes=modes,
                fmt=fmt,
                cores=int(args.cores),
                force=bool(args.force),
                heartbeat_sec=int(args.heartbeat_sec),
            )

        # report dopo
        for o in orders:
            ok, msg = _test_centers_for_order(repo_root, o, modes, fmt, verbose=bool(args.verbose))
            print(msg, flush=True)

        return

    print("Nessuna azione richiesta. Usa --test-centers oppure --generate-centers.", flush=True)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
