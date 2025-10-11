#!/usr/bin/env python3
"""
High-throughput RAW → BMP converter with folder pickers and parallelism.

Assumptions (tweak via CLI):
- Each RAW file contains N frames of size (H, W), uint8, laid out contiguously.
- Batch index is inferred from digits in the RAW filename (e.g., batch_9609.raw → 9609).
- Output filenames are a **single combined index**: combined = batch * FRAMES_PER_FILE + i → "{combined}.bmp"

Usage:
  python convert_raw_to_bmp_parallel_gui.py            # pops up dialogs to select input & output folders
  python convert_raw_to_bmp_parallel_gui.py --in D:/raw --out D:/bmp --height 608 --width 1024 --n 10 --frames-per-file 10

Requires: numpy, imageio, tkinter (standard library on Windows).
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Tuple
import numpy as np
import imageio.v2 as imageio

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

DEFAULT_H = 608
DEFAULT_W = 1024
DEFAULT_N = 10
DEFAULT_FRAMES_PER_FILE = 10

# ---------------------------- helpers ----------------------------
def infer_batch_index(p: Path) -> int:
    s = p.stem
    digits = ''.join(ch for ch in s if ch.isdigit()) or '0'
    return int(digits)


def expected_size_bytes(H: int, W: int, N: int) -> int:
    return int(H) * int(W) * int(N)


# Worker runs in a separate process: map RAW → (N,H,W) and write BMPs
def _convert_one_raw(raw_path: str,
                     batch_idx: int,
                     H: int,
                     W: int,
                     N: int,
                     out_dir: str,
                     frames_per_file: int) -> tuple[int, int]:
    rp = Path(raw_path)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Validate size before mapping
    sz = rp.stat().st_size
    need = expected_size_bytes(H, W, N)
    if sz < need:
        return batch_idx, -1  # signal short file
    if sz > need:
        # If larger, we only read the first N*H*W bytes as a view
        # Using memmap with "shape" will cap to expected bytes
        pass

    mm = np.memmap(rp, mode='r', dtype=np.uint8, shape=(N, H, W))

    # Write frames
    written = 0
    for i in range(N):
        combined = batch_idx * frames_per_file + i
        bmp_name = f"{combined}.bmp"
        imageio.imwrite(outp / bmp_name, mm[i], prefer_uint8=True)
        written += 1

    # Ensure memmap file handle closes in worker
    del mm
    return batch_idx, written


def convert_dir(raw_dir: str | Path,
                out_dir: str | Path,
                H: int = DEFAULT_H,
                W: int = DEFAULT_W,
                N: int = DEFAULT_N,
                frames_per_file: int = DEFAULT_FRAMES_PER_FILE,
                pattern: str = "*.raw",
                max_workers: int | None = None,
                batches: Iterable[int] | None = None) -> None:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(raw_dir.glob(pattern))
    if not files:
        print(f"[WARN] No files matched {pattern} under {raw_dir}")
        return

    if batches is None:
        jobs = [(infer_batch_index(p), p) for p in files]
    else:
        idx_to_file = {infer_batch_index(p): p for p in files}
        jobs = [(b, idx_to_file[b]) for b in batches if b in idx_to_file]
        if not jobs:
            print("[WARN] No jobs constructed from provided batches.")
            return

    total = 0
    short = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_convert_one_raw,
                      str(p), int(b), int(H), int(W), int(N), str(out_dir), int(frames_per_file))
            for (b, p) in jobs
        ]
        for fu in as_completed(futures):
            b, n = fu.result()
            if n == -1:
                print(f"[SKIP] batch {b}: file smaller than expected N*H*W bytes")
                short.append(b)
            else:
                print(f"[OK]   batch {b}: wrote {n} frames → {out_dir}")
                total += n

    print(f"DONE. Total frames written: {total}")
    if short:
        print(f"Short files (skipped): {short}")


# ---------------------------- CLI / GUI ----------------------------
def pick_folder_dialog(title: str) -> str | None:
    if not TK_AVAILABLE:
        return None
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)
    try:
        path = filedialog.askdirectory(title=title)
        return path or None
    finally:
        root.destroy()


def main():
    ap = argparse.ArgumentParser(description="Parallel RAW→BMP converter with folder picker")
    ap.add_argument('--in', dest='inp', help='Input folder of RAW files')
    ap.add_argument('--out', dest='out', help='Output folder for BMPs')
    ap.add_argument('--height', type=int, default=DEFAULT_H)
    ap.add_argument('--width', type=int, default=DEFAULT_W)
    ap.add_argument('--n', type=int, default=DEFAULT_N, help='Frames per RAW file')
    ap.add_argument('--frames-per-file', type=int, default=DEFAULT_FRAMES_PER_FILE)
    ap.add_argument('--pattern', default='*.raw', help='Glob for RAW files (e.g., *.raw, *.bin)')
    ap.add_argument('--workers', type=int, default=None, help='Max processes (default: os.cpu_count())')
    args = ap.parse_args()

    raw_dir = args.inp
    out_dir = args.out

    # If not provided, let the user pick
    if not raw_dir:
        raw_dir = pick_folder_dialog("Select RAW input folder")
    if not out_dir:
        out_dir = pick_folder_dialog("Select BMP output folder")

    if not raw_dir or not out_dir:
        print("No folders selected. Exiting.")
        return

    print(f"Input : {raw_dir}")
    print(f"Output: {out_dir}")
    print(f"Dims  : H={args.height}, W={args.width}, N={args.n}")
    print(f"FRPF  : {args.frames_per_file} (combined index = batch*FRPF + i)")
    print(f"Pattern: {args.pattern}")

    convert_dir(raw_dir, out_dir,
                H=args.height, W=args.width, N=args.n,
                frames_per_file=args.frames_per_file,
                pattern=args.pattern,
                max_workers=args.workers)


if __name__ == "__main__":
    main()
