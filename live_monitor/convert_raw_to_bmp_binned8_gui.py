#!/usr/bin/env python3
"""
High-throughput RAW → *binned* BMP converter with folder pickers, parallelism, and timing.

What’s new vs your original:
- **Built-in 8×8 binning**: 608×1024 → 76×128 per frame (64× smaller)
- **Vectorized tiling** (reshape + mean) — the fastest path for equal-size, non-overlapping ROIs
- **Detailed timings** per RAW file (map/read, bin, write) and overall bottleneck report
- **Safer N inference** from file size (optional)
- Same GUI flow (pop-up to pick input/output if CLI not provided)

Usage:
  python convert_raw_to_bmp_binned8_gui.py
  python convert_raw_to_bmp_binned8_gui.py --in D:/raw --out D:/bmp --height 608 --width 1024 --n 10 --frames-per-file 10 --bin 8

Requires: numpy, imageio, tkinter (standard library on Windows).
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Tuple, Optional, Dict, Any
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
DEFAULT_BIN = 8

# ---------------------------- helpers ----------------------------
def infer_batch_index(p: Path) -> int:
    s = p.stem
    digits = ''.join(ch for ch in s if ch.isdigit()) or '0'
    return int(digits)

def expected_size_bytes(H: int, W: int, N: int) -> int:
    return int(H) * int(W) * int(N)

def infer_n_from_size(p: Path, H: int, W: int) -> Optional[int]:
    frame_bytes = int(H) * int(W)
    if frame_bytes <= 0:
        return None
    sz = p.stat().st_size
    if sz % frame_bytes == 0:
        return sz // frame_bytes
    return None

def _bin8x(arr_hw: np.ndarray, bin_size: int) -> np.ndarray:
    """Vectorized non-overlapping binning by reshape trick.
       arr_hw: (H, W) uint8 → returns (H//b, W//b) uint8"""
    H, W = arr_hw.shape
    b = int(bin_size)
    assert H % b == 0 and W % b == 0, "H and W must be divisible by bin size for tiling."
    hh = H // b
    ww = W // b
    # mean over inner (b, b). Use float32 for speed then round→uint8
    out = arr_hw.reshape(hh, b, ww, b).mean(axis=(1, 3), dtype=np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8, copy=False)

def _bin8x_batch(mm: np.memmap, N: int, H: int, W: int, bin_size: int) -> np.ndarray:
    """Vectorized batch binning using reshape; returns (N, H//b, W//b) uint8"""
    b = int(bin_size)
    assert H % b == 0 and W % b == 0
    hh = H // b
    ww = W // b
    # reshape to (N, hh, b, ww, b), mean over (2,4)
    # Cast to float32 to compute mean, then round→uint8
    out = mm.reshape(N, hh, b, ww, b).mean(axis=(2, 4), dtype=np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8, copy=False)

# Worker runs in a separate process: map RAW → (N,H,W), bin to (N, H//b, W//b), and write BMPs
def _convert_one_raw(raw_path: str,
                     batch_idx: int,
                     H: int,
                     W: int,
                     N_user: int,
                     out_dir: str,
                     frames_per_file: int,
                     bin_size: int,
                     infer_n: bool) -> tuple[int, Dict[str, Any]]:
    tic_total = time.perf_counter()

    rp = Path(raw_path)
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Decide N
    N = N_user
    if infer_n:
        n_inferred = infer_n_from_size(rp, H, W)
        if n_inferred is not None:
            N = n_inferred

    # Validate size before mapping
    sz = rp.stat().st_size
    need = expected_size_bytes(H, W, N)
    if sz < need:
        return batch_idx, {"written": -1, "reason": "short file", "map_s": 0.0, "bin_s": 0.0, "write_s": 0.0, "total_s": 0.0, "N": N}

    # Map (N,H,W)
    tic_map = time.perf_counter()
    mm = np.memmap(rp, mode='r', dtype=np.uint8, shape=(N, H, W))
    map_s = time.perf_counter() - tic_map

    # Bin (vectorized)
    tic_bin = time.perf_counter()
    binned = _bin8x_batch(mm, N, H, W, bin_size)  # (N, hh, ww) uint8
    bin_s = time.perf_counter() - tic_bin

    # Write frames
    tic_w = time.perf_counter()
    written = 0
    for i in range(N):
        combined = batch_idx * frames_per_file + i
        bmp_name = f"{combined}.bmp"
        imageio.imwrite(outp / bmp_name, binned[i], prefer_uint8=True)
        written += 1
    write_s = time.perf_counter() - tic_w

    # Close views
    del mm, binned

    total_s = time.perf_counter() - tic_total
    return batch_idx, {"written": written, "map_s": map_s, "bin_s": bin_s, "write_s": write_s, "total_s": total_s, "N": N, "bytes": need}

def convert_dir(raw_dir: str | Path,
                out_dir: str | Path,
                H: int = DEFAULT_H,
                W: int = DEFAULT_W,
                N: int = DEFAULT_N,
                frames_per_file: int = DEFAULT_FRAMES_PER_FILE,
                pattern: str = "*.raw",
                max_workers: int | None = None,
                batches: Iterable[int] | None = None,
                bin_size: int = DEFAULT_BIN,
                infer_n: bool = True) -> None:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if H % bin_size != 0 or W % bin_size != 0:
        raise SystemExit(f"H={H} and W={W} must be divisible by bin={bin_size} for 0-copy tiling.")

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

    print(f"[INFO] Binning: {bin_size}×{bin_size}  → output {H//bin_size}×{W//bin_size} BMPs")
    totals = {"frames": 0, "bytes": 0, "map_s": 0.0, "bin_s": 0.0, "write_s": 0.0, "total_s": 0.0}
    failed = []

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_convert_one_raw,
                      str(p), int(b), int(H), int(W), int(N),
                      str(out_dir), int(frames_per_file),
                      int(bin_size), bool(infer_n))
            for (b, p) in jobs
        ]
        for fu in as_completed(futures):
            b, info = fu.result()
            if info["written"] == -1:
                print(f"[SKIP] batch {b}: file smaller than expected N*H*W bytes (N={info['N']})")
                failed.append(b)
                continue
            print(f"[OK]   batch {b}: wrote {info['written']} binned frames "
                  f"(map {info['map_s']:.3f}s, bin {info['bin_s']:.3f}s, write {info['write_s']:.3f}s, total {info['total_s']:.3f}s)")
            totals["frames"] += info["written"]
            totals["bytes"] += info.get("bytes", 0)
            totals["map_s"] += info["map_s"]
            totals["bin_s"] += info["bin_s"]
            totals["write_s"] += info["write_s"]
            totals["total_s"] += info["total_s"]

    wall_s = time.perf_counter() - t0
    print(f"\nDONE. Total frames written: {totals['frames']}  |  Wall time: {wall_s:.3f}s")
    if totals["frames"] > 0:
        fps = totals['frames'] / wall_s if wall_s > 0 else float('nan')
        gb = totals['bytes'] / (1024**3) if totals["bytes"] else 0.0
        gbps = gb / wall_s if wall_s > 0 else 0.0
        print(f"Throughput: {fps:.1f} frames/s   |   Input read: {gb:.3f} GiB total ({gbps:.3f} GiB/s)")

        # Bottleneck analysis (aggregate CPU time across workers)
        t_map = totals["map_s"]; t_bin = totals["bin_s"]; t_write = totals["write_s"]
        t_sum = t_map + t_bin + t_write
        if t_sum > 0:
            p_map = 100.0 * t_map / t_sum
            p_bin = 100.0 * t_bin / t_sum
            p_write = 100.0 * t_write / t_sum
            print(f"Aggregate CPU time breakdown (sum over workers): "
                  f"map {p_map:.1f}% | bin {p_bin:.1f}% | write {p_write:.1f}%")
            parts = {"map": t_map, "bin": t_bin, "write": t_write}
            bottleneck = max(parts, key=parts.get)
            print(f"Bottleneck likely: **{bottleneck.upper()}**")
    if failed:
        print(f"Short files (skipped): {failed}")

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
    ap = argparse.ArgumentParser(description="Parallel RAW→*binned* BMP converter with folder picker and timing")
    ap.add_argument('--in', dest='inp', help='Input folder of RAW files')
    ap.add_argument('--out', dest='out', help='Output folder for BMPs (binned)')
    ap.add_argument('--height', type=int, default=DEFAULT_H)
    ap.add_argument('--width', type=int, default=DEFAULT_W)
    ap.add_argument('--n', type=int, default=DEFAULT_N, help='Frames per RAW file (fallback if size inference disabled/fails)')
    ap.add_argument('--frames-per-file', type=int, default=DEFAULT_FRAMES_PER_FILE)
    ap.add_argument('--pattern', default='*.raw', help='Glob for RAW files (e.g., *.raw, *.bin)')
    ap.add_argument('--workers', type=int, default=None, help='Max processes (default: os.cpu_count())')
    ap.add_argument('--bin', type=int, default=DEFAULT_BIN, help='Bin size (default 8); must divide H and W.')
    ap.add_argument('--no-infer-n', action='store_true', help='Disable inferring N from file size.')
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
    print(f"Bin   : {args.bin}  (expects H%bin==0 and W%bin==0)")
    if args.no_infer_n:
        print("N inference: OFF")
    else:
        print("N inference: ON (prefer size-based per-file N)")

    convert_dir(raw_dir, out_dir,
                H=args.height, W=args.width, N=args.n,
                frames_per_file=args.frames_per_file,
                pattern=args.pattern,
                max_workers=args.workers,
                bin_size=args.bin,
                infer_n=(not args.no_infer_n))

if __name__ == "__main__":
    main()
