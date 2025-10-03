#!/usr/bin/env python3
"""
read_bin_benchmark.py
---------------------
Benchmark fast reading + spatial binning for high‑FPS .raw frame batches, then
optionally plot a time series from a selected binned ROI and/or the global mean.

Assumptions
- Your raw files are named: batch_XXXX.raw (any XXXX), each containing FRAMES_PER_FILE frames
- Each frame is stored as uint8 with shape (FRAME_H, FRAME_W) in row-major order
- Each file therefore has size = FRAME_W * FRAME_H * FRAMES_PER_FILE bytes

Example usage
-------------
python read_bin_benchmark.py \
  --run_dir output/EVT_Py_convert \
  --frame_w 1024 --frame_h 608 \
  --frames_per_file 10 \
  --expected_frames 50000 \
  --bin_y 32 --bin_x 32 \
  --fps 5000 \
  --plot \
  --roi 9,15 \
  --save_csv roi_trace.csv

Notes
-----
- If frame dims are not multiples of bin size, the script crops to the largest divisible region.
- Prints throughput (frames/s) for read+bin end-to-end.
- Uses numpy.memmap and vectorized reshape/mean (no Python loops over pixels).
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
import numpy as np

def human(n):
    for unit in ['','K','M','G','T']:
        if abs(n) < 1000.0:
            return f"{n:3.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}P"

def parse_args():
    p = argparse.ArgumentParser(description="Read .raw batches, bin frames, and benchmark throughput.")
    p.add_argument('--run_dir', type=str, required=True, help="Directory containing batch_*.raw files")
    p.add_argument('--frame_w', type=int, required=True)
    p.add_argument('--frame_h', type=int, required=True)
    p.add_argument('--frames_per_file', type=int, required=True)
    p.add_argument('--expected_frames', type=int, default=0, help="Stop after this many frames (0 = process all)")
    p.add_argument('--bin_y', type=int, default=1, help="Binning factor along height")
    p.add_argument('--bin_x', type=int, default=1, help="Binning factor along width")
    p.add_argument('--fps', type=float, default=0.0, help="Acquisition rate (Hz) for time axis; 0 to omit")
    p.add_argument('--plot', action='store_true', help="Plot ROI and global traces with matplotlib")
    p.add_argument('--roi', type=str, default=None, help="ROI bin index as 'y,x' in the binned grid (default=center)")
    p.add_argument('--save_csv', type=str, default=None, help="Optional CSV to save time, roi_trace, global_trace")
    p.add_argument('--verbose', action='store_true', help="Print extra details")
    return p.parse_args()

def main():
    args = parse_args()
    RUN_DIR = Path(args.run_dir)
    W, H = args.frame_w, args.frame_h
    FPF = args.frames_per_file
    EXP = args.expected_frames
    by, bx = args.bin_y, args.bin_x

    if not RUN_DIR.is_dir():
        print(f"ERROR: run_dir not found: {RUN_DIR}", file=sys.stderr)
        sys.exit(1)

    needed_bytes = W * H * FPF  # uint8
    files = sorted(f for f in RUN_DIR.glob("batch_*.raw") if f.stat().st_size >= needed_bytes)
    if not files:
        print("ERROR: No batch_*.raw files found with correct size.", file=sys.stderr)
        sys.exit(1)

    # Determine cropping to match binning
    Hc = (H // by) * by
    Wc = (W // bx) * bx
    if (Hc != H or Wc != W) and args.verbose:
        print(f"Cropping frames from ({H},{W}) to ({Hc},{Wc}) to match binning {by}x{bx}")

    gh, gw = (Hc // by), (Wc // bx)  # grid size after binning
    if gh == 0 or gw == 0:
        print("ERROR: Bin size larger than frame; reduce bin_x/bin_y.", file=sys.stderr)
        sys.exit(1)

    # Count frames to process
    total_frames_available = len(files) * FPF
    total_frames = total_frames_available if EXP <= 0 else min(EXP, total_frames_available)

    # ROI selection
    if args.roi:
        try:
            y_str, x_str = args.roi.split(',')
            roi_y, roi_x = int(y_str), int(x_str)
        except Exception as e:
            print("ERROR: --roi must be 'y,x' (e.g., 5,7)", file=sys.stderr)
            sys.exit(1)
    else:
        roi_y, roi_x = gh // 2, gw // 2  # center

    if not (0 <= roi_y < gh and 0 <= roi_x < gw):
        print(f"ERROR: ROI ({roi_y},{roi_x}) out of range for grid ({gh},{gw})", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Found {len(files)} files, {FPF} frames/file → up to {total_frames_available} frames")
        print(f"Processing {total_frames} frames")
        print(f"Binning factors: by={by}, bx={bx} → grid: {gh} x {gw} (per frame)")
        print(f"ROI bin index: ({roi_y}, {roi_x})")
        print(f"Per-file size check: need >= {needed_bytes} bytes")

    # Preallocate traces
    roi_trace = np.empty(total_frames, dtype=np.float32)
    global_trace = np.empty(total_frames, dtype=np.float32)

    # Benchmark
    t0 = time.perf_counter()
    processed = 0
    bytes_read = 0

    for f in files:
        if processed >= total_frames:
            break
        # Map the whole batch
        mm = np.memmap(f, dtype=np.uint8, mode='r', shape=(FPF, H, W))
        n = min(FPF, total_frames - processed)

        # Work on first n frames; crop to divisible region
        frames = mm[:n, :Hc, :Wc]

        # Vectorized binning: (n, gh, by, gw, bx) → mean over (by,bx)
        # Keep float32 to avoid overflow; this is the dominant compute step.
        binned = frames.reshape(n, gh, by, gw, bx).mean(axis=(2, 4), dtype=np.float32)

        # ROI + global traces
        roi_trace[processed:processed+n] = binned[:, roi_y, roi_x]
        global_trace[processed:processed+n] = binned.mean(axis=(1, 2), dtype=np.float32)

        processed += n
        bytes_read += int(n) * H * W  # uint8 bytes

    t1 = time.perf_counter()

    elapsed = t1 - t0
    fps = processed / elapsed if elapsed > 0 else float('inf')
    mbps = (bytes_read / elapsed) / (1024*1024) if elapsed > 0 else 0.0

    print("\n=== Benchmark ===")
    print(f"Frames processed : {processed}")
    print(f"Time elapsed     : {elapsed:0.3f} s")
    print(f"Throughput       : {fps:0.1f} frames/s")
    print(f"Read bandwidth   : {mbps:0.1f} MiB/s (raw uint8 stream)")
    print(f"Binning grid     : {gh} x {gw} (bin_y={by}, bin_x={bx})")
    print(f"ROI index        : ({roi_y}, {roi_x})")

    # Optional save CSV
    if args.save_csv:
        import csv
        out = Path(args.save_csv)
        if args.fps > 0:
            t = np.arange(processed, dtype=np.float32) / float(args.fps)
        else:
            t = np.arange(processed, dtype=np.float32)
        with out.open('w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(["t", "roi_trace", "global_trace"])
            for i in range(processed):
                w.writerow([f"{t[i]:.6f}", f"{roi_trace[i]:.6f}", f"{global_trace[i]:.6f}"])
        print(f"Saved CSV: {out.resolve()}")

    # Optional plots
    if args.plot:
        import matplotlib.pyplot as plt
        if args.fps > 0:
            t = np.arange(processed, dtype=np.float32) / float(args.fps)
            xlabel = "Time (s)"
        else:
            t = np.arange(processed, dtype=np.float32)
            xlabel = "Frame index"

        # ROI trace
        plt.figure()
        plt.plot(t, roi_trace[:processed])
        plt.xlabel(xlabel)
        plt.ylabel("ROI mean (a.u.)")
        plt.title(f"ROI Trace (bin {roi_y},{roi_x})")
        plt.tight_layout()

        # Global trace
        plt.figure()
        plt.plot(t, global_trace[:processed])
        plt.xlabel(xlabel)
        plt.ylabel("Global mean (a.u.)")
        plt.title("Global Mean Trace")
        plt.tight_layout()

        plt.show()

if __name__ == "__main__":
    main()
