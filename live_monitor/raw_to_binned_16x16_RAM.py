#!/usr/bin/env python3
# RAW -> 16x16 uint16 sums (RAM) + save NPY & ImageJ BigTIFF
# - Minimal UI: pick input & output folder (Tk)
# - Simple natural sort, no heavy edge-case handling

import re
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import tifffile as tiff

# ---- CONFIG ----
H, W = 608, 1024          # frame size
BIN = 16                  # spatial bin (BIN x BIN)
N_FRAMES_PER_RAW = 10     # frames per .raw
# ----------------

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def pick_folder(title):
    root = tk.Tk(); root.withdraw()
    d = filedialog.askdirectory(title=title)
    root.update(); root.destroy()
    return Path(d) if d else None

def main():
    in_dir = pick_folder("Select RAW input folder")
    if not in_dir: print("No input selected"); return
    out_dir = pick_folder("Select output folder (will save .npy and .tif)") or in_dir / "binned_16x16"
    out_dir.mkdir(parents=True, exist_ok=True)

    raws = sorted(in_dir.glob("*.raw"), key=lambda p: natural_key(p.name))
    if not raws: print("No .raw files"); return

    hh, ww = H // BIN, W // BIN
    total_frames = len(raws) * N_FRAMES_PER_RAW
    stack = np.empty((total_frames, hh, ww), dtype=np.uint16)

    print(f"Files: {len(raws)} | Frames total: {total_frames} | Grid: {hh}x{ww}")

    for idx, rp in enumerate(raws):
        buf = np.fromfile(rp, dtype=np.uint8, count=N_FRAMES_PER_RAW * H * W)
        frames = buf.reshape(N_FRAMES_PER_RAW, H, W)[:, :hh*BIN, :ww*BIN]
        sums = frames.reshape(N_FRAMES_PER_RAW, hh, BIN, ww, BIN).sum(axis=(2,4), dtype=np.uint32)
        off = idx * N_FRAMES_PER_RAW
        stack[off:off+N_FRAMES_PER_RAW] = sums.astype(np.uint16, copy=False)
        if (idx+1) % 10 == 0 or idx == len(raws)-1:
            print(f"  processed {idx+1}/{len(raws)} files" )

    npy_path = out_dir / "stack_16x16_uint16.npy"
    np.save(npy_path, stack)
    print("Saved:", npy_path)

    tif_path = out_dir / "stack_16x16_uint16.tif"
    # ImageJ-friendly BigTIFF
    tiff.imwrite(tif_path, stack, dtype=np.uint16, imagej=True)
    print("Saved:", tif_path)

if __name__ == "__main__":
    main()
