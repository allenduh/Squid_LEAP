#!/usr/bin/env python3
# Visualize per-pixel temporal STD of a 3D stack (T,H,W).
# - Opens a .npy (uint8 or uint16, shape (T,H,W)) or an ImageJ-compatible .tif
# - Computes std over time per pixel
# - Displays a heatmap and optionally saves outputs next to the input
#
# Suggested usage:
#   python visualize_std_map.py

import numpy as np
import tifffile as tiff
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from pathlib import Path

def load_stack(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path, mmap_mode=None)
        if arr.ndim != 3:
            raise ValueError(f"Expected (T,H,W) array, got shape {arr.shape}")
        return arr
    elif path.suffix.lower() in {".tif", ".tiff"}:
        arr = tiff.imread(str(path))
        if arr.ndim != 3:
            raise ValueError(f"Expected (T,H,W) TIFF, got shape {arr.shape}")
        return arr
    else:
        raise ValueError("Pick a .npy or .tif/.tiff file")

def main():
    root = tk.Tk(); root.withdraw()
    fpath = filedialog.askopenfilename(
        title="Select NPY or TIFF stack (T,H,W)",
        filetypes=[("Stacks", "*.npy *.tif *.tiff"), ("All files", "*.*")]
    )
    root.update(); root.destroy()
    if not fpath:
        print("No file chosen."); return
    p = Path(fpath)
    print(f"Loading: {p}")
    stack = load_stack(p)  # (T,H,W), uint8/uint16
    T, H, W = stack.shape
    print(f"Shape: (T,H,W)=({T},{H},{W}), dtype={stack.dtype}")

    # Compute per-pixel STD over time
    std_map = stack.astype(np.float32).std(axis=0)  # (H,W), float32
    vmax = np.percentile(std_map, 99) if std_map.size > 0 else 1.0

    # Show heatmap
    plt.figure(figsize=(6,5))
    im = plt.imshow(std_map, cmap="magma", vmin=0, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04, label="STD (a.u.)")
    plt.title("Per-pixel STD over time")
    plt.tight_layout()
    plt.show(block=False)

    # Save outputs next to input
    out_base = p.with_suffix("")
    png_out = out_base.parent / f"{out_base.name}_std.png"
    tif_out = out_base.parent / f"{out_base.name}_std.tif"

    plt.savefig(png_out, dpi=200)
    print(f"Saved PNG: {png_out}")

    # Save the std map as 32-bit float TIFF for later processing
    tiff.imwrite(tif_out, std_map.astype(np.float32))
    print(f"Saved TIFF: {tif_out}")

    # Optional: quick mask of top percentile (e.g., top 1% std)
    pct = 99.0
    thr = np.percentile(std_map, pct)
    mask = (std_map >= thr).astype(np.uint8)
    mask_out = out_base.parent / f"{out_base.name}_std_top{int(pct)}pct_mask.tif"
    tiff.imwrite(mask_out, mask, dtype=np.uint8)
    print(f"Saved top-{pct}% mask: {mask_out}")

if __name__ == "__main__":
    main()
