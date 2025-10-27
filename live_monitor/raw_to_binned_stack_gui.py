#!/usr/bin/env python3
"""
RAW → binned stack (8×8) → TIFF + NPZ (float32), with simple folder pickers.
- Input RAW: uint8 frames, contiguous (N,H,W) per file
- Assumes sensor 608×1024 by default (change in GUI)
- Infers N per file from file size; preserves order
- Binning is a true mean in float32 (no rounding), so shot noise scales ∝ sqrt(64)

Output:
  - <out_dir>/stack_8x8.tif   (multipage float32; ImageJ-compatible)
  - <out_dir>/stack_8x8.npz   (arr float32 [T,76,128] + metadata)
"""

import os, sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

try:
    import tifffile as tiff
except ImportError as e:
    raise SystemExit("Please 'pip install tifffile'") from e

# ---------------- core ----------------
def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def infer_frames_per_file(raw_path: Path, H: int, W: int) -> int:
    fb = int(H) * int(W)
    if fb <= 0: raise ValueError("H and W must be positive")
    sz = raw_path.stat().st_size
    if sz % fb != 0:
        raise ValueError(f"File size not divisible by H*W: {raw_path.name} ({sz} bytes)")
    return sz // fb

def bin8x_batch(mm: np.memmap, H: int, W: int, bin_size: int = 8) -> np.ndarray:
    """mm: (N,H,W) uint8 → (N, H//b, W//b) float32 (mean, no rounding)"""
    b = int(bin_size)
    assert H % b == 0 and W % b == 0, "H and W must be divisible by bin size"
    N = mm.shape[0]
    hh, ww = H // b, W // b
    # reshape (N, hh, b, ww, b) and mean over (2,4)
    return mm.reshape(N, hh, b, ww, b).mean(axis=(2,4), dtype=np.float32)

def convert_folder(in_dir: Path, out_dir: Path, H: int, W: int, bin_size: int = 8):
    raws = sorted(in_dir.glob("*.raw"), key=lambda p: natural_key(p.name))
    if not raws:
        raise FileNotFoundError(f"No *.raw under {in_dir}")
    if (H % bin_size) or (W % bin_size):
        raise ValueError(f"H={H}, W={W} must be divisible by bin={bin_size}")
    hh, ww = H // bin_size, W // bin_size

    pieces = []
    per_file_info = []
    total_frames = 0
    for rp in raws:
        N = infer_frames_per_file(rp, H, W)
        mm = np.memmap(rp, mode="r", dtype=np.uint8, shape=(N, H, W))
        binned = bin8x_batch(mm, H, W, bin_size)  # (N, hh, ww) float32
        pieces.append(binned)  # keep float32
        per_file_info.append(dict(file=str(rp), frames=int(N)))
        total_frames += N
        del mm

    stack = np.concatenate(pieces, axis=0)  # (T, hh, ww) float32
    del pieces

    out_dir.mkdir(parents=True, exist_ok=True)
    tiff_path = out_dir / "stack_8x8.tif"
    npz_path  = out_dir / "stack_8x8.npz"

    # Write float32 multipage TIFF (ImageJ can open float32 stacks)
    # imagej=True adds metadata ImageJ likes (optional but handy)
    tiff.imwrite(str(tiff_path), stack, imagej=True)

    # Save metadata + data for Python
    meta = dict(
        height=H, width=W, bin=bin_size,
        out_h=hh, out_w=ww,
        total_frames=int(total_frames),
        files=per_file_info,
    )
    np.savez_compressed(npz_path, data=stack, meta=meta)

    return tiff_path, npz_path, stack.shape

# ---------------- GUI ----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAW → 8×8 binned stack (TIFF + NPZ)")
        self.geometry("520x260")
        self.in_dir = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.height = tk.IntVar(value=608)
        self.width  = tk.IntVar(value=1024)
        self._build()

    def _build(self):
        pad = {"padx": 6, "pady": 4}
        f = ttk.Frame(self); f.pack(fill="both", expand=True, **pad)

        ttk.Label(f, text="RAW input folder").grid(row=0, column=0, sticky="e")
        ttk.Entry(f, textvariable=self.in_dir, width=42).grid(row=0, column=1, sticky="we")
        ttk.Button(f, text="Browse…", command=self._pick_in).grid(row=0, column=2)

        ttk.Label(f, text="Output folder").grid(row=1, column=0, sticky="e")
        ttk.Entry(f, textvariable=self.out_dir, width=42).grid(row=1, column=1, sticky="we")
        ttk.Button(f, text="Browse…", command=self._pick_out).grid(row=1, column=2)

        g = ttk.LabelFrame(f, text="Sensor geometry")
        g.grid(row=2, column=0, columnspan=3, sticky="we", **pad)
        ttk.Label(g, text="Height").grid(row=0, column=0, sticky="e"); ttk.Entry(g, textvariable=self.height, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(g, text="Width").grid(row=0, column=2, sticky="e");  ttk.Entry(g, textvariable=self.width,  width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(g, text="Bin").grid(row=0, column=4, sticky="e");    ttk.Label(g, text="8×8 (fixed)").grid(row=0, column=5, sticky="w")

        ttk.Button(f, text="Run", command=self._run).grid(row=3, column=0, sticky="w", **pad)
        ttk.Button(f, text="Quit", command=self.destroy).grid(row=3, column=2, sticky="e", **pad)

        self.log = tk.Text(f, height=8)
        self.log.grid(row=4, column=0, columnspan=3, sticky="nsew", **pad)
        f.columnconfigure(1, weight=1); f.rowconfigure(4, weight=1)

    def _pick_in(self):
        d = filedialog.askdirectory(title="Select RAW input folder")
        if d: self.in_dir.set(d)

    def _pick_out(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d: self.out_dir.set(d)

    def _run(self):
        try:
            in_dir = Path(self.in_dir.get())
            out_dir = Path(self.out_dir.get())
            if not in_dir.exists(): raise FileNotFoundError("Pick a valid RAW input folder.")
            if not out_dir.exists(): out_dir.mkdir(parents=True, exist_ok=True)

            H, W = int(self.height.get()), int(self.width.get())
            self._log(f"Reading: {in_dir}\nOutput : {out_dir}\nDims   : H={H}, W={W}\nBin    : 8×8")
            tiff_p, npz_p, shp = convert_folder(in_dir, out_dir, H, W, bin_size=8)
            self._log(f"Done. Stack shape {shp} (float32)\nTIFF → {tiff_p}\nNPZ  → {npz_p}")
            messagebox.showinfo("Done", f"TIFF: {tiff_p}\nNPZ: {npz_p}\nShape: {shp}")
        except Exception as e:
            self._log(f"[ERROR] {e}")
            messagebox.showerror("Error", str(e))

    def _log(self, msg: str):
        self.log.insert("end", msg+"\n"); self.log.see("end"); self.update_idletasks()

def main():
    App().mainloop()

if __name__ == "__main__":
    main()
