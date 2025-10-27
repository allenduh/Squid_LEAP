#!/usr/bin/env python3
"""
RAW → 8×8 binned stack (float32) with **parallel per-file processing** and a GUI progress bar.
- Picks input/output folders via dialogs (no terminal typing).
- Uses all CPU cores (ProcessPoolExecutor) to read+bin each RAW concurrently.
- Shows progress: "processed K / total" in a ttk.Progressbar + label.
- Produces: stack_8x8.tif (float32, ImageJ-friendly) + stack_8x8.npz.
- Binning is a true mean in float32 (no rounding), preserving √64 shot-noise scaling.

Assumptions
- Each .raw file = contiguous uint8 frames, shape (N,H,W). N is inferred from file size.
- Default sensor: 608×1024 (editable in GUI). Bin size fixed at 8×8.
"""

from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import tifffile as tiff
except ImportError as e:
    raise SystemExit("Please 'pip install tifffile'") from e

BIN = 8

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

def bin8x_batch(mm: np.memmap, H: int, W: int, bin_size: int = BIN) -> np.ndarray:
    """mm: (N,H,W) uint8 → (N, H//b, W//b) float32 (mean, no rounding)"""
    b = int(bin_size)
    assert H % b == 0 and W % b == 0, "H and W must be divisible by bin size"
    N = mm.shape[0]
    hh, ww = H // b, W // b
    return mm.reshape(N, hh, b, ww, b).mean(axis=(2,4), dtype=np.float32)

def _worker(idx: int, raw_path: str, H: int, W: int) -> tuple[int, np.ndarray, int, str]:
    """Runs in a separate process. Returns (index, binned_float32_stack, N, file_path)."""
    rp = Path(raw_path)
    N = infer_frames_per_file(rp, H, W)
    mm = np.memmap(rp, mode="r", dtype=np.uint8, shape=(N, H, W))
    try:
        binned = bin8x_batch(mm, H, W, BIN)   # (N, H//8, W//8) float32
    finally:
        del mm
    return idx, binned, N, str(rp)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAW → 8×8 binned stack (TIFF+NPZ) — Parallel")
        self.geometry("560x320")
        self.in_dir = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.height = tk.IntVar(value=608)
        self.width  = tk.IntVar(value=1024)
        self.workers = tk.IntVar(value=os.cpu_count() or 4)
        self._build()

    def _build(self):
        pad = {"padx": 6, "pady": 4}
        f = ttk.Frame(self); f.pack(fill="both", expand=True, **pad)

        ttk.Label(f, text="RAW input folder").grid(row=0, column=0, sticky="e")
        ttk.Entry(f, textvariable=self.in_dir, width=46).grid(row=0, column=1, sticky="we")
        ttk.Button(f, text="Browse…", command=self._pick_in).grid(row=0, column=2)

        ttk.Label(f, text="Output folder").grid(row=1, column=0, sticky="e")
        ttk.Entry(f, textvariable=self.out_dir, width=46).grid(row=1, column=1, sticky="we")
        ttk.Button(f, text="Browse…", command=self._pick_out).grid(row=1, column=2)

        g = ttk.LabelFrame(f, text="Settings")
        g.grid(row=2, column=0, columnspan=3, sticky="we", **pad)
        ttk.Label(g, text="Height").grid(row=0, column=0, sticky="e"); ttk.Entry(g, textvariable=self.height, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(g, text="Width").grid(row=0, column=2, sticky="e");  ttk.Entry(g, textvariable=self.width,  width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(g, text="Bin").grid(row=0, column=4, sticky="e");    ttk.Label(g, text="8×8 (fixed)").grid(row=0, column=5, sticky="w")
        ttk.Label(g, text="Workers").grid(row=0, column=6, sticky="e");ttk.Entry(g, textvariable=self.workers, width=6).grid(row=0, column=7, sticky="w")

        # Progress UI
        pfrm = ttk.LabelFrame(f, text="Progress")
        pfrm.grid(row=3, column=0, columnspan=3, sticky="we", **pad)
        self.pb = ttk.Progressbar(pfrm, orient="horizontal", mode="determinate", length=420)
        self.pb.grid(row=0, column=0, sticky="we", **pad)
        self.plabel = ttk.Label(pfrm, text="Waiting…")
        self.plabel.grid(row=0, column=1, sticky="e")

        ttk.Button(f, text="Run", command=self._run).grid(row=4, column=0, sticky="w", **pad)
        ttk.Button(f, text="Quit", command=self.destroy).grid(row=4, column=2, sticky="e", **pad)

        self.log = tk.Text(f, height=8)
        self.log.grid(row=5, column=0, columnspan=3, sticky="nsew", **pad)
        f.columnconfigure(1, weight=1); f.rowconfigure(5, weight=1)

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
            if (H % BIN) or (W % BIN):
                raise ValueError(f"H={H}, W={W} must be divisible by {BIN}")

            raws = sorted(in_dir.glob("*.raw"), key=lambda p: natural_key(p.name))
            if not raws: raise FileNotFoundError("No *.raw in the input folder")
            total_files = len(raws)
            self.pb.configure(maximum=total_files, value=0)
            self._set_plabel(0, total_files)
            self._log(f"Found {total_files} raw files. Using {self.workers.get()} workers…")

            # Run in background to keep GUI responsive
            import threading
            threading.Thread(target=self._process_parallel, args=(raws, out_dir, H, W), daemon=True).start()

        except Exception as e:
            self._log(f"[ERROR] {e}")
            messagebox.showerror("Error", str(e))

    def _process_parallel(self, raws, out_dir: Path, H: int, W: int):
        try:
            hh, ww = H // BIN, W // BIN
            # Submit workers
            futures = []
            results = {}
            per_file_info = []
            processed = 0
            total_frames = 0

            with ProcessPoolExecutor(max_workers=int(self.workers.get())) as ex:
                for idx, rp in enumerate(raws):
                    futures.append(ex.submit(_worker, idx, str(rp), H, W))

                for fu in as_completed(futures):
                    idx, binned, N, fpath = fu.result()
                    results[idx] = binned            # (N, hh, ww) float32
                    per_file_info.append(dict(file=fpath, frames=int(N)))
                    processed += 1; total_frames += int(N)
                    self._progress_tick(processed, len(raws))

            # Concatenate in original order
            stack = np.concatenate([results[i] for i in range(len(raws))], axis=0)  # (T, hh, ww) float32

            # Save outputs
            tiff_path = out_dir / "stack_8x8.tif"
            npz_path  = out_dir / "stack_8x8.npz"
            tiff.imwrite(str(tiff_path), stack, imagej=True)
            meta = dict(height=H, width=W, bin=BIN, out_h=hh, out_w=ww,
                        total_frames=int(total_frames), files=per_file_info)
            np.savez_compressed(npz_path, data=stack, meta=meta)

            self._log(f"Done. Processed {processed}/{len(raws)} files, T={total_frames} frames.")
            self._log(f"TIFF → {tiff_path}\nNPZ  → {npz_path}\nStack shape: {stack.shape} (float32)")
            self._notify_done(tiff_path, npz_path, stack.shape)

        except Exception as e:
            self._log(f"[ERROR] {e}")
            messagebox.showerror("Error", str(e))

    def _progress_tick(self, k, n):
        self.pb['value'] = k
        self._set_plabel(k, n)
        self.update_idletasks()

    def _set_plabel(self, k, n):
        self.plabel.configure(text=f"{k} / {n} raw files")

    def _notify_done(self, tiff_p, npz_p, shape):
        try:
            messagebox.showinfo("Done", f"TIFF: {tiff_p}\nNPZ: {npz_p}\nShape: {shape}")
        except Exception:
            pass

    def _log(self, msg: str):
        self.log.insert("end", msg+"\n"); self.log.see("end"); self.update_idletasks()

def main():
    App().mainloop()

if __name__ == "__main__":
    main()
