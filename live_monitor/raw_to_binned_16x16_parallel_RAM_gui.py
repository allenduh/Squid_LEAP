#!/usr/bin/env python3
# RAW -> 16x16 uint16 sums (Parallel to RAM) + save NPY & ImageJ BigTIFF
# - Tkinter app: choose input/output, click Run
# - Parallel per-file processing with ProcessPoolExecutor
# - Results are assembled into one in-RAM NumPy array, then saved once
import os
import re
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tifffile as tiff

# ------- CONFIG (edit as needed) -------
H, W = 608, 1024           # frame size
BIN = 16                   # spatial bin (BIN x BIN)
N_FRAMES_PER_RAW = 10      # frames per .raw
DTYPE_RAW = np.uint8       # raw dtype on disk
# --------------------------------------

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def read_and_bin_one_raw(raw_path: str) -> np.ndarray:
    """
    Read one .raw and return binned array of shape
    (N_FRAMES_PER_RAW, H//BIN, W//BIN), uint16.
    """
    rp = Path(raw_path)
    buf = np.fromfile(rp, dtype=DTYPE_RAW, count=N_FRAMES_PER_RAW * H * W)
    frames = buf.reshape(N_FRAMES_PER_RAW, H, W)
    hh, ww = H // BIN, W // BIN
    frames = frames[:, :hh*BIN, :ww*BIN]
    sums = frames.reshape(N_FRAMES_PER_RAW, hh, BIN, ww, BIN).sum(axis=(2,4), dtype=np.uint32)
    return sums.astype(np.uint16, copy=False)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAW → 16×16 binned (parallel RAM)")
        self.geometry("680x360")
        self.in_dir = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.workers = tk.IntVar(value= (os.cpu_count() or 8))
        self._build()

    def _build(self):
        pad = {"padx":8, "pady":6}
        root = ttk.Frame(self); root.pack(fill="both", expand=True)

        ttk.Label(root, text="RAW input").grid(row=0, column=0, sticky="e")
        ttk.Entry(root, textvariable=self.in_dir, width=56).grid(row=0, column=1, sticky="we")
        ttk.Button(root, text="Browse…", command=self._pick_in).grid(row=0, column=2, **pad)

        ttk.Label(root, text="Output").grid(row=1, column=0, sticky="e")
        ttk.Entry(root, textvariable=self.out_dir, width=56).grid(row=1, column=1, sticky="we")
        ttk.Button(root, text="Browse…", command=self._pick_out).grid(row=1, column=2, **pad)

        s = ttk.LabelFrame(root, text="Settings"); s.grid(row=2, column=0, columnspan=3, sticky="we", **pad)
        ttk.Label(s, text=f"Frame: {H}×{W}  |  Bin: {BIN}×{BIN}  |  Frames/RAW: {N_FRAMES_PER_RAW}").grid(row=0, column=0, sticky="w")
        ttk.Label(s, text="Workers").grid(row=0, column=1, sticky="e", padx=(12,4))
        ttk.Entry(s, textvariable=self.workers, width=6).grid(row=0, column=2, sticky="w")

        p = ttk.LabelFrame(root, text="Progress"); p.grid(row=3, column=0, columnspan=3, sticky="we", **pad)
        self.pb = ttk.Progressbar(p, mode="determinate"); self.pb.grid(row=0, column=0, sticky="we", padx=8, pady=8)
        self.plabel = ttk.Label(p, text="Idle"); self.plabel.grid(row=0, column=1, sticky="w", padx=8)
        p.columnconfigure(0, weight=1)

        btns = ttk.Frame(root); btns.grid(row=4, column=0, columnspan=3, sticky="we", **pad)
        ttk.Button(btns, text="Run", command=self.run).pack(side="left")
        ttk.Button(btns, text="Quit", command=self.destroy).pack(side="right")

        self.log = tk.Text(root, height=10, wrap="none")
        self.log.grid(row=5, column=0, columnspan=3, sticky="nsew", **pad)
        root.columnconfigure(1, weight=1); root.rowconfigure(5, weight=1)

    def _pick_in(self):
        d = filedialog.askdirectory(title="Select RAW input folder")
        if d: self.in_dir.set(d)

    def _pick_out(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d: self.out_dir.set(d)

    def _ui(self, fn, *a, **k): self.after(0, lambda: fn(*a, **k))
    def _log(self, s): self.log.insert("end", s + "\n"); self.log.see("end")
    def _progress(self, k, n): self.pb["maximum"]=max(1,n); self.pb["value"]=k; self.plabel.configure(text=f"{k}/{n}"); self.update_idletasks()

    def run(self):
        in_dir = Path(self.in_dir.get())
        if not in_dir.exists(): messagebox.showerror("Error", "Pick a valid input folder"); return
        out_dir = Path(self.out_dir.get()) if self.out_dir.get() else in_dir / "binned_16x16_parallel"
        out_dir.mkdir(parents=True, exist_ok=True)

        raws = sorted(in_dir.glob("*.raw"), key=lambda p: natural_key(p.name))
        if not raws: messagebox.showerror("Error","No .raw files"); return

        hh, ww = H // BIN, W // BIN
        total_files = len(raws)
        total_frames = total_files * N_FRAMES_PER_RAW
        stack = np.empty((total_frames, hh, ww), dtype=np.uint16)

        self._log(f"Files: {total_files}  |  Frames total: {total_frames}  |  Grid: {hh}x{ww}")
        self._progress(0, total_files)

        def bg():
            processed = 0; errors = 0
            with ProcessPoolExecutor(max_workers=max(1, int(self.workers.get()))) as ex:
                futs = { ex.submit(read_and_bin_one_raw, str(p)): (i, p) for i, p in enumerate(raws) }
                for fu in as_completed(futs):
                    i, rp = futs[fu]
                    try:
                        payload = fu.result()  # (N_FRAMES_PER_RAW, hh, ww) uint16
                        off = i * N_FRAMES_PER_RAW
                        stack[off:off+N_FRAMES_PER_RAW] = payload  # RAM write
                        self._ui(self._log, f"[OK] {rp.name} -> [{off}:{off+N_FRAMES_PER_RAW})")
                    except Exception as e:
                        errors += 1
                        self._ui(self._log, f"[ERROR] {rp.name}: {e}")
                    finally:
                        processed += 1
                        self._ui(self._progress, processed, total_files)

            # Save once at the end
            npy = out_dir / "stack_16x16_uint16.npy"
            np.save(npy, stack)
            tif = out_dir / "stack_16x16_uint16.tif"
            tiff.imwrite(tif, stack, dtype=np.uint16, imagej=True)

            self._ui(self._log, f"[SAVED] {npy}")
            self._ui(self._log, f"[SAVED] {tif}")
            self._ui(messagebox.showinfo, "Done", f"NPY: {npy}\nTIFF: {tif}\nErrors: {errors}")

        import threading; threading.Thread(target=bg, daemon=True).start()

def main(): App().mainloop()

if __name__ == "__main__":
    main()
