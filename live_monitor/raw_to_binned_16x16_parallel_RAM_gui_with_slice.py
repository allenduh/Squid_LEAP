#!/usr/bin/env python3
# RAW -> (A) 16x16 uint16 sums in RAM (parallel)  +  (B) optional unbinned frame slice (uint8)
# Saves both as NPY and ImageJ BigTIFF.

import re
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tifffile as tiff

H, W = 608, 1024
BIN = 16
N_FRAMES_PER_RAW = 10
DTYPE_RAW = np.uint8

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def read_and_bin_one_raw(raw_path: str) -> np.ndarray:
    rp = Path(raw_path)
    buf = np.fromfile(rp, dtype=DTYPE_RAW, count=N_FRAMES_PER_RAW * H * W)
    frames = buf.reshape(N_FRAMES_PER_RAW, H, W)
    hh, ww = H // BIN, W // BIN
    frames = frames[:, :hh*BIN, :ww*BIN]
    sums = frames.reshape(N_FRAMES_PER_RAW, hh, BIN, ww, BIN).sum(axis=(2,4), dtype=np.uint32)
    return sums.astype(np.uint16, copy=False)

def write_tiff_imagej_uint16(path: Path, stack_uint16: np.ndarray):
    tiff.imwrite(path, stack_uint16, dtype=np.uint16, imagej=True)

def write_tiff_imagej_uint8(path: Path, stack_uint8: np.ndarray):
    tiff.imwrite(path, stack_uint8, dtype=np.uint8, imagej=True)

def extract_unbinned_slice(raws, start_frame, end_frame_excl):
    if end_frame_excl <= start_frame:
        raise ValueError("End frame must be greater than start frame.")
    total_frames = len(raws) * N_FRAMES_PER_RAW
    if start_frame < 0 or end_frame_excl > total_frames:
        raise ValueError(f"Requested range [{start_frame}, {end_frame_excl}) outside [0, {total_frames})")
    T = end_frame_excl - start_frame
    out = np.empty((T, H, W), dtype=np.uint8)
    file_start = start_frame // N_FRAMES_PER_RAW
    off_start  = start_frame %  N_FRAMES_PER_RAW
    file_last  = (end_frame_excl - 1) // N_FRAMES_PER_RAW
    off_last   = (end_frame_excl - 1) %  N_FRAMES_PER_RAW
    dst = 0
    for fi in range(file_start, file_last + 1):
        rp = raws[fi]
        buf = np.fromfile(rp, dtype=DTYPE_RAW, count=N_FRAMES_PER_RAW * H * W)
        frames = buf.reshape(N_FRAMES_PER_RAW, H, W)
        s = off_start if fi == file_start else 0
        e = (off_last + 1) if fi == file_last else N_FRAMES_PER_RAW
        n = e - s
        out[dst:dst+n] = frames[s:e]
        dst += n
    return out

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAW → 16×16 binned (parallel RAM) + optional unbinned slice")
        self.geometry("760x460")
        self.in_dir = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.workers = tk.IntVar(value= max(1, (os_cpu_count() or 8) - 1))
        self.slice_enable = tk.BooleanVar(value=False)
        self.slice_start = tk.IntVar(value=24000)
        self.slice_end   = tk.IntVar(value=24200)
        self._build()

    def _build(self):
        pad = {"padx":8, "pady":6}
        root = ttk.Frame(self); root.pack(fill="both", expand=True)

        ttk.Label(root, text="RAW input").grid(row=0, column=0, sticky="e")
        ttk.Entry(root, textvariable=self.in_dir, width=60).grid(row=0, column=1, sticky="we")
        ttk.Button(root, text="Browse…", command=self._pick_in).grid(row=0, column=2, **pad)

        ttk.Label(root, text="Output").grid(row=1, column=0, sticky="e")
        ttk.Entry(root, textvariable=self.out_dir, width=60).grid(row=1, column=1, sticky="we")
        ttk.Button(root, text="Browse…", command=self._pick_out).grid(row=1, column=2, **pad)

        s = ttk.LabelFrame(root, text="Settings"); s.grid(row=2, column=0, columnspan=3, sticky="we", **pad)
        ttk.Label(s, text=f"Frame: {H}×{W}   Bin: {BIN}×{BIN}   Frames/RAW: {N_FRAMES_PER_RAW}").grid(row=0, column=0, sticky="w")
        ttk.Label(s, text="Workers").grid(row=0, column=1, sticky="e", padx=(16,4))
        ttk.Entry(s, textvariable=self.workers, width=6).grid(row=0, column=2, sticky="w")

        slice_box = ttk.LabelFrame(root, text="Optional unbinned export (uint8)")
        slice_box.grid(row=3, column=0, columnspan=3, sticky="we", **pad)
        ttk.Checkbutton(slice_box, text="Export unbinned frames", variable=self.slice_enable).grid(row=0, column=0, sticky="w")
        ttk.Label(slice_box, text="Start (inclusive)").grid(row=0, column=1, sticky="e")
        ttk.Entry(slice_box, textvariable=self.slice_start, width=10).grid(row=0, column=2, sticky="w")
        ttk.Label(slice_box, text="End (exclusive)").grid(row=0, column=3, sticky="e")
        ttk.Entry(slice_box, textvariable=self.slice_end, width=10).grid(row=0, column=4, sticky="w")

        p = ttk.LabelFrame(root, text="Progress"); p.grid(row=4, column=0, columnspan=3, sticky="we", **pad)
        self.pb = ttk.Progressbar(p, mode="determinate"); self.pb.grid(row=0, column=0, sticky="we", padx=8, pady=8)
        self.plabel = ttk.Label(p, text="Idle"); self.plabel.grid(row=0, column=1, sticky="w", padx=8)
        p.columnconfigure(0, weight=1)

        btns = ttk.Frame(root); btns.grid(row=5, column=0, columnspan=3, sticky="we", **pad)
        ttk.Button(btns, text="Run", command=self.run).pack(side="left")
        ttk.Button(btns, text="Quit", command=self.destroy).pack(side="right")

        self.log = tk.Text(root, height=12, wrap="none")
        self.log.grid(row=6, column=0, columnspan=3, sticky="nsew", **pad)
        root.columnconfigure(1, weight=1); root.rowconfigure(6, weight=1)

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

        self._log(f"Files: {total_files}  |  Frames total: {total_frames}  |  Grid: {hh}x{ww}")
        self._progress(0, total_files)

        stack16 = np.empty((total_frames, hh, ww), dtype=np.uint16)

        def bg():
            processed = 0; errors = 0
            with ProcessPoolExecutor(max_workers=max(1, int(self.workers.get()))) as ex:
                futs = { ex.submit(read_and_bin_one_raw, str(p)): (i, p) for i, p in enumerate(raws) }
                for fu in as_completed(futs):
                    i, rp = futs[fu]
                    try:
                        payload = fu.result()
                        off = i * N_FRAMES_PER_RAW
                        stack16[off:off+N_FRAMES_PER_RAW] = payload
                        self._ui(self._log, f"[OK] {rp.name} -> [{off}:{off+N_FRAMES_PER_RAW})")
                    except Exception as e:
                        errors += 1
                        self._ui(self._log, f"[ERROR] {rp.name}: {e}")
                    finally:
                        processed += 1
                        self._ui(self._progress, processed, total_files)

            # Save binned
            npy16 = out_dir / "stack_16x16_uint16.npy"
            np.save(npy16, stack16)
            tif16 = out_dir / "stack_16x16_uint16.tif"
            write_tiff_imagej_uint16(tif16, stack16)
            self._ui(self._log, f"[SAVED] {npy16}")
            self._ui(self._log, f"[SAVED] {tif16}")

            # Optional unbinned slice
            if self.slice_enable.get():
                s = int(self.slice_start.get())
                e = int(self.slice_end.get())
                try:
                    raw_slice = extract_unbinned_slice(raws, s, e)
                    npy8 = out_dir / f"raw_slice_{s}_{e}_uint8.npy"
                    np.save(npy8, raw_slice)
                    tif8 = out_dir / f"raw_slice_{s}_{e}_uint8.tif"
                    write_tiff_imagej_uint8(tif8, raw_slice)
                    self._ui(self._log, f"[SAVED] {npy8}")
                    self._ui(self._log, f"[SAVED] {tif8}")
                except Exception as ex:
                    self._ui(self._log, f"[SLICE ERROR] {ex}")

            self._ui(messagebox.showinfo, "Done",
                     "Binned stack + optional unbinned slice saved.")

        import threading; threading.Thread(target=bg, daemon=True).start()

def os_cpu_count():
    try:
        import os
        return os.cpu_count()
    except Exception:
        return 8

def main(): App().mainloop()
if __name__ == "__main__":
    main()
