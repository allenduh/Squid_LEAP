#!/usr/bin/env python3
"""
RAW -> 16x16 uint16 sums (NPY memmap + ImageJ BigTIFF)
- Folder picker UI (Tk)
- Parallel per-file binning via ProcessPoolExecutor
- Safe natural sort; validates raw size per file
- Memory-mapped output written incrementally and persisted automatically
"""

from __future__ import annotations
import os
import json
import re
import sys
from pathlib import Path
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tifffile as tiff
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# ------------------ CONFIG ------------------
BIN = 16                 # spatial bin size (16x16 blocks)
N_FRAMES_PER_RAW = 10    # frames per .raw file
DTYPE_RAW = np.uint8     # camera data type on disk
# -------------------------------------------


def natural_key(s: str):
    """Key function for natural sorting (2 < 10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


def read_and_bin_one_raw(raw_path: str, H: int, W: int, hh: int, ww: int) -> np.ndarray:
    """
    Worker body (pure function so it's safe for process pool):
      - Reads a single .raw file of uint8, expected shape (N_FRAMES_PER_RAW, H, W)
      - Crops to multiples of BIN
      - Sums 16x16 blocks (uint32 accumulation), then casts to uint16
      - Returns array shape: (N_FRAMES_PER_RAW, hh, ww), dtype=uint16
    """
    raw_path = Path(raw_path)

    # Expected count & validation
    expected_count = N_FRAMES_PER_RAW * H * W
    buf = np.fromfile(raw_path, dtype=DTYPE_RAW, count=expected_count)

    if buf.size != expected_count:
        raise ValueError(f"{raw_path.name}: size mismatch (got {buf.size}, expected {expected_count} elements)")

    frames = buf.reshape(N_FRAMES_PER_RAW, H, W)

    # Crop to multiples of BIN
    Hb = hh * BIN
    Wb = ww * BIN
    frames = frames[:, :Hb, :Wb]

    # Vectorized 16x16 sum with safe dtype (avoid overflow), then cast
    sums = frames.reshape(N_FRAMES_PER_RAW, hh, BIN, ww, BIN).sum(axis=(2, 4), dtype=np.uint32)
    return sums.astype(np.uint16, copy=False)  # 16*16*255 = 65280 fits in uint16


def write_tiff_imagej_safe(tiff_out: Path, frames_u16: np.ndarray) -> None:
    """
    Write (T, H, W) uint16 stack as BigTIFF, ImageJ-friendly.
    Uses one strip per image for compatibility.
    """
    T, H, W = frames_u16.shape
    with tiff.TiffWriter(str(tiff_out), bigtiff=True) as tw:
        for k in range(T):
            tw.write(
                frames_u16[k],
                dtype=np.uint16,
                photometric='minisblack',
                contiguous=True,
                rowsperstrip=H,
                compression=None,
            )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAW → 16×16 uint16 sums (BigTIFF + NPY)")
        self.geometry("720x420")

        # UI state
        self.in_dir = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.height = tk.IntVar(value=608)
        self.width = tk.IntVar(value=1024)
        self.workers = tk.IntVar(value=os.cpu_count() or 8)

        self._build()

    # ---------------- UI ----------------

    def _build(self):
        pad = {"padx": 8, "pady": 6}
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="RAW input").grid(row=0, column=0, sticky="e")
        ttk.Entry(root, textvariable=self.in_dir, width=58).grid(row=0, column=1, sticky="we")
        ttk.Button(root, text="Browse…", command=self._pick_in).grid(row=0, column=2, **pad)

        ttk.Label(root, text="Output").grid(row=1, column=0, sticky="e")
        ttk.Entry(root, textvariable=self.out_dir, width=58).grid(row=1, column=1, sticky="we")
        ttk.Button(root, text="Browse…", command=self._pick_out).grid(row=1, column=2, **pad)

        box = ttk.LabelFrame(root, text="Settings")
        box.grid(row=2, column=0, columnspan=3, sticky="we", **pad)

        ttk.Label(box, text="Height (H)").grid(row=0, column=0, sticky="e")
        ttk.Entry(box, textvariable=self.height, width=8).grid(row=0, column=1, sticky="w")

        ttk.Label(box, text="Width (W)").grid(row=0, column=2, sticky="e")
        ttk.Entry(box, textvariable=self.width, width=8).grid(row=0, column=3, sticky="w")

        ttk.Label(box, text="Bin").grid(row=0, column=4, sticky="e")
        ttk.Label(box, text=f"{BIN} × {BIN} (uint16 sums)").grid(row=0, column=5, sticky="w")

        ttk.Label(box, text="Workers").grid(row=0, column=6, sticky="e")
        ttk.Entry(box, textvariable=self.workers, width=8).grid(row=0, column=7, sticky="w")

        pbox = ttk.LabelFrame(root, text="Progress")
        pbox.grid(row=3, column=0, columnspan=3, sticky="we", **pad)
        self.pb = ttk.Progressbar(pbox, mode="determinate")
        self.pb.grid(row=0, column=0, sticky="we", padx=8, pady=8)
        self.plabel = ttk.Label(pbox, text="Waiting…")
        self.plabel.grid(row=0, column=1, sticky="w", padx=8)
        pbox.columnconfigure(0, weight=1)

        btns = ttk.Frame(root)
        btns.grid(row=4, column=0, columnspan=3, sticky="we", **pad)
        ttk.Button(btns, text="Run", command=self._run).pack(side="left")
        ttk.Button(btns, text="Quit", command=self.destroy).pack(side="right")

        self.log = tk.Text(root, height=12, wrap="none")
        self.log.grid(row=5, column=0, columnspan=3, sticky="nsew", **pad)

        root.columnconfigure(1, weight=1)
        root.rowconfigure(5, weight=1)

    def _pick_in(self):
        d = filedialog.askdirectory(title="Select RAW input folder")
        if d:
            self.in_dir.set(d)

    def _pick_out(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.out_dir.set(d)

    def _ui(self, fn, *args, **kwargs):
        self.after(0, lambda: fn(*args, **kwargs))

    def _log(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")

    def _progress(self, k: int, n: int):
        self.pb["maximum"] = max(1, n)
        self.pb["value"] = k
        self.plabel.configure(text=f"{k} / {n}")
        self.update_idletasks()

    # --------------- RUN ----------------

    def _run(self):
        try:
            in_dir = Path(self.in_dir.get())
            out_dir = Path(self.out_dir.get()) if self.out_dir.get() else in_dir / "bmp_export_16x16"
            if not in_dir.exists():
                raise FileNotFoundError("Pick a valid RAW input folder")
            out_dir.mkdir(parents=True, exist_ok=True)

            H = int(self.height.get()); W = int(self.width.get())
            if (H % BIN) or (W % BIN):
                raise ValueError(f"H={H}, W={W} must be divisible by BIN={BIN}")

            raws = sorted(in_dir.glob("*.raw"), key=lambda p: natural_key(p.name))
            if not raws:
                raise FileNotFoundError("No *.raw files found in the chosen folder")

            hh, ww = H // BIN, W // BIN
            total_files = len(raws)
            total_frames = N_FRAMES_PER_RAW * total_files

            # Memory-mapped output stack (persists on disk automatically)
            npy_path = out_dir / "stack_16x16_uint16.npy"
            stack = np.lib.format.open_memmap(
                npy_path, mode="w+", dtype=np.uint16, shape=(total_frames, hh, ww)
            )

            self._log(f"Input: {in_dir}")
            self._log(f"Output: {out_dir}")
            self._log(f"Files: {total_files}, frames/file={N_FRAMES_PER_RAW}, total frames={total_frames}")
            self._log(f"Block grid: {hh} × {ww}; value range 0..{255*BIN*BIN}")

            self._progress(0, total_files)

            def bg():
                processed = 0
                errors = 0
                with ProcessPoolExecutor(max_workers=int(self.workers.get())) as ex:
                    futs = {
                        ex.submit(read_and_bin_one_raw, str(p), H, W, hh, ww): (idx, p)
                        for idx, p in enumerate(raws)
                    }
                    for fu in as_completed(futs):
                        idx, p = futs[fu]
                        try:
                            payload = fu.result()  # (N_FRAMES_PER_RAW, hh, ww) uint16
                            off = idx * N_FRAMES_PER_RAW
                            stack[off:off + N_FRAMES_PER_RAW] = payload  # memmap slice write
                            self._ui(self._log, f"[OK] {p.name} -> [{off}:{off+N_FRAMES_PER_RAW})")
                        except Exception as e:
                            errors += 1
                            self._ui(self._log, f"[ERROR] {p.name}: {e}")
                        finally:
                            processed += 1
                            self._ui(self._progress, processed, total_files)

                # After pool completes, write BigTIFF (from the memmap on disk)
                tiff_path = out_dir / "stack_16x16_uint16.tif"
                try:
                    write_tiff_imagej_safe(tiff_path, stack)
                    self._ui(self._log, f"[TIFF] Wrote {tiff_path}")
                except Exception as e:
                    self._ui(self._log, f"[TIFF ERROR] {e}")

                # Metadata
                meta = dict(
                    height=H, width=W, bin=BIN, out_h=hh, out_w=ww,
                    total_frames=int(total_frames),
                    value_range=[0, int(255*BIN*BIN)],
                    frames_per_raw=N_FRAMES_PER_RAW,
                    dtype_raw=str(DTYPE_RAW),
                    stack_npy=str(npy_path.name),
                    tiff=str(tiff_path.name),
                    errors=int(errors),
                )
                (out_dir / "stack_16x16_meta.json").write_text(json.dumps(meta, indent=2))

                self._ui(messagebox.showinfo, "Done",
                         f"TIFF: {tiff_path}\nNPY: {npy_path}\nShape: {(total_frames, hh, ww)}\nErrors: {errors}")

            threading.Thread(target=bg, daemon=True).start()

        except Exception as e:
            self._log(f"[ERROR] {e}")
            messagebox.showerror("Error", str(e))


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
