#!/usr/bin/env python3
"""
RAW → 16×16 binned *SUMS* (uint16, max=255*16*16=65280) → BigTIFF + NPY
- GUI pickers (no terminal typing)
- Uses all CPU cores (one RAW per process)
- Progress “K / total” + per-file log
- Robust: skips empty / truncated RAWs (uses floor(bytes/(H*W)))
- Outputs:
  - stack_16x16_uint16.tif  (BigTIFF, uint16, compressed)
  - stack_16x16_uint16.npy  (uint16 memmap on disk)
  - stack_16x16_meta.json   (small metadata)
"""

from __future__ import annotations
import os, json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import tifffile as tiff
except ImportError as e:
    raise SystemExit("Please 'pip install tifffile'") from e

BIN = 16  # changed from 8 → 16

def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def frames_info(raw_path: Path, H: int, W: int) -> tuple[int, int]:
    fb = int(H) * int(W)
    sz = raw_path.stat().st_size
    if fb <= 0 or sz <= 0: return 0, 0
    return sz // fb, sz % fb

def sum_bin_batch(mm: np.memmap, H: int, W: int, b: int = BIN) -> np.ndarray:
    """mm: (N,H,W) uint8 → (N, H//b, W//b) uint16 (sum over b×b, lossless)."""
    assert H % b == 0 and W % b == 0
    N = mm.shape[0]; hh, ww = H // b, W // b
    # sum in uint32 to avoid overflow, then cast down (max 255*b*b <= 65280 when b=16)
    sums = mm.reshape(N, hh, b, ww, b).sum(axis=(2, 4), dtype=np.uint32)
    return sums.astype(np.uint16, copy=False)

def _worker(idx: int, raw_path: str, H: int, W: int) -> tuple[int, str, object, int]:
    """Returns (idx, status, payload, frames). payload is uint16 array when ok, else error string."""
    try:
        rp = Path(raw_path)
        N, rem = frames_info(rp, H, W)
        if N <= 0:
            return idx, "skip", f"{rp.name}: empty or too small (bytes={rp.stat().st_size})", 0
        mm = np.memmap(rp, mode="r", dtype=np.uint8, shape=(N, H, W))
        try:
            block = sum_bin_batch(mm, H, W, BIN)  # (N, H//BIN, W//BIN) uint16
        finally:
            del mm
        return idx, "ok", block, int(N)
    except Exception as e:
        return idx, "error", f"{Path(raw_path).name}: {e}", 0

# ---------------- GUI app ----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAW → 16×16 uint16 sums (BigTIFF + NPY)")
        self.geometry("640x380")
        self.in_dir = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.height = tk.IntVar(value=608)
        self.width  = tk.IntVar(value=1024)
        self.workers = tk.IntVar(value=os.cpu_count() or 4)
        self._build()

    def _build(self):
        pad = {"padx":6, "pady":4}
        f = ttk.Frame(self); f.pack(fill="both", expand=True, **pad)

        ttk.Label(f, text="RAW input folder").grid(row=0,column=0,sticky="e")
        ttk.Entry(f, textvariable=self.in_dir, width=52).grid(row=0,column=1,sticky="we")
        ttk.Button(f, text="Browse…", command=self._pick_in).grid(row=0,column=2)

        ttk.Label(f, text="Output folder").grid(row=1,column=0,sticky="e")
        ttk.Entry(f, textvariable=self.out_dir, width=52).grid(row=1,column=1,sticky="we")
        ttk.Button(f, text="Browse…", command=self._pick_out).grid(row=1,column=2)

        g = ttk.LabelFrame(f, text="Settings"); g.grid(row=2,column=0,columnspan=3,sticky="we",**pad)
        ttk.Label(g, text="H").grid(row=0,column=0,sticky="e"); ttk.Entry(g,textvariable=self.height,width=7).grid(row=0,column=1,sticky="w")
        ttk.Label(g, text="W").grid(row=0,column=2,sticky="e"); ttk.Entry(g,textvariable=self.width,width=7).grid(row=0,column=3,sticky="w")
        ttk.Label(g, text="Bin").grid(row=0,column=4,sticky="e"); ttk.Label(g,text="16×16 (uint16 sums)").grid(row=0,column=5,sticky="w")
        ttk.Label(g, text="Workers").grid(row=0,column=6,sticky="e"); ttk.Entry(g,textvariable=self.workers,width=6).grid(row=0,column=7,sticky="w")

        p = ttk.LabelFrame(f, text="Progress"); p.grid(row=3,column=0,columnspan=3,sticky="we",**pad)
        self.pb = ttk.Progressbar(p, orient="horizontal", mode="determinate", length=480); self.pb.grid(row=0,column=0,sticky="we",**pad)
        self.plabel = ttk.Label(p, text="Waiting…"); self.plabel.grid(row=0,column=1,sticky="e")

        ttk.Button(f, text="Run", command=self._run).grid(row=4,column=0,sticky="w",**pad)
        ttk.Button(f, text="Quit", command=self.destroy).grid(row=4,column=2,sticky="e",**pad)

        self.log = tk.Text(f, height=10); self.log.grid(row=5,column=0,columnspan=3,sticky="nsew",**pad)
        f.columnconfigure(1, weight=1); f.rowconfigure(5, weight=1)

    def _pick_in(self):  d = filedialog.askdirectory(title="Select RAW input folder");  self.in_dir.set(d or self.in_dir.get())
    def _pick_out(self): d = filedialog.askdirectory(title="Select output folder");     self.out_dir.set(d or self.out_dir.get())

    def _run(self):
        try:
            in_dir, out_dir = Path(self.in_dir.get()), Path(self.out_dir.get())
            if not in_dir.exists(): raise FileNotFoundError("Pick a valid RAW input folder.")
            out_dir.mkdir(parents=True, exist_ok=True)
            H, W = int(self.height.get()), int(self.width.get())
            if (H % BIN) or (W % BIN): raise ValueError(f"H={H}, W={W} must be divisible by {BIN}")

            raws = sorted(in_dir.glob("*.raw"), key=lambda p: natural_key(p.name))
            if not raws: raise FileNotFoundError("No *.raw in the input folder")

            # First pass: compute frames per file and select usable ones
            Ns = [frames_info(p, H, W)[0] for p in raws]
            valid_idxs = [i for i, N in enumerate(Ns) if N > 0]
            if not valid_idxs: raise FileNotFoundError("No readable frames.")

            Ns_valid = [Ns[i] for i in valid_idxs]
            # Offsets in final stack for each valid index
            offsets = np.cumsum([0] + Ns_valid[:-1])
            offset_map = {i: off for i, off in zip(valid_idxs, offsets)}
            total_frames = int(sum(Ns_valid))

            hh, ww = H//BIN, W//BIN

            # Disk-backed array to fill incrementally
            npy_path = out_dir / "stack_16x16_uint16.npy"
            stack_mm = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.uint16, shape=(total_frames, hh, ww))

            self.pb.configure(maximum=len(valid_idxs), value=0); self._tick(0, len(valid_idxs))
            self._log(f"Files: {len(raws)} (usable {len(valid_idxs)})  total frames={total_frames}")
            self._log(f"Value range: 0..{255*BIN*BIN} (=255*{BIN}*{BIN}). Storing uint16 sums (lossless).")

            # Process in parallel
            import threading
            threading.Thread(target=self._bg, args=(raws, valid_idxs, offset_map, stack_mm, out_dir, H, W, hh, ww, total_frames), daemon=True).start()

        except Exception as e:
            self._log(f"[ERROR] {e}"); messagebox.showerror("Error", str(e))

    def _bg(self, raws, valid_idxs, offset_map, stack_mm, out_dir, H, W, hh, ww, total_frames):
        try:
            processed = 0
            with ProcessPoolExecutor(max_workers=int(self.workers.get())) as ex:
                futs = {ex.submit(_worker, i, str(raws[i]), H, W): i for i in valid_idxs}
                for fu in as_completed(futs):
                    i = futs[fu]
                    idx, status, payload, frames = fu.result()
                    if status == "ok":
                        off = offset_map[i]
                        stack_mm[off:off+frames] = payload  # write to disk-backed array
                        self._log(f"[OK] {raws[i].name}: {frames} frames → [{off}:{off+frames})")
                    else:
                        self._log(f"[SKIP] {payload}")
                    processed += 1; self._tick(processed, len(valid_idxs))

            # Write BigTIFF (uint16) with compression
            tiff_path = out_dir / "stack_16x16_uint16.tif"
            with tiff.TiffWriter(str(tiff_path), bigtiff=True) as tw:
                for k in range(stack_mm.shape[0]):
                    tw.write(stack_mm[k], dtype=np.uint16, photometric="minisblack",
                             compression="zlib", rowsperstrip=stack_mm.shape[1])

            # Save meta JSON (fast)
            meta = dict(height=H, width=W, bin=BIN, out_h=hh, out_w=ww,
                        total_frames=int(total_frames),
                        value_range=[0, int(255*BIN*BIN)],
                        stack_npy=str(npy_path.name))
            (out_dir / "stack_16x16_meta.json").write_text(json.dumps(meta, indent=2))

            self._log(f"Done. TIFF → {tiff_path}\nNPY  → {out_dir/'stack_16x16_uint16.npy'}\nMax sum = 255*{BIN}*{BIN} = {255*BIN*BIN}")
            messagebox.showinfo("Done", f"TIFF: {tiff_path}\nNPY: {out_dir/'stack_16x16_uint16.npy'}\nRange: 0..{255*BIN*BIN}")

        except Exception as e:
            self._log(f"[ERROR] {e}"); messagebox.showerror("Error", str(e))

    def _tick(self, k, n): self.pb['value']=k; self.plabel.configure(text=f"{k} / {n} raw files"); self.update_idletasks()
    def _log(self, msg: str): self.log.insert("end", msg+"\n"); self.log.see("end"); self.update_idletasks()

def main(): App().mainloop()
if __name__ == "__main__": main()
