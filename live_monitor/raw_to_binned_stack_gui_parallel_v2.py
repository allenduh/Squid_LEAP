#!/usr/bin/env python3

from __future__ import annotations
import os, json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import tifffile as tiff
import re

BIN, N = 16, 10  # constants

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def worker(i: int, raw_path: str, H: int, W: int, hh: int, ww: int) -> tuple[int,str,object]:
    p = Path(raw_path); 

    buf = np.fromfile(p, dtype=np.uint8)
    arr = buf.reshape(N, H, W)
    return i, arr.reshape(N, hh, BIN, ww, BIN).sum(axis=(2,4), dtype=np.uint16)

def write_tiff_imagej_safe(tiff_out, frames_u16):  # frames_u16: (T, H, W), dtype=uint16
    H, W = frames_u16.shape[1:]
    with tiff.TiffWriter(str(tiff_out), bigtiff=True) as tw:
        for k in range(frames_u16.shape[0]):
            tw.write(
                frames_u16[k],
                dtype=np.uint16,
                photometric='minisblack',
                contiguous=True,
                rowsperstrip=H,      # one strip per page (ImageJ-friendly)
                compression=None,    # safest
            )

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAW → 16×16 uint16 sums (BigTIFF + NPY)")
        self.geometry("620x360")
        self.in_dir, self.out_dir = tk.StringVar(), tk.StringVar()
        self.height, self.width = tk.IntVar(value=608), tk.IntVar(value=1024)
        self.workers = tk.IntVar(value=os.cpu_count() or 8)
        self._build()

    def _build(self):
        pad = {"padx":6, "pady":4}
        f = ttk.Frame(self); f.pack(fill="both", expand=True, **pad)
        ttk.Label(f, text="RAW input").grid(row=0,column=0,sticky="e")
        ttk.Entry(f, textvariable=self.in_dir, width=50).grid(row=0,column=1,sticky="we")
        ttk.Button(f, text="Browse…", command=self._pick_in).grid(row=0,column=2)
        ttk.Label(f, text="Output").grid(row=1,column=0,sticky="e")
        ttk.Entry(f, textvariable=self.out_dir, width=50).grid(row=1,column=1,sticky="we")
        ttk.Button(f, text="Browse…", command=self._pick_out).grid(row=1,column=2)
        g = ttk.LabelFrame(f, text="Settings"); g.grid(row=2,column=0,columnspan=3,sticky="we",**pad)
        ttk.Label(g, text="H").grid(row=0,column=0,sticky="e"); ttk.Entry(g,textvariable=self.height,width=7).grid(row=0,column=1,sticky="w")
        ttk.Label(g, text="W").grid(row=0,column=2,sticky="e"); ttk.Entry(g,textvariable=self.width,width=7).grid(row=0,column=3,sticky="w")
        ttk.Label(g, text="Bin").grid(row=0,column=4,sticky="e"); ttk.Label(g,text="16×16 (uint16 sums)").grid(row=0,column=5,sticky="w")
        ttk.Label(g, text="Workers").grid(row=0,column=6,sticky="e"); ttk.Entry(g,textvariable=self.workers,width=6).grid(row=0,column=7,sticky="w")
        p = ttk.LabelFrame(f, text="Progress"); p.grid(row=3,column=0,columnspan=3,sticky="we",**pad)
        self.pb = ttk.Progressbar(p, mode="determinate", length=460); self.pb.grid(row=0,column=0,sticky="we",**pad)
        self.plabel = ttk.Label(p, text="Waiting…"); self.plabel.grid(row=0,column=1,sticky="e")
        ttk.Button(f, text="Run", command=self._run).grid(row=4,column=0,sticky="w",**pad)
        ttk.Button(f, text="Quit", command=self.destroy).grid(row=4,column=2,sticky="e",**pad)
        self.log = tk.Text(f, height=10); self.log.grid(row=5,column=0,columnspan=3,sticky="nsew",**pad)
        f.columnconfigure(1, weight=1); f.rowconfigure(5, weight=1)

    def _pick_in(self):  d = filedialog.askdirectory(title="Select RAW input");  self.in_dir.set(d or self.in_dir.get())
    def _pick_out(self): d = filedialog.askdirectory(title="Select output");     self.out_dir.set(d or self.out_dir.get())
    def _ui(self, fn, *a, **k): self.after(0, lambda: fn(*a, **k))
    def _log(self, s): self.log.insert("end", s+"\n"); self.log.see("end")

    def _run(self):
        try:
            in_dir, out_dir = Path(self.in_dir.get()), Path(self.out_dir.get())
            if not in_dir.exists(): raise FileNotFoundError("Pick a valid RAW input folder")
            out_dir.mkdir(parents=True, exist_ok=True)
            H, W = int(self.height.get()), int(self.width.get())
            if (H % BIN) or (W % BIN): raise ValueError(f"H={H}, W={W} must be divisible by {BIN}")
            raws = sorted(in_dir.glob("*.raw"), key=lambda p: natural_key(p.name))
            if not raws: raise FileNotFoundError("No *.raw in the folder")

            hh, ww = H//BIN, W//BIN
            total_files, total_frames = len(raws), N*len(raws)
            npy_path = out_dir/"stack_16x16_uint16.npy"
            stack = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.uint16, shape=(total_frames, hh, ww))

            self.pb.configure(maximum=total_files, value=0); self.plabel.configure(text=f"0 / {total_files}")
            self._log(f"Files: {total_files}, frames/file={N}, total frames={total_frames}")
            self._log(f"Value range: 0..{255*BIN*BIN} (=255*{BIN}*{BIN})")

            def bg():
                processed = 0
                with ProcessPoolExecutor(max_workers=int(self.workers.get())) as ex:
                    futs = {ex.submit(worker, i, str(raws[i]), H, W, hh, ww): i for i in range(total_files)}
                    for fu in as_completed(futs):
                        i = futs[fu]
                        i2, payload = fu.result()

                        off = i*N; stack[off:off+N] = payload
                        self._ui(self._log, f"[OK] {raws[i].name} → [{off}:{off+N})")
                        
                        processed += 1; self._ui(self._progress, processed, total_files)

                tiff_path = out_dir/"stack_16x16_uint16.tif"
                tiff.imwrite(tiff_path, stack, dtype=np.uint16,imagej=True)

                
       

                meta = dict(height=H,width=W,bin=BIN,out_h=hh,out_w=ww,total_frames=total_frames,
                            value_range=[0,255*BIN*BIN], stack_npy=npy_path.name, tiff=tiff_path.name)
                (out_dir/"stack_16x16_meta.json").write_text(json.dumps(meta, indent=2))
                self._ui(messagebox.showinfo, "Done", f"TIFF: {tiff_path}\nNPY: {npy_path}\nShape: {(total_frames,hh,ww)}")

            import threading; threading.Thread(target=bg, daemon=True).start()

        except Exception as e:
            self._log(f"[ERROR] {e}"); messagebox.showerror("Error", str(e))

    def _progress(self, k, n): self.pb['value']=k; self.plabel.configure(text=f"{k} / {n}"); self.update_idletasks()

def main(): App().mainloop()
if __name__ == "__main__": main()

