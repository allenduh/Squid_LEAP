#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, sys, json, re, threading, queue, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    import tifffile as tiff
except ImportError as e:
    raise SystemExit("Please 'pip install tifffile'") from e

# ============================== compression ==============================

@dataclass
class CompressionManager:
    height: int
    width: int
    frames_per_raw: int
    frames_per_file: int = 10
    box_size: int = 32
    in_dir: Path = Path(".")
    exp_dir: Path = Path(".")
    tiff_out: Optional[Path] = None
    tiff_mode: str = "binned"    # {'binned','raw'}
    batch: int = 64
    start: int = 0
    end: Optional[int] = None
    max_files: Optional[int] = None
    fps: Optional[float] = None
    prefetch_batches: int = 2
    bigtiff_threshold_gb: float = 4.0
    def roi_config_path(self) -> Path:
        return self.exp_dir / "roi_grid_config.json"

# ============================== roi ==============================

def load_roi(cfg_path: Path):
    if not cfg_path.exists():
        raise SystemExit(f"Missing ROI json: {cfg_path} (use your previous ROI selector)")
    cfg = json.loads(cfg_path.read_text())
    rows, cols = int(cfg["rows"]), int(cfg["cols"])
    centers = np.asarray(cfg.get("centers_xy_f") or cfg.get("centers") or cfg.get("cell_centers"))
    if centers is None or centers.shape != (rows * cols, 2):
        raise RuntimeError("centers missing or wrong shape in ROI json")
    centers = np.rint(centers).astype(np.int32)
    polys = cfg.get("cell_polygons", None)
    if polys is not None:
        polys = np.array(polys, dtype=np.float32)
    else:
        polys = np.zeros((rows * cols, 4, 2), dtype=np.float32)
    return rows, cols, centers, polys

def build_packed_union_indices(H: int, W: int, centers: np.ndarray, bs: int):
    h = bs // 2
    cx, cy = centers[:, 0], centers[:, 1]
    x0 = np.clip(cx - h, 0, W)
    y0 = np.clip(cy - h, 0, H)
    x1 = np.clip(x0 + bs, 0, W)
    y1 = np.clip(y0 + bs, 0, H)
    B = centers.shape[0]
    pos_list: List[np.ndarray] = []
    starts = np.empty(B, dtype=np.int64)
    areas = np.empty(B, dtype=np.int64)
    cur = 0
    for b in range(B):
        xa, xb = int(x0[b]), int(x1[b])
        ya, yb = int(y0[b]), int(y1[b])
        starts[b] = cur
        if xb > xa and yb > ya:
            yy, xx = np.mgrid[ya:yb, xa:xb]
            p = (yy * W + xx).ravel()
            pos_list.append(p); cur += p.size; areas[b] = p.size
        else:
            areas[b] = 1
    pos = np.concatenate(pos_list) if pos_list else np.array([], dtype=np.int64)
    return pos.astype(np.int64), starts, areas.astype(np.float32)

# ============================== io_raw ==============================

_DIGITS = re.compile(r"(\d+)")
def _infer_batch_index(p: Path) -> int:
    m = _DIGITS.findall(p.stem)
    return int(m[-1]) if m else 0

def _expected_size_bytes(H: int, W: int, N: int) -> int:
    return int(H) * int(W) * int(N)

def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]

def list_raw_files(in_dir: Path, max_files: Optional[int]) -> List[Tuple[int, Path]]:
    files = sorted(in_dir.glob("*.raw"), key=lambda p: _natural_key(p.name))
    if max_files is not None and max_files >= 0:
        files = files[:max_files]
    jobs = [(_infer_batch_index(p), p) for p in files]
    jobs.sort(key=lambda t: t[0])
    return jobs

def memmap_first_frame(raw_path: Path, H: int, W: int, N: int) -> np.ndarray:
    mm = np.memmap(raw_path, mode="r", dtype=np.uint8, shape=(N, H, W))
    fr = np.array(mm[0], copy=True); del mm
    return fr

# ============================== binning ==============================

def bin_batch(batch_arr: np.ndarray, pos: np.ndarray, starts: np.ndarray, areas: np.ndarray) -> np.ndarray:
    Bf, H, W = batch_arr.shape
    flat = batch_arr.reshape(Bf, -1)
    gathered = flat[:, pos]
    sums = np.add.reduceat(gathered, starts, axis=1)
    means = (sums / areas)
    return means

# ============================== writers ==============================

class TiffOut:
    def __init__(self, path: Path, bigtiff: bool):
        self._tw = tiff.TiffWriter(str(path), bigtiff=bigtiff)
    def write_page(self, arr: np.ndarray):
        self._tw.write(arr, contiguous=True, photometric="minisblack")
    def close(self):
        self._tw.close()

# ============================== pipeline ==============================

class Prefetcher(threading.Thread):
    daemon = True
    def __init__(self, jobs: List[Tuple[int, Path]], H: int, W: int, N: int, start: int, end: Optional[int], batch: int, q: "queue.Queue"):
        super().__init__()
        self.jobs, self.H, self.W, self.N = jobs, H, W, N
        self.start, self.end, self.batch, self.q = start, end, batch, q
    def run(self):
        idx = 0; buf = []; seq_start = None
        for _, rp in self.jobs:
            need = _expected_size_bytes(self.H, self.W, self.N); sz = rp.stat().st_size
            if sz < need:
                print(f"[SKIP] {rp.name} is short: {sz} < {need}"); continue
            mm = np.memmap(rp, mode="r", dtype=np.uint8, shape=(self.N, self.H, self.W))
            try:
                for i in range(self.N):
                    if idx < self.start: idx += 1; continue
                    if self.end is not None and self.end >= 0 and idx >= self.end:
                        if buf: self.q.put((seq_start, np.stack(buf, 0)))
                        self.q.put(None); return
                    if not buf: seq_start = idx
                    buf.append(np.asarray(mm[i], copy=False)); idx += 1
                    if len(buf) == self.batch:
                        self.q.put((seq_start, np.stack(buf, 0))); buf.clear()
            finally:
                del mm
        if buf: self.q.put((seq_start, np.stack(buf, 0)))
        self.q.put(None)

def run_pipeline(cm: CompressionManager, log=lambda *a, **k: None):
    rows, cols, centers, polys = load_roi(cm.roi_config_path())
    jobs = list_raw_files(cm.in_dir, cm.max_files)
    if not jobs: raise SystemExit(f"No RAW files found under {cm.in_dir}")
    first_path = jobs[0][1]
    first_frame = memmap_first_frame(first_path, cm.height, cm.width, cm.frames_per_raw).astype(np.float32)
    pos, starts, areas = build_packed_union_indices(cm.height, cm.width, centers, cm.box_size)
    B = centers.shape[0]
    total_frames = len(jobs) * cm.frames_per_raw
    s = max(0, cm.start); e = min(total_frames, cm.end) if (cm.end is not None and cm.end >= 0) else total_frames
    T = max(0, e - s)
    if T == 0: raise SystemExit("Frame range empty: nothing to process.")
    traces = np.empty((B, T), dtype=np.float32)
    if cm.tiff_out is None:
        cm.tiff_out = cm.exp_dir / ("stack_binned.tif" if cm.tiff_mode == "binned" else "stack_raw.tif")
    if cm.tiff_mode == "binned":
        use_bigtiff = False
    else:
        bytes_est = T * cm.height * cm.width
        use_bigtiff = (bytes_est >= int(cm.bigtiff_threshold_gb * (1024**3)))
    cm.exp_dir.mkdir(parents=True, exist_ok=True)
    tw = TiffOut(cm.tiff_out, bigtiff=use_bigtiff)
    qin: "queue.Queue[Optional[Tuple[int,np.ndarray]]]" = queue.Queue(maxsize=cm.prefetch_batches)
    pf = Prefetcher(jobs, cm.height, cm.width, cm.frames_per_raw, cm.start, cm.end, cm.batch, qin); pf.start()
    next_write = 0; pending: Dict[int, np.ndarray] = {}; written = 0
    try:
        while True:
            item = qin.get()
            if item is None:
                while next_write in pending:
                    ba = pending.pop(next_write)
                    written = _consume_batch(ba, next_write, traces, rows, cols, cm, pos, starts, areas, tw, log, written, T)
                    next_write += len(ba)
                break
            seq_start, ba = item
            pending[seq_start] = ba
            while next_write in pending:
                ba2 = pending.pop(next_write)
                written = _consume_batch(ba2, next_write, traces, rows, cols, cm, pos, starts, areas, tw, log, written, T)
                next_write += len(ba2)
    finally:
        tw.close()
    out = dict(
        rows=rows, cols=cols,
        box_sizes=np.array([int(cm.box_size)], dtype=np.int32),
        centers=centers, polygons=polys,
        trace_boxes=traces[None, ...],
        frame_count=T, first_frame=first_frame, first_frame_path=str(first_path),
        block_labels=np.array([f"r{r}_c{c}" for r in range(rows) for c in range(cols)], dtype=object),
        block_rc=np.array([[r, c] for r in range(rows) for c in range(cols)], dtype=np.int32),
        frame_start=int(s), frame_end=int(e),
        fps=float(cm.fps) if cm.fps is not None else np.nan,
    )
    npz_path = cm.exp_dir / "exp_block_data.npz"
    np.savez_compressed(npz_path, **out)
    log(f"[DONE] NPZ -> {npz_path} | TIFF -> {cm.tiff_out} | traces {traces.shape} | pages {T} ({cm.tiff_mode})")
    return str(cm.tiff_out), str(npz_path)

def _consume_batch(batch_arr: np.ndarray, seq_start: int, traces: np.ndarray, rows: int, cols: int,
                   cm: CompressionManager, pos: np.ndarray, starts: np.ndarray, areas: np.ndarray,
                   tw: TiffOut, log, written: int, T: int) -> int:
    means_batch = bin_batch(batch_arr, pos, starts, areas)
    Bf = means_batch.shape[0]
    end_idx = (seq_start - cm.start) + Bf
    traces[:, (seq_start - cm.start):end_idx] = means_batch.T
    if cm.tiff_mode == "binned":
        binned_frames = means_batch.reshape(Bf, rows, cols).astype(np.float32, copy=False)
        for k in range(Bf): tw.write_page(binned_frames[k])
    else:
        for k in range(Bf): tw.write_page(batch_arr[k])
    written = (seq_start - cm.start) + Bf
    if written % 200 == 0 or written == T: log(f"[{written}/{T}] frames processed…")
    return written

# ============================== app_gui ==============================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAW → (Binned TIFF + NPZ)")
        self.geometry("720x560")
        self.resizable(True, True)
        self.in_dir = tk.StringVar()
        self.exp_dir = tk.StringVar()
        self.tiff_out = tk.StringVar()
        self.height = tk.IntVar(value=608)
        self.width = tk.IntVar(value=1024)
        self.frames_per_raw = tk.IntVar(value=10)
        self.frames_per_file = tk.IntVar(value=10)
        self.box = tk.IntVar(value=32)
        self.batch = tk.IntVar(value=64)
        self.start = tk.IntVar(value=0)
        self.end = tk.StringVar(value="")
        self.max_files = tk.StringVar(value="")
        self.fps = tk.StringVar(value="500")
        self.tiff_mode = tk.StringVar(value="binned")
        self.prefetch = tk.IntVar(value=2)
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}
        frm = ttk.Frame(self); frm.pack(fill="both", expand=True, **pad)

        ttk.Label(frm, text="RAW input folder").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.in_dir, width=60).grid(row=0, column=1, sticky="we")
        ttk.Button(frm, text="Browse…", command=self._pick_in).grid(row=0, column=2, **pad)

        ttk.Label(frm, text="EXP folder (ROI json)").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.exp_dir, width=60).grid(row=1, column=1, sticky="we")
        ttk.Button(frm, text="Browse…", command=self._pick_exp).grid(row=1, column=2, **pad)

        ttk.Button(frm, text="Open previous ROI selector", command=self._open_roi_selector).grid(row=2, column=1, sticky="w", **pad)

        g = ttk.LabelFrame(frm, text="Geometry & RAW layout")
        g.grid(row=3, column=0, columnspan=3, sticky="we", **pad)
        for i in range(3): g.columnconfigure(i, weight=1)
        ttk.Label(g, text="Height").grid(row=0, column=0, sticky="e"); ttk.Entry(g, textvariable=self.height, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(g, text="Width").grid(row=0, column=2, sticky="e");  ttk.Entry(g, textvariable=self.width,  width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(g, text="Frames per RAW (N)").grid(row=1, column=0, sticky="e"); ttk.Entry(g, textvariable=self.frames_per_raw, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(g, text="Frames per file (compat)").grid(row=1, column=2, sticky="e"); ttk.Entry(g, textvariable=self.frames_per_file, width=8).grid(row=1, column=3, sticky="w")

        b = ttk.LabelFrame(frm, text="Binning & Output")
        b.grid(row=4, column=0, columnspan=3, sticky="we", **pad)
        for i in range(3): b.columnconfigure(i, weight=1)
        ttk.Label(b, text="Box size").grid(row=0, column=0, sticky="e"); ttk.Entry(b, textvariable=self.box, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(b, text="TIFF mode").grid(row=0, column=2, sticky="e")
        ttk.Combobox(b, textvariable=self.tiff_mode, values=["binned", "raw"], width=10, state="readonly").grid(row=0, column=3, sticky="w")
        ttk.Label(b, text="TIFF out (optional)").grid(row=1, column=0, sticky="e")
        ttk.Entry(b, textvariable=self.tiff_out, width=48).grid(row=1, column=1, sticky="we", columnspan=2)
        ttk.Button(b, text="Choose…", command=self._pick_tiff_out).grid(row=1, column=3, sticky="w", **pad)

        r = ttk.LabelFrame(frm, text="Range & Performance")
        r.grid(row=5, column=0, columnspan=3, sticky="we", **pad)
        for i in range(3): r.columnconfigure(i, weight=1)
        ttk.Label(r, text="Start").grid(row=0, column=0, sticky="e"); ttk.Entry(r, textvariable=self.start, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(r, text="End (exclusive)").grid(row=0, column=2, sticky="e"); ttk.Entry(r, textvariable=self.end, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(r, text="Max files").grid(row=1, column=0, sticky="e"); ttk.Entry(r, textvariable=self.max_files, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(r, text="Batch").grid(row=1, column=2, sticky="e"); ttk.Entry(r, textvariable=self.batch, width=8).grid(row=1, column=3, sticky="w")
        ttk.Label(r, text="Prefetch batches").grid(row=2, column=0, sticky="e"); ttk.Entry(r, textvariable=self.prefetch, width=8).grid(row=2, column=1, sticky="w")
        ttk.Label(r, text="FPS (metadata)").grid(row=2, column=2, sticky="e"); ttk.Entry(r, textvariable=self.fps, width=8).grid(row=2, column=3, sticky="w")

        runf = ttk.Frame(frm); runf.grid(row=6, column=0, columnspan=3, sticky="we", **pad)
        ttk.Button(runf, text="Run", command=self._run).pack(side="left")
        ttk.Button(runf, text="Quit", command=self.destroy).pack(side="right")

        self.logw = tk.Text(frm, height=12)
        self.logw.grid(row=7, column=0, columnspan=3, sticky="nsew", **pad)
        frm.rowconfigure(7, weight=1); frm.columnconfigure(1, weight=1)

    def _pick_in(self):
        d = filedialog.askdirectory(title="Select RAW input folder")
        if d: self.in_dir.set(d)

    def _pick_exp(self):
        d = filedialog.askdirectory(title="Select EXP folder (roi_grid_config.json)")
        if d: self.exp_dir.set(d)

    def _pick_tiff_out(self):
        f = filedialog.asksaveasfilename(title="Select TIFF output path", defaultextension=".tif",
                                         filetypes=[("TIFF", "*.tif *.tiff")])
        if f: self.tiff_out.set(f)

    def _open_roi_selector(self):
        candidates = [Path(self.exp_dir.get()) / "roi_select_gui.py", Path.cwd() / "roi_select_gui.py"]
        for c in candidates:
            if c.exists():
                try:
                    subprocess.Popen([sys.executable, str(c)], cwd=str(c.parent))
                    self._log(f"Launched ROI selector: {c}"); return
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to launch ROI selector:\n{e}"); return
        messagebox.showinfo("Not found", "Could not find 'roi_select_gui.py'. Place it in the EXP folder or current directory.")

    def _log(self, msg: str):
        self.logw.insert("end", msg + "\n"); self.logw.see("end"); self.update_idletasks()

    def _run(self):
        try:
            cm = CompressionManager(
                height=int(self.height.get()), width=int(self.width.get()),
                frames_per_raw=int(self.frames_per_raw.get()), frames_per_file=int(self.frames_per_file.get()),
                box_size=int(self.box.get()),
                in_dir=Path(self.in_dir.get()), exp_dir=Path(self.exp_dir.get()),
                tiff_out=(Path(self.tiff_out.get()) if self.tiff_out.get().strip() else None),
                tiff_mode=self.tiff_mode.get(), batch=int(self.batch.get()),
                start=int(self.start.get()), end=(int(self.end.get()) if self.end.get().strip() else None),
                max_files=(int(self.max_files.get()) if self.max_files.get().strip() else None),
                fps=(float(self.fps.get()) if self.fps.get().strip() else None),
                prefetch_batches=int(self.prefetch.get()),
            )
        except Exception as e:
            messagebox.showerror("Invalid parameters", str(e)); return

        if not cm.in_dir.exists():
            messagebox.showerror("Missing input", f"Input folder not found: {cm.in_dir}"); return
        if not cm.exp_dir.exists():
            messagebox.showerror("Missing EXP", f"EXP folder not found: {cm.exp_dir}"); return
        if not cm.roi_config_path().exists():
            messagebox.showwarning("ROI json missing", f"ROI json not found at {cm.roi_config_path()}.\nUse your previous ROI selector to create it."); return

        self._log(f"Starting… in={cm.in_dir} exp={cm.exp_dir} tiff_mode={cm.tiff_mode} box={cm.box_size} batch={cm.batch}")
        def _run_bg():
            try:
                def logger(msg): self._log(msg)
                tiff_path, npz_path = run_pipeline(cm, log=logger)
                self._log(f"Done.\nTIFF: {tiff_path}\nNPZ: {npz_path}")
                messagebox.showinfo("Done", f"TIFF: {tiff_path}\nNPZ: {npz_path}")
            except Exception as e:
                self._log(f"[ERROR] {e}"); messagebox.showerror("Error", str(e))
        threading.Thread(target=_run_bg, daemon=True).start()

def main():
    app = App(); app.mainloop()

if __name__ == "__main__":
    main()
