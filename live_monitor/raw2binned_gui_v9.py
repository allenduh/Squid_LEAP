#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, sys, json, re, threading, queue, subprocess, traceback, time
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
    roi_json_file: Path = Path("roi_grid_config.json")
    tiff_out: Optional[Path] = None
    tiff_mode: str = "binned"     # {'binned','raw'}
    batch: int = 64
    start: int = 0
    end: Optional[int] = None
    max_files: Optional[int] = None
    fps: Optional[float] = 5000.0
    prefetch_batches: int = 2
    bigtiff_threshold_gb: float = 4.0

    def exp_dir(self) -> Path:
        return self.roi_json_file.parent

    def roi_config_path(self) -> Path:
        return self.roi_json_file

# ============================== roi ==============================

def load_roi(cfg_path: Path):
    if not cfg_path.exists():
        raise SystemExit(f"Missing ROI json: {cfg_path}")
    cfg = json.loads(cfg_path.read_text())
    rows_v = int(cfg["rows"])
    cols_v = int(cfg["cols"])
    centers = np.asarray(cfg.get("centers_xy_f") or cfg.get("centers") or cfg.get("cell_centers"))
    if centers is None or centers.shape != (rows_v * cols_v, 2):
        raise RuntimeError("centers missing or wrong shape in ROI json")
    centers = np.rint(centers).astype(np.int32)

    polys = cfg.get("cell_polygons", None)
    if polys is not None:
        polys = np.array(polys, dtype=np.float32)
    else:
        polys = np.zeros((rows_v * cols_v, 4, 2), dtype=np.float32)
    return rows_v, cols_v, centers, polys

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

def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]

def list_raw_files(in_dir: Path, max_files: Optional[int]) -> List[Tuple[int, Path]]:
    files = sorted(in_dir.glob("*.raw"), key=lambda p: _natural_key(p.name))
    if max_files is not None and max_files >= 0:
        files = files[:max_files]
    jobs = [(_infer_batch_index(p), p) for p in files]
    jobs.sort(key=lambda t: t[0])
    return jobs

def infer_frames_per_raw_by_size(raw_path: Path, H: int, W: int) -> Optional[int]:
    bpp = 1
    frame_bytes = H * W * bpp
    if frame_bytes <= 0: return None
    sz = raw_path.stat().st_size
    if sz % frame_bytes == 0:
        return sz // frame_bytes
    return None

def read_first_nonzero_frame(raw_path: Path, H: int, W: int, frames_hint: Optional[int]=None, max_scan:int=50) -> Optional[np.ndarray]:
    frame_bytes = H * W
    N = frames_hint or infer_frames_per_raw_by_size(raw_path, H, W) or 0
    # direct first frame
    with open(raw_path, "rb") as f:
        data = f.read(frame_bytes)
    if len(data) >= frame_bytes:
        fr = np.frombuffer(data[:frame_bytes], dtype=np.uint8).reshape(H, W)
        if np.any(fr): return fr
    # scan with mmap if N plausible
    if N > 0:
        try:
            mm = np.memmap(raw_path, mode="r", dtype=np.uint8, shape=(N, H, W))
            for i in range(min(N, max_scan)):
                fr = np.array(mm[i], copy=True)
                if np.any(fr): del mm; return fr
            del mm
        except Exception:
            pass
    # try transposed
    N2 = frames_hint or infer_frames_per_raw_by_size(raw_path, W, H) or 0
    if N2 > 0:
        try:
            mm = np.memmap(raw_path, mode="r", dtype=np.uint8, shape=(N2, W, H))
            for i in range(min(N2, max_scan)):
                frt = np.array(mm[i], copy=True).T
                if np.any(frt): del mm; return frt
            del mm
        except Exception:
            pass
    return None

# ============================== binning ==============================

def bin_batch(batch_arr: np.ndarray, pos: np.ndarray, starts: np.ndarray, areas: np.ndarray) -> np.ndarray:
    Bf = batch_arr.shape[0]
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
    def __init__(self, jobs: List[Tuple[int, Path]], H: int, W: int, N: int,
                 start_idx: int, end_idx: Optional[int], batch: int, q: "queue.Queue"):
        super().__init__()
        self.jobs, self.H, self.W, self.N = jobs, H, W, N
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.batch = batch
        self.q = q
    def run(self):
        idx = 0; buf = []; seq_start = None
        for _, rp in self.jobs:
            need = int(self.H) * int(self.W) * int(self.N)
            sz = rp.stat().st_size
            if sz < need:
                print(f"[SKIP] {rp.name} is short: {sz} < {need}"); continue
            mm = np.memmap(rp, mode="r", dtype=np.uint8, shape=(self.N, self.H, self.W))
            try:
                for i in range(self.N):
                    if idx < self.start_idx: idx += 1; continue
                    if self.end_idx is not None and self.end_idx >= 0 and idx >= self.end_idx:
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
    rows_v, cols_v, centers, polys = load_roi(cm.roi_config_path())
    jobs = list_raw_files(cm.in_dir, cm.max_files)
    if not jobs: raise SystemExit(f"No RAW files found under {cm.in_dir}")
    first_path = jobs[0][1]
    # Use robust first frame for metadata only
    inferN = infer_frames_per_raw_by_size(first_path, cm.height, cm.width) or cm.frames_per_raw
    fr0 = read_first_nonzero_frame(first_path, cm.height, cm.width, frames_hint=inferN) or np.zeros((cm.height, cm.width), dtype=np.uint8)
    first_frame = fr0.astype(np.float32)

    pos, starts, areas = build_packed_union_indices(cm.height, cm.width, centers, cm.box_size)
    B = centers.shape[0]
    total_frames = len(jobs) * cm.frames_per_raw
    s_idx = max(0, cm.start); e_idx = min(total_frames, cm.end) if (cm.end is not None and cm.end >= 0) else total_frames
    T = max(0, e_idx - s_idx)
    if T == 0: raise SystemExit("Frame range empty: nothing to process.")
    traces = np.empty((B, T), dtype=np.float32)

    out_dir = cm.exp_dir()
    tiff_path = out_dir / ("stack_binned.tif" if cm.tiff_mode == "binned" else "stack_raw.tif") if cm.tiff_out is None else cm.tiff_out
    use_bigtiff = False if cm.tiff_mode == "binned" else (T * cm.height * cm.width >= int(cm.bigtiff_threshold_gb * (1024**3)))
    out_dir.mkdir(parents=True, exist_ok=True)
    tw = TiffOut(tiff_path, bigtiff=use_bigtiff)

    qin: "queue.Queue[Optional[Tuple[int,np.ndarray]]]" = queue.Queue(maxsize=cm.prefetch_batches)
    pf = Prefetcher(jobs, cm.height, cm.width, cm.frames_per_raw, s_idx, e_idx, cm.batch, qin)
    pf.start()
    next_write = 0; pending: Dict[int, np.ndarray] = {}; written = 0
    try:
        while True:
            item = qin.get()
            if item is None:
                while next_write in pending:
                    ba = pending.pop(next_write)
                    written = _consume_batch(ba, next_write, traces, rows_v, cols_v, cm, pos, starts, areas, tw, log, written, T)
                    next_write += ba.shape[0]
                break
            seq_start, ba = item
            pending[seq_start] = ba
            while next_write in pending:
                ba2 = pending.pop(next_write)
                written = _consume_batch(ba2, next_write, traces, rows_v, cols_v, cm, pos, starts, areas, tw, log, written, T)
                next_write += ba2.shape[0]
    finally:
        tw.close()

    out = dict(
        rows=rows_v, cols=cols_v,
        box_sizes=np.array([int(cm.box_size)], dtype=np.int32),
        centers=centers, polygons=polys,
        trace_boxes=traces[None, ...],
        frame_count=T, first_frame=first_frame, first_frame_path=str(first_path),
        block_labels=np.array([f"r{r}_c{c}" for r in range(rows_v) for c in range(cols_v)], dtype=object),
        block_rc=np.array([[r, c] for r in range(rows_v) for c in range(cols_v)], dtype=np.int32),
        frame_start=int(s_idx), frame_end=int(e_idx),
        fps=float(cm.fps) if cm.fps is not None else np.nan,
    )
    npz_path = out_dir / "exp_block_data.npz"
    np.savez_compressed(npz_path, **out)
    log(f"[DONE] NPZ -> {npz_path} | TIFF -> {tiff_path} | traces {traces.shape} | pages {T} ({cm.tiff_mode})")
    return str(tiff_path), str(npz_path)

def _consume_batch(batch_arr: np.ndarray, seq_start: int, traces: np.ndarray, rows_v: int, cols_v: int,
                   cm: CompressionManager, pos: np.ndarray, starts: np.ndarray, areas: np.ndarray,
                   tw: TiffOut, log, written: int, T: int) -> int:
    means_batch = bin_batch(batch_arr, pos, starts, areas)
    Bf = means_batch.shape[0]
    s_local = (seq_start - cm.start)
    e_local = s_local + Bf
    traces[:, s_local:e_local] = means_batch.T
    if cm.tiff_mode == "binned":
        binned_frames = means_batch.reshape(Bf, rows_v, cols_v).astype(np.float32, copy=False)
        for k in range(Bf): tw.write_page(binned_frames[k])
    else:
        for k in range(Bf): tw.write_page(batch_arr[k])
    written = e_local
    if (written % 200) == 0 or (written == T): log(f"[{written}/{T}] frames processed…")
    return written

# ============================== app_gui ==============================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_STATE = Path(__file__).with_suffix(".state.json")

def to_int_or_none(text: str) -> Optional[int]:
    t = text.strip()
    if t == "": return None
    return int(t)

def auto_contrast_u8(img: np.ndarray, lo_p=1.0, hi_p=99.0) -> np.ndarray:
    arr = img.astype(np.float32, copy=False)
    lo = np.percentile(arr, lo_p)
    hi = np.percentile(arr, hi_p)
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
        if hi <= lo:
            return np.zeros_like(img, dtype=np.uint8)
    scaled = (arr - lo) * (255.0 / max(1e-6, (hi - lo)))
    return np.clip(scaled, 0, 255).astype(np.uint8)

def create_seed_bmp(in_dir: Path, out_dir: Path, H: int, W: int, frames_hint: Optional[int]) -> Path:
    jobs = list_raw_files(in_dir, max_files=5)
    if not jobs:
        raise RuntimeError("No RAW files found to create a seed image.")
    fr = None
    for _, rp in jobs:
        N_guess = frames_hint or infer_frames_per_raw_by_size(rp, H, W)
        fr = read_first_nonzero_frame(rp, H, W, frames_hint=N_guess)
        if fr is not None:
            break
    if fr is None:
        raise RuntimeError("Could not find a non-zero frame in the first few RAW files. Check height/width.")
    fr = auto_contrast_u8(fr)
    out_bmp = out_dir / "roi_seed.bmp"
    try:
        from PIL import Image  # type: ignore
        Image.fromarray(fr).save(out_bmp)
    except Exception:
        tiff.imwrite(str(out_dir / "roi_seed.tif"), fr)
        out_bmp = out_dir / "roi_seed.tif"
    return out_bmp

def load_app_state() -> Dict[str, str]:
    try:
        return json.loads(APP_STATE.read_text())
    except Exception:
        return {}

def save_app_state(d: Dict[str, str]) -> None:
    try:
        APP_STATE.write_text(json.dumps(d, indent=2))
    except Exception:
        pass

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAW → (Binned TIFF + NPZ)")
        self.geometry("920x760")
        self.resizable(True, True)
        # Vars
        self.in_dir = tk.StringVar()
        self.roi_json = tk.StringVar()  # direct JSON
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
        self.fps = tk.StringVar(value="5000")
        self.tiff_mode = tk.StringVar(value="binned")
        self.prefetch = tk.IntVar(value=2)
        self.selector_path = tk.StringVar()  # path to roi_select_gui.py
        self._restore_state()
        self._build_ui()

    def _restore_state(self):
        s = load_app_state()
        if "selector_path" in s:
            self.selector_path.set(s["selector_path"])

    def _persist_state(self):
        s = load_app_state()
        s["selector_path"] = self.selector_path.get()
        save_app_state(s)

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}
        frm = ttk.Frame(self); frm.pack(fill="both", expand=True, **pad)

        ttk.Label(frm, text="RAW input folder").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.in_dir, width=62).grid(row=0, column=1, sticky="we")
        ttk.Button(frm, text="Browse…", command=self._pick_in).grid(row=0, column=2, **pad)

        ttk.Label(frm, text="ROI JSON").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.roi_json, width=62).grid(row=1, column=1, sticky="we")
        ttk.Button(frm, text="Browse JSON…", command=self._pick_roi_json).grid(row=1, column=2, **pad)

        sframe = ttk.LabelFrame(frm, text="ROI Selector")
        sframe.grid(row=2, column=0, columnspan=3, sticky="we", **pad)
        ttk.Label(sframe, text="Path to roi_select_gui.py").grid(row=0, column=0, sticky="e")
        ttk.Entry(sframe, textvariable=self.selector_path, width=50).grid(row=0, column=1, sticky="we")
        ttk.Button(sframe, text="Browse…", command=self._pick_selector_path).grid(row=0, column=2, sticky="w")
        ttk.Button(sframe, text="Save", command=self._persist_state).grid(row=0, column=3, sticky="w", padx=4)
        sframe.columnconfigure(1, weight=1)

        rframe = ttk.Frame(frm); rframe.grid(row=3, column=0, columnspan=3, sticky="w", **pad)
        ttk.Button(rframe, text="Create/Select ROI JSON (one-click flow)", command=self._one_click_roi_json).pack(side="left")

        g = ttk.LabelFrame(frm, text="Geometry & RAW layout")
        g.grid(row=4, column=0, columnspan=3, sticky="we", **pad)
        for i in range(4): g.columnconfigure(i, weight=1)
        ttk.Label(g, text="Height").grid(row=0, column=0, sticky="e"); ttk.Entry(g, textvariable=self.height, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(g, text="Width").grid(row=0, column=2, sticky="e");  ttk.Entry(g, textvariable=self.width,  width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(g, text="Frames per RAW (N)").grid(row=1, column=0, sticky="e"); ttk.Entry(g, textvariable=self.frames_per_raw, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(g, text="Frames per file (compat)").grid(row=1, column=2, sticky="e"); ttk.Entry(g, textvariable=self.frames_per_file, width=8).grid(row=1, column=3, sticky="w")

        b = ttk.LabelFrame(frm, text="Binning & Output")
        b.grid(row=5, column=0, columnspan=3, sticky="we", **pad)
        for i in range(4): b.columnconfigure(i, weight=1)
        ttk.Label(b, text="Box size").grid(row=0, column=0, sticky="e"); ttk.Entry(b, textvariable=self.box, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(b, text="TIFF mode").grid(row=0, column=2, sticky="e")
        ttk.Combobox(b, textvariable=self.tiff_mode, values=["binned", "raw"], width=10, state="readonly").grid(row=0, column=3, sticky="w")
        ttk.Label(b, text="TIFF out (optional)").grid(row=1, column=0, sticky="e")
        ttk.Entry(b, textvariable=self.tiff_out, width=48).grid(row=1, column=1, sticky="we", columnspan=2)
        ttk.Button(b, text="Choose…", command=self._pick_tiff_out).grid(row=1, column=3, sticky="w", **pad)

        r = ttk.LabelFrame(frm, text="Range & Performance")
        r.grid(row=6, column=0, columnspan=3, sticky="we", **pad)
        for i in range(4): r.columnconfigure(i, weight=1)
        ttk.Label(r, text="Start").grid(row=0, column=0, sticky="e"); ttk.Entry(r, textvariable=self.start, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(r, text="End (exclusive)").grid(row=0, column=2, sticky="e"); ttk.Entry(r, textvariable=self.end, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(r, text="Max files (quick test)").grid(row=1, column=0, sticky="e"); ttk.Entry(r, textvariable=self.max_files, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(r, text="Batch (frames/chunk)").grid(row=1, column=2, sticky="e"); ttk.Entry(r, textvariable=self.batch, width=8).grid(row=1, column=3, sticky="w")
        ttk.Label(r, text="Prefetch queue depth").grid(row=2, column=0, sticky="e"); ttk.Entry(r, textvariable=self.prefetch, width=8).grid(row=2, column=1, sticky="w")
        ttk.Label(r, text="FPS (metadata)").grid(row=2, column=2, sticky="e"); ttk.Entry(r, textvariable=self.fps, width=8).grid(row=2, column=3, sticky="w")

        runf = ttk.Frame(frm); runf.grid(row=7, column=0, columnspan=3, sticky="we", **pad)
        ttk.Button(runf, text="Run", command=self._run).pack(side="left")
        ttk.Button(runf, text="Quit", command=self.destroy).pack(side="right")

        self.logw = tk.Text(frm, height=12)
        self.logw.grid(row=8, column=0, columnspan=3, sticky="nsew", **pad)
        frm.rowconfigure(8, weight=1); frm.columnconfigure(1, weight=1)

    def _pick_in(self):
        d = filedialog.askdirectory(title="Select RAW input folder")
        if d: self.in_dir.set(d)

    def _pick_roi_json(self):
        f = filedialog.askopenfilename(title="Select ROI JSON", filetypes=[("JSON","*.json")])
        if f: self.roi_json.set(f)

    def _pick_tiff_out(self):
        f = filedialog.asksaveasfilename(title="Select TIFF output path", defaultextension=".tif",
                                         filetypes=[("TIFF", "*.tif *.tiff")])
        if f: self.tiff_out.set(f)

    def _pick_selector_path(self):
        f = filedialog.askopenfilename(title="Select roi_select_gui.py", filetypes=[("Python","*.py"), ("All","*.*")])
        if f: self.selector_path.set(f)

    def _one_click_roi_json(self):
        """Single button: ensure seed BMP exists, launch selector on ROI folder, wait, and fill JSON path."""
        try:
            in_dir = Path(self.in_dir.get())
            if not in_dir.exists():
                messagebox.showerror("Missing input", "Pick RAW input folder first."); return
            # ROI folder from existing JSON path or ask for folder
            if self.roi_json.get().strip():
                roi_dir = Path(self.roi_json.get()).parent
                roi_json_path = Path(self.roi_json.get())
            else:
                d = filedialog.askdirectory(title="Select folder to hold ROI JSON + seed BMP")
                if not d: return
                roi_dir = Path(d)
                roi_json_path = roi_dir / "roi_grid_config.json"
                self.roi_json.set(str(roi_json_path))

            # Create seed bmp if not present
            cand = None
            for ext in ("*.bmp","*.png","*.tif","*.tiff"):
                hits = list(roi_dir.glob(ext))
                if hits:
                    cand = hits[0]; break
            if cand is None:
                self._log("Creating seed image from RAW…")
                try:
                    out_path = create_seed_bmp(in_dir, roi_dir, int(self.height.get()), int(self.width.get()), int(self.frames_per_raw.get()))
                    cand = out_path
                    self._log(f"Seed written: {cand}")
                except Exception as e:
                    self._log(f"[ERROR] Seed creation failed: {e}\n{traceback.format_exc()}")
                    messagebox.showerror("Seed creation failed", str(e)); return

            # Selector script
            if not self.selector_path.get().strip():
                sf = filedialog.askopenfilename(title="Select roi_select_gui.py", filetypes=[("Python","*.py"), ("All","*.*")])
                if not sf: return
                self.selector_path.set(sf)
                self._persist_state()
            script = Path(self.selector_path.get())
            if not script.exists():
                messagebox.showerror("Selector missing", f"Cannot find roi_select_gui.py at:\n{script}"); return

            # Launch (pass ROI folder preferred; your script uses folder to find BMP)
            self._log(f"Launching ROI selector on folder: {roi_dir}")
            def _bg():
                try:
                    # Try with ROI folder argument first
                    try:
                        p = subprocess.Popen([sys.executable, str(script), str(roi_dir)], cwd=str(script.parent))
                    except Exception:
                        # Fallback: without args (script should use CWD)
                        p = subprocess.Popen([sys.executable, str(script)], cwd=str(roi_dir))
                    rc = p.wait()
                    self._log(f"ROI selector exited with code {rc}")
                    # After exit, check JSON exists
                    if roi_json_path.exists():
                        self._log(f"[ROI] JSON found: {roi_json_path}")
                        self.roi_json.set(str(roi_json_path))
                        messagebox.showinfo("ROI JSON ready", f"Saved:\n{roi_json_path}")
                    else:
                        self._log(f"[ROI] JSON NOT found in {roi_dir}. If your selector saved elsewhere, browse it and set here.")
                        messagebox.showwarning("JSON not found", "Selector closed but JSON wasn't found in the ROI folder.")
                except Exception as e:
                    self._log(f"[ERROR] Launching selector: {e}\n{traceback.format_exc()}")
                    messagebox.showerror("Launch error", str(e))
            threading.Thread(target=_bg, daemon=True).start()
        except Exception as e:
            self._log(f"[ERROR] One-click ROI flow: {e}\n{traceback.format_exc()}")
            messagebox.showerror("Error", str(e))

    def _log(self, msg: str):
        self.logw.insert("end", msg + "\n"); self.logw.see("end"); self.update_idletasks()

    def _run(self):
        try:
            roi_json_path = Path(self.roi_json.get()) if self.roi_json.get().strip() else None
            if roi_json_path is None:
                messagebox.showwarning("ROI missing", "Please provide ROI JSON (click 'Create/Select ROI JSON')."); return
            cm = CompressionManager(
                height=int(self.height.get()), width=int(self.width.get()),
                frames_per_raw=int(self.frames_per_raw.get()), frames_per_file=int(self.frames_per_file.get()),
                box_size=int(self.box.get()),
                in_dir=Path(self.in_dir.get()),
                roi_json_file=roi_json_path,
                tiff_out=(Path(self.tiff_out.get()) if self.tiff_out.get().strip() else None),
                tiff_mode=self.tiff_mode.get(), batch=int(self.batch.get()),
                start=int(self.start.get()), end=to_int_or_none(self.end.get()),
                max_files=to_int_or_none(self.max_files.get()),
                fps=(float(self.fps.get()) if self.fps.get().strip() else 5000.0),
                prefetch_batches=int(self.prefetch.get()),
            )
        except Exception as e:
            messagebox.showerror("Invalid parameters", str(e)); return

        if not cm.in_dir.exists():
            messagebox.showerror("Missing input", f"Input folder not found: {cm.in_dir}"); return
        if not cm.roi_config_path().exists():
            messagebox.showwarning("ROI json missing", f"ROI json not found at {cm.roi_config_path()}.\nUse the ROI button to create it."); return

        self._log(f"Starting… in={cm.in_dir} ROI JSON={cm.roi_config_path()} tiff_mode={cm.tiff_mode} box={cm.box_size} batch={cm.batch}")
        def _run_bg():
            try:
                def logger(msg): self._log(msg)
                tiff_path, npz_path = run_pipeline(cm, log=logger)
                self._log(f"Done.\nTIFF: {tiff_path}\nNPZ: {npz_path}")
                messagebox.showinfo("Done", f"TIFF: {tiff_path}\nNPZ: {npz_path}")
            except Exception as e:
                self._log(f"[ERROR] {e}\n{traceback.format_exc()}"); messagebox.showerror("Error", str(e))
        threading.Thread(target=_run_bg, daemon=True).start()

def main():
    app = App(); app.mainloop()

if __name__ == "__main__":
    main()
