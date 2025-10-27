#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os, sys, json, re, threading, queue, subprocess, traceback
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

# ============================== roi (embedded) ==============================
# Adapted from your roi_select_gui.py: draggable 4-point, W/S, D/A, G, ENTER/Q save.
# (So you can open a proper selector inside this GUI without a separate script.)

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]

def list_raw_files(in_dir: Path, max_files: Optional[int]) -> List[Tuple[int, Path]]:
    files = sorted(in_dir.glob("*.raw"), key=lambda p: _natural_key(p.name))
    if max_files is not None and max_files >= 0:
        files = files[:max_files]
    _DIGITS = re.compile(r"(\d+)")
    def _infer_batch_index(p: Path) -> int:
        m = _DIGITS.findall(p.stem)
        return int(m[-1]) if m else 0
    jobs = [(_infer_batch_index(p), p) for p in files]
    jobs.sort(key=lambda t: t[0])
    return jobs

def infer_frames_per_raw_by_size(raw_path: Path, H: int, W: int) -> Optional[int]:
    bpp = 1
    frame_bytes = int(H) * int(W) * bpp
    if frame_bytes <= 0: return None
    sz = raw_path.stat().st_size
    if sz % frame_bytes == 0:
        return sz // frame_bytes
    return None

def read_first_nonzero_frame(raw_path: Path, H: int, W: int, frames_hint: Optional[int]=None, max_scan:int=50) -> Optional[np.ndarray]:
    """Try hard to get a non-zero frame for preview without relying strictly on N."""
    frame_bytes = int(H) * int(W)
    # 1) Try direct first H*W
    try:
        with open(raw_path, "rb") as f:
            data = f.read(frame_bytes)
        if len(data) >= frame_bytes:
            fr = np.frombuffer(data[:frame_bytes], dtype=np.uint8).reshape(H, W)
            if np.any(fr): return fr
    except Exception:
        pass
    # 2) Mmap-scan with (N,H,W)
    N = frames_hint or infer_frames_per_raw_by_size(raw_path, H, W) or 0
    if N > 0:
        try:
            mm = np.memmap(raw_path, mode="r", dtype=np.uint8, shape=(N, H, W))
            for i in range(min(N, max_scan)):
                fr = np.array(mm[i], copy=True)
                if np.any(fr): del mm; return fr
            del mm
        except Exception:
            pass
    # 3) Try swapped H/W
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

def auto_contrast_u8(img: np.ndarray, lo_p=1.0, hi_p=99.0) -> np.ndarray:
    """Stretch contrast to avoid 'all black' preview; safe if image is constant."""
    arr = img.astype(np.float32, copy=False)
    lo = np.percentile(arr, lo_p)
    hi = np.percentile(arr, hi_p)
    if hi <= lo:
        lo, hi = arr.min(), arr.max()
        if hi <= lo:  # truly flat
            return np.zeros_like(img, dtype=np.uint8)
    scaled = (arr - lo) * (255.0 / max(1e-6, (hi - lo)))
    return np.clip(scaled, 0, 255).astype(np.uint8)

class ROIGridUI:
    """Draggable 4-point ROI selector with live grid; ENTER saves JSON; Q save-only.
       Keys: F=4-pt, drag handles, W/S rows±, D/A cols±, G toggle boxes, ENTER save, Q save-only"""
    def __init__(self, img: np.ndarray, out_json: Path, rows=8, cols=9,
                 cell_w_um=100.0, cell_h_um=80.0, row_gap_um=20.0, col_gap_um=0.0):
        import matplotlib.pyplot as plt
        self.img = img
        self.out_json = out_json
        self.rows, self.cols = int(rows), int(cols)
        self.cell_w_um, self.cell_h_um = float(cell_w_um), float(cell_h_um)
        self.row_gap_um, self.col_gap_um = float(row_gap_um), float(col_gap_um)
        self.corners: Optional[np.ndarray] = None
        self.handles: list[Circle] = []
        self.drag_idx: Optional[int] = None
        self.show_boxes = True
        self._artists = []
        self._temp_clicks: list[tuple[float,float]] = []
        self.proceed = False

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img, cmap="gray"); self.ax.set_axis_off()
        self.ax.set_title("F=4-pt | drag | W/S rows | D/A cols | G boxes | ENTER save | Q save-only")
        self.cid_k = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.cid_p = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid_m = self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        self.cid_r = self.fig.canvas.mpl_connect('button_release_event', self._on_release)

    # -- events --
    def _on_key(self, e):
        k = (e.key or "").lower()
        if k=='f':
            self._temp_clicks.clear()
            print("[ROI] 4-point mode: click four corners (any order).")
        elif k=='w':
            self.rows += 1; self._redraw()
        elif k=='s':
            self.rows = max(1, self.rows-1); self._redraw()
        elif k=='d':
            self.cols += 1; self._redraw()
        elif k=='a':
            self.cols = max(1, self.cols-1); self._redraw()
        elif k=='g':
            self.show_boxes = not self.show_boxes; self._redraw()
        elif k=='enter':
            if self.corners is None:
                print("[ROI] Set four corners before saving."); return
            self._save_roi(); self.proceed = True; plt.close(self.fig)
        elif k=='q':
            self._save_roi(); self.proceed = False; plt.close(self.fig)

    def _on_click(self, e):
        if e.inaxes!=self.ax or e.xdata is None: return
        if self.corners is not None and self.handles:
            for i,h in enumerate(self.handles):
                contains,_ = h.contains(e)
                if contains: self.drag_idx = i; return
        if len(self._temp_clicks) < 4:
            self._temp_clicks.append((e.xdata, e.ydata)); self._draw_temp()
            if len(self._temp_clicks)==4:
                self.corners = np.array(self._order_corners(self._temp_clicks), float)
                self._install_handles(); self._temp_clicks.clear(); self._redraw()

    def _on_move(self, e):
        if self.drag_idx is None or self.corners is None: return
        if e.inaxes!=self.ax or e.xdata is None: return
        self.corners[self.drag_idx] = [e.xdata, e.ydata]
        self.handles[self.drag_idx].center = (e.xdata, e.ydata)
        self._redraw(lite=True)

    def _on_release(self, e): self.drag_idx=None

    # -- internals --
    def _draw_temp(self):
        keep=[]; 
        for tag,art in self._artists:
            if tag=='temp':
                try: art.remove()
                except: pass
            else: keep.append((tag,art))
        self._artists=keep
        for (x,y) in self._temp_clicks:
            dot, = self.ax.plot(x,y,'o',ms=4,mec='cyan',mfc='none',ls='')
            self._artists.append(('temp', dot))
        self.fig.canvas.draw_idle()

    @staticmethod
    def _order_corners(points: List[Tuple[float,float]]):
        pts = np.array(points,float)
        cx,cy = pts[:,0].mean(), pts[:,1].mean()
        ang = np.arctan2(pts[:,1]-cy, pts[:,0]-cx)
        idx = np.argsort(ang); pts = pts[idx]
        tl_idx = np.lexsort((pts[:,0], pts[:,1]))[0]
        return np.roll(pts, -tl_idx, axis=0)  # TL,TR,BR,BL

    def _redraw(self, lite=False):
        keep=[]; 
        for tag,art in self._artists:
            if tag=='grid':
                try: art.remove()
                except: pass
            else: keep.append((tag,art))
        self._artists=keep
        if self.corners is None:
            self.fig.canvas.draw_idle(); return
        TL,TR,BR,BL = self.corners; A = TR-TL; B = BL-TL
        BRc = TL + A + B
        xs=[TL[0],TR[0],BRc[0],BL[0],TL[0]]; ys=[TL[1],TR[1],BRc[1],BL[1],TL[1]]
        l, = self.ax.plot(xs,ys,c='cyan',lw=1,alpha=0.9); self._artists.append(('grid', l))

        if self.show_boxes:
            total_w_um = self.cols*self.cell_w_um
            total_h_um = self.rows*self.cell_h_um + (self.rows-1)*self.row_gap_um
            half_w = self.cell_w_um/2; half_h = self.cell_h_um/2
            for r in range(self.rows):
                v_center_um = r*(self.cell_h_um+self.row_gap_um)+half_h
                v0=(v_center_um-half_h)/total_h_um; v1=(v_center_um+half_h)/total_h_um
                for c in range(self.cols):
                    u_center_um = c*(self.cell_w_um+self.col_gap_um)+half_w
                    u0=(u_center_um-half_w)/total_w_um; u1=(u_center_um+half_w)/total_w_um
                    def M(u,v): return TL + u*A + v*B
                    p1,p2,p3,p4 = M(u0,v0),M(u1,v0),M(u1,v1),M(u0,v1)
                    xs=[p1[0],p2[0],p3[0],p4[0],p1[0]]; ys=[p1[1],p2[1],p3[1],p4[1],p1[1]]
                    l, = self.ax.plot(xs,ys,c='cyan',lw=0.8,alpha=0.85)
                    self._artists.append(('grid', l))
        self._install_handles(); self.fig.canvas.draw_idle()

    def _install_handles(self):
        for h in self.handles:
            try: h.remove()
            except: pass
        self.handles.clear()
        if self.corners is None: return
        for (x,y) in self.corners:
            circ = Circle((x,y), radius=5, facecolor='none', edgecolor='yellow', lw=1.2)
            self.ax.add_patch(circ); self.handles.append(circ)

    def _save_roi(self):
        if self.corners is None: return
        TL,TR,BR,BL = self.corners; A=TR-TL; B=BL-TL
        total_w_um = self.cols*self.cell_w_um
        total_h_um = self.rows*self.cell_h_um + (self.rows-1)*self.row_gap_um
        half_w=self.cell_w_um/2; half_h=self.cell_h_um/2
        centers=[]; polys=[]
        for r in range(self.rows):
            v_center_um = r*(self.cell_h_um+self.row_gap_um)+half_h
            v0=(v_center_um-half_h)/total_h_um; v1=(v_center_um+half_h)/total_h_um
            for c in range(self.cols):
                u_center_um = c*(self.cell_w_um+self.col_gap_um)+half_w
                u0=(u_center_um-half_w)/total_w_um; u1=(u_center_um+half_w)/total_w_um
                def M(u,v): return TL + u*A + v*B
                p1,p2,p3,p4 = M(u0,v0),M(u1,v0),M(u1,v1),M(u0,v1)
                cx,cy = M((u0+u1)/2,(v0+v1)/2)
                centers.append([float(cx),float(cy)])
                polys.append([[float(p1[0]),float(p1[1])],
                              [float(p2[0]),float(p2[1])],
                              [float(p3[0]),float(p3[1])],
                              [float(p4[0]),float(p4[1])]])
        payload = dict(
            rows=int(self.rows), cols=int(self.cols),
            cell_w_um=float(self.cell_w_um), cell_h_um=float(self.cell_h_um),
            row_gap_um=float(self.row_gap_um), col_gap_um=float(self.col_gap_um),
            corners_xy=[TL.tolist(),TR.tolist(),BR.tolist(),BL.tolist()],
            centers_xy_f=centers, cell_polygons=polys
        )
        with open(self.out_json,'w',encoding='utf-8') as f:
            json.dump(payload,f,indent=2)
        print(f"[ROI] Saved → {self.out_json}")

# ============================== original v3 binning pipeline ==============================

def load_roi(cfg_path: Path):
    if not cfg_path.exists():
        raise SystemExit(f"Missing ROI json: {cfg_path}")
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

class TiffOut:
    def __init__(self, path: Path, bigtiff: bool):
        self._tw = tiff.TiffWriter(str(path), bigtiff=bigtiff)
    def write_page(self, arr: np.ndarray):
        self._tw.write(arr, contiguous=True, photometric="minisblack")
    def close(self):
        self._tw.close()

class Prefetcher(threading.Thread):
    daemon = True
    def __init__(self, jobs: List[Tuple[int, Path]], H: int, W: int, N: int,
                 start_idx: int, end_idx: Optional[int], batch: int, q: "queue.Queue"):
        super().__init__()
        self.jobs, self.H, self.W, self.N = jobs, H, W, N
        self.start_idx = start_idx   # renamed to avoid Thread.start() clash
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
    rows, cols, centers, polys = load_roi(cm.roi_config_path())
    jobs = list_raw_files(cm.in_dir, cm.max_files)
    if not jobs: raise SystemExit(f"No RAW files found under {cm.in_dir}")
    first_path = jobs[0][1]

    # >>> Fix "seed all zero": robust first-frame only for metadata/preview
    inferN = infer_frames_per_raw_by_size(first_path, cm.height, cm.width) or cm.frames_per_raw
    fr0 = read_first_nonzero_frame(first_path, cm.height, cm.width, frames_hint=inferN) or np.zeros((cm.height, cm.width), dtype=np.uint8)
    first_frame = fr0.astype(np.float32)
    # <<<

    pos, starts, areas = build_packed_union_indices(cm.height, cm.width, centers, cm.box_size)
    B = centers.shape[0]
    total_frames = len(jobs) * cm.frames_per_raw
    s = max(0, cm.start); e = min(total_frames, cm.end) if (cm.end is not None and cm.end >= 0) else total_frames
    T = max(0, e - s)
    if T == 0: raise SystemExit("Frame range empty: nothing to process.")
    traces = np.empty((B, T), dtype=np.float32)

    exp_dir = cm.exp_dir()
    if cm.tiff_out is None:
        tiff_path = exp_dir / ("stack_binned.tif" if cm.tiff_mode == "binned" else "stack_raw.tif")
    else:
        tiff_path = cm.tiff_out

    if cm.tiff_mode == "binned":
        use_bigtiff = False
    else:
        bytes_est = T * cm.height * cm.width
        use_bigtiff = (bytes_est >= int(cm.bigtiff_threshold_gb * (1024**3)))

    exp_dir.mkdir(parents=True, exist_ok=True)
    tw = TiffOut(tiff_path, bigtiff=use_bigtiff)

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
                    next_write += ba.shape[0]
                break
            seq_start, ba = item
            pending[seq_start] = ba
            while next_write in pending:
                ba2 = pending.pop(next_write)
                written = _consume_batch(ba2, next_write, traces, rows, cols, cm, pos, starts, areas, tw, log, written, T)
                next_write += ba2.shape[0]
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
    npz_path = exp_dir / "exp_block_data.npz"
    np.savez_compressed(npz_path, **out)
    log(f"[DONE] NPZ -> {npz_path} | TIFF -> {tiff_path} | traces {traces.shape} | pages {T} ({cm.tiff_mode})")
    return str(tiff_path), str(npz_path)

def _consume_batch(batch_arr: np.ndarray, seq_start: int, traces: np.ndarray, rows: int, cols: int,
                   cm: CompressionManager, pos: np.ndarray, starts: np.ndarray, areas: np.ndarray,
                   tw: TiffOut, log, written: int, T: int) -> int:
    means_batch = bin_batch(batch_arr, pos, starts, areas)
    Bf = means_batch.shape[0]
    s_local = (seq_start - cm.start)
    e_local = s_local + Bf
    traces[:, s_local:e_local] = means_batch.T
    if cm.tiff_mode == "binned":
        binned_frames = means_batch.reshape(Bf, rows, cols).astype(np.float32, copy=False)
        for k in range(Bf): tw.write_page(binned_frames[k])
    else:
        for k in range(Bf): tw.write_page(batch_arr[k])
    written = e_local
    if (written % 200) == 0 or (written == T): log(f"[{written}/{T}] frames processed…")
    return written

def bin_batch(batch_arr: np.ndarray, pos: np.ndarray, starts: np.ndarray, areas: np.ndarray) -> np.ndarray:
    Bf = batch_arr.shape[0]
    flat = batch_arr.reshape(Bf, -1)
    gathered = flat[:, pos]
    sums = np.add.reduceat(gathered, starts, axis=1)
    means = (sums / areas)
    return means

# ============================== app_gui ==============================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

def _safe_int(text: str) -> Optional[int]:
    t = text.strip()
    if t == "": return None
    return int(t)

def _robust_seed_from_raw(in_dir: Path, H: int, W: int, N_hint: Optional[int]) -> np.ndarray:
    """Return a displayable uint8 image from the first few RAW files, with auto-contrast."""
    jobs = list_raw_files(in_dir, max_files=5)
    if not jobs:
        raise RuntimeError("No RAW files found to create a seed image.")
    for _, rp in jobs:
        N_guess = N_hint or infer_frames_per_raw_by_size(rp, H, W)
        fr = read_first_nonzero_frame(rp, H, W, frames_hint=N_guess)
        if fr is not None:
            return auto_contrast_u8(fr)
    raise RuntimeError("Could not find a non-zero frame in the first few RAW files. Check height/width.")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAW → (Binned TIFF + NPZ) [v3 + ROI]")
        self.geometry("860x700")
        self.resizable(True, True)
        self.in_dir = tk.StringVar()
        self.roi_json = tk.StringVar()
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
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}
        frm = ttk.Frame(self); frm.pack(fill="both", expand=True, **pad)

        ttk.Label(frm, text="RAW input folder").grid(row=0, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.in_dir, width=62).grid(row=0, column=1, sticky="we")
        ttk.Button(frm, text="Browse…", command=self._pick_in).grid(row=0, column=2, **pad)

        ttk.Label(frm, text="ROI JSON").grid(row=1, column=0, sticky="e")
        ttk.Entry(frm, textvariable=self.roi_json, width=62).grid(row=1, column=1, sticky="we")
        ttk.Button(frm, text="Browse JSON…", command=self._pick_roi_json).grid(row=1, column=2, **pad)

        rframe = ttk.Frame(frm); rframe.grid(row=2, column=0, columnspan=3, sticky="w", **pad)
        ttk.Button(rframe, text="Open ROI selector (from RAW)", command=self._open_roi_selector_from_raw).pack(side="left")

        g = ttk.LabelFrame(frm, text="Geometry & RAW layout")
        g.grid(row=3, column=0, columnspan=3, sticky="we", **pad)
        for i in range(4): g.columnconfigure(i, weight=1)
        ttk.Label(g, text="Height").grid(row=0, column=0, sticky="e"); ttk.Entry(g, textvariable=self.height, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(g, text="Width").grid(row=0, column=2, sticky="e");  ttk.Entry(g, textvariable=self.width,  width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(g, text="Frames per RAW (N)").grid(row=1, column=0, sticky="e"); ttk.Entry(g, textvariable=self.frames_per_raw, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(g, text="Frames per file (compat)").grid(row=1, column=2, sticky="e"); ttk.Entry(g, textvariable=self.frames_per_file, width=8).grid(row=1, column=3, sticky="w")

        b = ttk.LabelFrame(frm, text="Binning & Output")
        b.grid(row=4, column=0, columnspan=3, sticky="we", **pad)
        for i in range(4): b.columnconfigure(i, weight=1)
        ttk.Label(b, text="Box size").grid(row=0, column=0, sticky="e"); ttk.Entry(b, textvariable=self.box, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(b, text="TIFF mode").grid(row=0, column=2, sticky="e")
        ttk.Combobox(b, textvariable=self.tiff_mode, values=["binned", "raw"], width=10, state="readonly").grid(row=0, column=3, sticky="w")
        ttk.Label(b, text="TIFF out (optional)").grid(row=1, column=0, sticky="e")
        ttk.Entry(b, textvariable=self.tiff_out, width=48).grid(row=1, column=1, sticky="we", columnspan=2)
        ttk.Button(b, text="Choose…", command=self._pick_tiff_out).grid(row=1, column=3, sticky="w", **pad)

        r = ttk.LabelFrame(frm, text="Range & Performance")
        r.grid(row=5, column=0, columnspan=3, sticky="we", **pad)
        for i in range(4): r.columnconfigure(i, weight=1)
        ttk.Label(r, text="Start").grid(row=0, column=0, sticky="e"); ttk.Entry(r, textvariable=self.start, width=8).grid(row=0, column=1, sticky="w")
        ttk.Label(r, text="End (exclusive)").grid(row=0, column=2, sticky="e"); ttk.Entry(r, textvariable=self.end, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(r, text="Max files (quick test)").grid(row=1, column=0, sticky="e"); ttk.Entry(r, textvariable=self.max_files, width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(r, text="Batch (frames/chunk)").grid(row=1, column=2, sticky="e"); ttk.Entry(r, textvariable=self.batch, width=8).grid(row=1, column=3, sticky="w")
        ttk.Label(r, text="Prefetch queue depth").grid(row=2, column=0, sticky="e"); ttk.Entry(r, textvariable=self.prefetch, width=8).grid(row=2, column=1, sticky="w")
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

    def _pick_roi_json(self):
        f = filedialog.askopenfilename(title="Select ROI JSON", filetypes=[("JSON","*.json")])
        if f: self.roi_json.set(f)

    def _pick_tiff_out(self):
        f = filedialog.asksaveasfilename(title="Select TIFF output path", defaultextension=".tif",
                                         filetypes=[("TIFF", "*.tif *.tiff")])
        if f: self.tiff_out.set(f)

    def _open_roi_selector_from_raw(self):
        try:
            in_dir = Path(self.in_dir.get())
            if not in_dir.exists():
                messagebox.showerror("Missing input", "Pick RAW input folder first."); return

            # Where to save JSON?
            if self.roi_json.get().strip():
                out_json = Path(self.roi_json.get())
                out_dir = out_json.parent
            else:
                d = filedialog.askdirectory(title="Select folder to save ROI JSON")
                if not d: return
                out_dir = Path(d)
                out_json = out_dir / "roi_grid_config.json"
                self.roi_json.set(str(out_json))
            out_dir.mkdir(parents=True, exist_ok=True)

            # Build robust seed (auto-contrast) from RAW
            img_u8 = _robust_seed_from_raw(in_dir, int(self.height.get()), int(self.width.get()), int(self.frames_per_raw.get()))
            # Open embedded ROI UI
            ui = ROIGridUI(img_u8, out_json)
            print("[ROI UI] F=4-pt; drag; W/S rows; D/A cols; G toggle boxes; ENTER save; Q save-only")
            plt.show()
            if out_json.exists():
                self._log(f"[ROI] JSON saved: {out_json}")
                messagebox.showinfo("ROI JSON", f"Saved:\n{out_json}")
            else:
                self._log("[ROI] JSON was not saved. Press ENTER or Q to save.")
        except Exception as e:
            self._log(f"[ERROR] ROI selector: {e}\n{traceback.format_exc()}")
            messagebox.showerror("ROI selector error", str(e))

    def _log(self, msg: str):
        self.logw.insert("end", msg + "\n"); self.logw.see("end"); self.update_idletasks()

    def _run(self):
        def _as_int_or_none(text: str) -> Optional[int]:
            t = text.strip()
            if t == "": return None
            return int(t)
        try:
            roi_json_path = Path(self.roi_json.get()) if self.roi_json.get().strip() else None
            if roi_json_path is None:
                messagebox.showwarning("ROI missing", "Please create or select ROI JSON first."); return
            cm = CompressionManager(
                height=int(self.height.get()), width=int(self.width.get()),
                frames_per_raw=int(self.frames_per_raw.get()), frames_per_file=int(self.frames_per_file.get()),
                box_size=int(self.box.get()),
                in_dir=Path(self.in_dir.get()),
                roi_json_file=roi_json_path,
                tiff_out=(Path(self.tiff_out.get()) if self.tiff_out.get().strip() else None),
                tiff_mode=self.tiff_mode.get(), batch=int(self.batch.get()),
                start=int(self.start.get()), end=_as_int_or_none(self.end.get()),
                max_files=_as_int_or_none(self.max_files.get()),
                fps=(float(self.fps.get()) if self.fps.get().strip() else 5000.0),
                prefetch_batches=int(self.prefetch.get()),
            )
        except Exception as e:
            messagebox.showerror("Invalid parameters", str(e)); return

        if not cm.in_dir.exists():
            messagebox.showerror("Missing input", f"Input folder not found: {cm.in_dir}"); return
        if not cm.roi_config_path().exists():
            messagebox.showwarning("ROI json missing", f"ROI json not found at {cm.roi_config_path()}.\nUse the ROI selector to create one."); return

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
