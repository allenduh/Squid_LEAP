#!/usr/bin/env python3
# analyze_experiment_end_to_end_v5.py
#
# End-to-end pipeline for ONE experiment, using non-fitting statistics:
#  - Method 1 (Histogram Peaks): detrend + (optional) smoothing → histogram of values
#    → find two peaks (OFF/ON) → Δ = peak2_center - peak1_center;
#    Noise σ estimated from FWHM of each peak (FWHM ≈ 2.355σ).
#  - Method 1.5 (Combined "Peak Bands"): Use FWHM-derived bands around the two peaks
#    to select only plateau samples (excludes rise/fall) → Δ from band means;
#    pooled σ from the same bands. (Default)
#  - Method 2 (Cycle-wise Max–Min): segment by period, take robust (trimmed) max–min
#    inside cycles while ignoring middle quantiles; this avoids edges but can
#    overestimate if there are spikes. We conservatively use 10th–90th percentiles.
#
# Keeps the ROI grid UI from V4 and produces a multi-page PDF that includes:
#  - ROI overlay
#  - Heatmaps (Δ, SNR, Δ/Off)
#  - Traces grid
#  - An INFO page: experiment path, parameters, and aggregate Signal/Noise numbers.
#
# Dependencies: numpy, matplotlib, pillow, scipy
# (No scikit-learn required.)
#
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Sequence
import re, json, sys, time

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import butter, filtfilt, find_peaks, peak_widths, iirnotch

from scipy.stats import iqr


# ------------------------------ Dialog ------------------------------

LAST_FOLDER_FILE = Path.home() / ".end2end_last_folder.txt"

def _get_last_folder() -> Path:
    try:
        p = Path(LAST_FOLDER_FILE.read_text().strip())
        if p.exists():
            return p
    except Exception:
        pass
    return Path.home() / "Downloads"

def _set_last_folder(p: Path) -> None:
    try:
        LAST_FOLDER_FILE.write_text(str(p), encoding="utf-8")
    except Exception:
        pass

def pick_folder_dialog(title: str = "Pick experiment OR parent folder") -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        p = filedialog.askdirectory(initialdir=str(_get_last_folder()), title=title)
        if not p:
            raise RuntimeError("No folder chosen")
        return Path(p)
    except Exception:
        return Path.cwd()

# ------------------------------ Experiment discovery ------------------------------
EXPNAME_RE = re.compile(r"^(?P<job>J\d+)[^_]*_(?P<dose>\d+).*?__(?P<ma>\d+)mA_(?P<mv>\d+)mV_.*$", re.IGNORECASE)
BMP_RE     = re.compile(r"^(\d+)_(\d+)\.bmp$", re.IGNORECASE)
BMP_RE = re.compile(r'(?i)^batch[_-]?(\d+)[_-]frame[_-]?(\d+)\.bmp$')

BMP_PATTERNS = [
    re.compile(r'(?i)^batch[_-]?(\d+)[_-]frame[_-]?(\d+)\.bmp$'),
    re.compile(r'(?i)^(\d+)[_-](\d+)\.bmp$'),
]

BMP_PATTERNS = [
re.compile(r"(?i)^batch[_-]?(\d+)[_-]frame[_-]?(\d+)\.bmp$"), # batch-style
re.compile(r"(?i)^(\d+)[_-](\d+)\.bmp$"), # plain n1_sep_n2
]

def _natural_key(name: str):
    # makes 2 < 10 for mixed text+digits
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]
@dataclass
class ExperimentPath:
    base: Path
    job: str
    dose: int
    mA: int
    mV: int

def parse_exp_dirname(p: Path) -> Optional[ExperimentPath]:
    m = EXPNAME_RE.match(p.name)
    if not m: return None
    try:
        return ExperimentPath(
            base=p,
            job=m.group("job"),
            dose=int(m.group("dose")),
            mA=int(m.group("ma")),
            mV=int(m.group("mv")),
        )
    except Exception:
        return None

def is_experiment_dir(p: Path) -> bool:
    if not p.is_dir() or p.name.startswith('.'):
        return False
    if (p / "0").exists():
        return True
    for q in p.iterdir():
        if q.is_dir() and q.name.isdigit():
            return True
    return any(f.is_file() and f.suffix.lower()==".bmp" for f in p.iterdir())

def list_experiments_under(parent: Path) -> List[ExperimentPath]:
    exps: List[ExperimentPath] = []
    for cand in sorted([d for d in parent.iterdir() if d.is_dir() and not d.name.startswith(".")]):
        if is_experiment_dir(cand):
            meta = parse_exp_dirname(cand) or ExperimentPath(cand, "NA", -1, 999999, 100)
            exps.append(meta)
    return exps

def choose_experiment(path: Path) -> ExperimentPath:
    if path.is_dir() and is_experiment_dir(path):
        meta = parse_exp_dirname(path) or ExperimentPath(path, "NA", -1, 999999, 100)
        print(f"[Experiment] {meta.base}")
        return meta
    exps = list_experiments_under(path)
    if not exps:
        raise SystemExit(f"No experiment folders found under: {path}")
    exps.sort(key=lambda e: (e.job, e.dose, e.mA, e.mV, e.base.name))
    job0, dose0 = exps[0].job, exps[0].dose
    group = [e for e in exps if e.job==job0 and e.dose==dose0]
    group.sort(key=lambda e: (e.mA, e.mV))
    chosen = group[0]
    print(f"[Parent] Selected base (lowest mA for {job0}_{dose0}): {chosen.base}")
    return chosen

# ------------------------------ Image I/O ------------------------------

# def find_first_bmp(exp: Path) -> Path:
#     subdirs = [d for d in exp.iterdir() if d.is_dir() and not d.name.startswith(".")]
#     best_sub = exp/"0" if (exp/"0").exists() else None
#     if best_sub is None:
#         nums = [d for d in subdirs if d.name.isdigit()]
#         if nums:
#             best_sub = sorted(nums, key=lambda p:int(p.name))[0]
#     if best_sub is None:
#         best_sub = exp
#     cands = []
#     for f in best_sub.iterdir():
#         if f.is_file() and f.suffix.lower()==".bmp":
#             m = BMP_RE.match(f.name)
#             if m:
#                 frame, glob = int(m.group(1)), int(m.group(2))
#                 cands.append((glob, frame, f))
#     if not cands:
#         raise FileNotFoundError(f"No BMP files found in {best_sub}")
#     cands.sort(key=lambda t: (t[0], t[1]))
#     return cands[0][2]


def find_first_bmp(exp: Path) -> Path:
    """
    Return the earliest BMP frame under the experiment directory.
    Looks in (priority): bmp_export/, 0/, numeric subdirs, then root; searches recursively.
    Accepts both 'batch_##_frame_##.bmp' and '##_##.bmp' patterns via BMP_PATTERNS.
    Falls back to the first '*.bmp' if none match the patterns.
    """
    # Build prioritized bases
    bases: list[Path] = []
    if (exp / "bmp_export").exists():
        bases.append(exp / "bmp_export")
    if (exp / "0").exists():
        bases.append(exp / "0")
    nums = [d for d in exp.iterdir() if d.is_dir() and d.name.isdigit()]
    bases.extend(sorted(nums, key=lambda p: int(p.name)))
    bases.append(exp)

    # First try: pattern-aware match (batch/frame or plain digits)
    cands: list[tuple[int,int,Path]] = []
    for base in bases:
        for f in sorted(base.rglob("*.bmp"), key=lambda p: _natural_key(p.name)):
            for pat in BMP_PATTERNS:
                m = pat.match(f.name)
                if m:
                    b, fr = int(m.group(1)), int(m.group(2))
                    cands.append((b, fr, f))
                    break
    if cands:
        cands.sort(key=lambda t: (t[0], t[1]))
        return cands[0][2]

    # Fallback: any .bmp (in case names are non-standard)
    any_bmp = next((exp.rglob("*.bmp")), None)
    if any_bmp:
        return any_bmp

    raise FileNotFoundError(f"No BMP files found under {exp}")



# def list_all_frames(exp: Path) -> List[Path]:
#     tuples = []
#     subs = [d for d in exp.iterdir() if d.is_dir() and d.name.isdigit()]
#     subs.sort(key=lambda p:int(p.name))
#     search = subs if subs else [exp]
#     for d in search:
#         for f in d.iterdir():
#             if f.is_file() and f.suffix.lower()==".bmp":
#                 m = BMP_RE.match(f.name)
#                 if m:
#                     frame, glob = int(m.group(1)), int(m.group(2))
#                     tuples.append((glob, frame, f))
#     if not tuples:
#         raise SystemExit(f"No BMP frames found in {exp}")
#     tuples.sort(key=lambda t:(t[0], t[1]))
#     return [t[2] for t in tuples]

def list_all_frames(exp: Path) -> List[Path]:
    """
    Return all BMP frames sorted by (batch, frame) numerically.
    Accepts either 'batch_###_frame_###.bmp' or '###_###.bmp'.
    Searches recursively in exp/0, numeric subdirs, or exp.
    """
    tuples = []

    bases = []
    if (exp / "0").exists():
        bases.append(exp / "0")
    nums = [d for d in exp.iterdir() if d.is_dir() and d.name.isdigit()]
    if nums:
        bases.extend(sorted(nums, key=lambda p: int(p.name)))
    bases.append(exp)

    for base in bases:
        for f in sorted(base.rglob("*.bmp"), key=lambda p: _natural_key(p.name)):
            for pat in BMP_PATTERNS:
                m = pat.match(f.name)
                if m:
                    batch, frame = int(m.group(1)), int(m.group(2))
                    tuples.append((batch, frame, f))
                    break

    if not tuples:
        raise SystemExit(f"No BMP frames found under {exp}")

    tuples.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in tuples]


def load_bmp_gray(path: Path) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim==2:
        return arr.astype(np.float32)
    if arr.ndim==3 and arr.shape[2]==4:
        arr = arr[...,:3]
    if arr.ndim==3 and arr.shape[2]==3:
        r,g,b = arr[...,0], arr[...,1], arr[...,2]
        arr = 0.299*r + 0.587*g + 0.114*b
    return arr.astype(np.float32)

# ------------------------------ ROI UI ------------------------------
class ROIGridUI:
    def __init__(self, img: np.ndarray, out_json: Path,
                 rows=8, cols=9,
                 cell_w_um=100.0, cell_h_um=80.0, row_gap_um=20.0, col_gap_um=0.0):
        self.img = img
        self.out_json = out_json
        self.rows, self.cols = int(rows), int(cols)
        self.cell_w_um  = float(cell_w_um)
        self.cell_h_um  = float(cell_h_um)
        self.row_gap_um = float(row_gap_um)
        self.col_gap_um = float(col_gap_um)

        self.corners: Optional[np.ndarray] = None  # TL,TR,BR,BL
        self.handles: List[Circle] = []
        self.drag_idx: Optional[int] = None
        self.show_boxes = True
        self._artists = []
        self._temp_clicks: List[Tuple[float,float]] = []
        self.proceed = False  # set True when ENTER pressed with valid ROI

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img, cmap="gray")
        self.ax.set_axis_off()
        self.ax.set_title(
            "F: four-point | drag handles | W/S rows | D/A cols | G boxes | ENTER extract | Q save-only"
        )
        self.cid_k = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.cid_p = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid_m = self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        self.cid_r = self.fig.canvas.mpl_connect('button_release_event', self._on_release)

    # ---- events ----
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
        elif k=='h':
            print("Keys: F=4-pt, drag handles, W/S rows, D/A cols, G boxes, ENTER extract, Q save-only")
        elif k=='enter':
            if self.corners is None:
                print("[ROI] Set four corners before extracting.")
                return
            self._save_roi()
            self.proceed = True
            plt.close(self.fig)
        elif k=='q':
            self._save_roi()
            self.proceed = False
            plt.close(self.fig)

    def _on_click(self, e):
        if e.inaxes!=self.ax or e.xdata is None: return
        # drag?
        if self.corners is not None and self.handles:
            for i,h in enumerate(self.handles):
                contains,_ = h.contains(e)
                if contains:
                    self.drag_idx = i; return
        # collect 4 clicks
        if len(self._temp_clicks) < 4:
            self._temp_clicks.append((e.xdata, e.ydata))
            self._draw_temp()
            if len(self._temp_clicks)==4:
                self.corners = np.array(self._order_corners(self._temp_clicks), float)
                self._install_handles(); self._temp_clicks.clear()
                self._redraw()

    def _on_move(self, e):
        if self.drag_idx is None or self.corners is None: return
        if e.inaxes!=self.ax or e.xdata is None: return
        self.corners[self.drag_idx] = [e.xdata, e.ydata]
        self.handles[self.drag_idx].center = (e.xdata, e.ydata)
        self._redraw(lite=True)

    def _on_release(self, e): self.drag_idx=None

    # ---- internals ----
    def _draw_temp(self):
        keep=[]
        for tag,art in self._artists:
            if tag=='temp':
                try: art.remove()
                except: pass
            else:
                keep.append((tag,art))
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
        idx = np.argsort(ang)
        pts = pts[idx]
        tl_idx = np.lexsort((pts[:,0], pts[:,1]))[0]
        pts = np.roll(pts, -tl_idx, axis=0)
        TL,TR,BR,BL = pts[0],pts[1],pts[2],pts[3]
        return TL,TR,BR,BL

    def _map_uv(self, TL, A, B, u, v): return TL + u*A + v*B

    def _redraw(self, lite=False):
        keep=[]
        for tag,art in self._artists:
            if tag=='grid':
                try: art.remove()
                except: pass
            else:
                keep.append((tag,art))
        self._artists=keep
        if self.corners is None:
            self.fig.canvas.draw_idle(); return

        TL,TR,BR,BL = self.corners
        A = TR-TL; B = BL-TL
        BRc = TL + A + B
        xs=[TL[0],TR[0],BRc[0],BL[0],TL[0]]; ys=[TL[1],TR[1],BRc[1],BL[1],TL[1]]
        l, = self.ax.plot(xs,ys,c='cyan',lw=1,alpha=0.9); self._artists.append(('grid', l))

        if self.show_boxes:
            total_w_um = self.cols*100.0
            total_h_um = self.rows*80.0 + (self.rows-1)*20.0
            half_w = 50.0; half_h = 40.0
            for r in range(self.rows):
                v_center_um = r*(80.0+20.0)+half_h
                v0 = (v_center_um-half_h)/total_h_um; v1=(v_center_um+half_h)/total_h_um
                for c in range(self.cols):
                    u_center_um = c*(100.0+0.0)+half_w
                    u0 = (u_center_um-half_w)/total_w_um; u1=(u_center_um+half_w)/total_w_um
                    def M(u,v): return TL + u*A + v*B
                    p1,p2,p3,p4 = M(u0,v0),M(u1,v0),M(u1,v1),M(u0,v1)
                    xs=[p1[0],p2[0],p3[0],p4[0],p1[0]]; ys=[p1[1],p2[1],p3[1],p4[1],p1[1]]
                    l, = self.ax.plot(xs,ys,c='cyan',lw=0.8,alpha=0.85)
                    self._artists.append(('grid', l))
        self._install_handles()
        self.fig.canvas.draw_idle()

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
        if self.corners is None:
            print("[ROI] No corners set; nothing saved."); return
        TL,TR,BR,BL = self.corners
        A=TR-TL; B=BL-TL
        total_w_um = self.cols*100.0
        total_h_um = self.rows*80.0 + (self.rows-1)*20.0
        half_w=50.0; half_h=40.0
        centers=[]; polys=[]
        for r in range(self.rows):
            v_center_um = r*(80.0+20.0)+half_h
            v0=(v_center_um-half_h)/total_h_um; v1=(v_center_um+half_h)/total_h_um
            for c in range(self.cols):
                u_center_um = c*(100.0+0.0)+half_w
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
            cell_w_um=100.0, cell_h_um=80.0, row_gap_um=20.0, col_gap_um=0.0,
            corners_xy=[TL.tolist(),TR.tolist(),BR.tolist(),BL.tolist()],
            centers_xy_f=centers, cell_polygons=polys
        )
        with open(self.out_json,'w',encoding='utf-8') as f:
            json.dump(payload,f,indent=2)
        print(f"[ROI] Saved → {self.out_json}")


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def notch_filter(data,notch_freq,quality,fs, order):
    y = data
    for i in range(1,order):
      b,a=iirnotch(60*i,quality,fs)
      y = filtfilt(b,a,y)
    return y
    
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

#butterworth highpass
def butter_highpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=1):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

#butterworth lowpass
def butter_lowpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=3):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
# ------------------------------ Extraction (packed indices) ------------------------------

def _build_packed_union_indices(H:int, W:int, centers:np.ndarray, bs:int):
    h = bs // 2
    cx, cy = centers[:,0], centers[:,1]
    x0 = np.clip(cx - h, 0, W);  y0 = np.clip(cy - h, 0, H)
    x1 = np.clip(x0 + bs, 0, W); y1 = np.clip(y0 + bs, 0, H)

    B = centers.shape[0]
    pos_list = []
    starts   = np.empty(B, dtype=np.int64)
    areas    = np.empty(B, dtype=np.int64)
    cur = 0
    for b in range(B):
        xa, xb = int(x0[b]), int(x1[b])
        ya, yb = int(y0[b]), int(y1[b])
        starts[b] = cur
        if xb > xa and yb > ya:
            yy, xx = np.mgrid[ya:yb, xa:xb]
            p = (yy * W + xx).ravel()
            pos_list.append(p)
            cur += p.size
            areas[b] = p.size
        else:
            areas[b] = 0
    pos = np.concatenate(pos_list) if pos_list else np.array([], dtype=np.int64)
    areas = np.maximum(areas, 1)  # guard divide-by-zero
    return pos.astype(np.int64), starts, areas.astype(np.float32)

def extract_traces(exp, roi_json: Path, box_sizes: Sequence[int]=(32,)) -> Dict[str, Any]:
    cfg = json.loads(Path(roi_json).read_text())
    rows, cols = int(cfg['rows']), int(cfg['cols'])
    centers = np.asarray(cfg.get('centers_xy_f') or cfg.get('centers') or cfg.get('cell_centers'))
    if centers is None or centers.shape != (rows*cols, 2):
        raise RuntimeError("centers missing or wrong shape in ROI json")
    centers = np.rint(centers).astype(np.int32)

    if not box_sizes:
        raise ValueError("box_sizes must have one element")
    if len(box_sizes) != 1:
        raise ValueError(f"fast path supports a single box size; got {box_sizes}")
    bs = int(box_sizes[0])

    frames = list_all_frames(exp.base)
    T = len(frames)
    first = load_bmp_gray(frames[0])            
    H, W = first.shape
    video = np.stack([load_bmp_gray(fp) for fp in frames], axis=0)  # (T,H,W)

    pos, starts, areas = _build_packed_union_indices(H, W, centers, bs)

    flat = video.reshape(T, -1)                       
    gathered = flat[:, pos].astype(np.uint32, copy=False)  
    sums = np.add.reduceat(gathered, starts, axis=1)       
    means = (sums / areas[None, :]).astype(np.float32)     
    means = means.T                                        

    out = dict(
        experiment_dir=str(exp.base),
        first_frame_path=str(frames[0]),
        first_frame=first.astype(np.float32),
        rows=rows, cols=cols,
        cell_w_um=100.0, cell_h_um=80.0, row_gap_um=20.0, col_gap_um=0.0,
        box_sizes=np.array([bs], dtype=np.int32),
        centers=centers,
        polygons=np.array(cfg['cell_polygons'], dtype=np.float32),
        trace_boxes=means[None, ...],  # (1,B,T)
        frame_count=T,
        job_id=exp.job, dose=exp.dose, led_mA=exp.mA, perturb_mV=exp.mV,
        timestamp=exp.base.name.split('_')[-1] if '_' in exp.base.name else "",
        block_labels=np.array([f"r{r}_c{c}" for r in range(rows) for c in range(cols)], dtype=object),
        block_rc=np.array([[r,c] for r in range(rows) for c in range(cols)], dtype=int),
    )
    np.savez_compressed(exp.base / "exp_block_data.npz", **out)
    print(f"[Extract] Saved traces → {exp.base/'exp_block_data.npz'}")
    return out

# ------------------------------ Signal/Noise methods (V5) ------------------------------

def detrend_linear_(y: np.ndarray) -> np.ndarray:
    """Remove best-fit line y ≈ a + b t to flatten drift."""
    y = np.asarray(y, float).ravel()
    T = y.size
    t = np.arange(T, dtype=float)
    A = np.vstack([np.ones(T), t]).T  # [1, t]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    return  butter_highpass_filter(y - (a + b*t), 0.0001, fs=10, order=1)

def detrend_linear(y: np.ndarray) -> np.ndarray:
    """Remove best-fit line y ≈ a + b t to flatten drift."""
    y = np.asarray(y, float).ravel()
    return butter_highpass_filter(y, 0.2, fs=99, order=1)


def moving_average(y: np.ndarray, win: int) -> np.ndarray:
    if win is None or win <= 1:
        return y.astype(float, copy=True)
    win = int(win)
    pad = max(0, win//2)
    ypad = np.pad(y, (pad, pad), mode='edge')
    kernel = np.ones(win, dtype=float) / float(win)
    sm = np.convolve(ypad, kernel, mode='valid')
    return sm[:y.size]

def method1_histogram_peaks(y: np.ndarray, bins: int = 1024) -> Dict[str, float]:
    """
    Detrend → histogram → find two dominant peaks → centers & FWHM.
    Returns: dict with 'delta', 'off_mean', 'noise_std', 'centers' and 'widths' (value-units).
    """
    y = np.asarray(y, float).ravel()
    yd = detrend_linear(y)

    # light smoothing to tame pixel noise (does not bias peaks if small)
    ys = moving_average(yd, max(1, y.size // 200))

    hist, edges = np.histogram(ys, bins=bins)
    centers = 0.5*(edges[1:]+edges[:-1])

    # find two tallest peaks
    pk, _ = find_peaks(hist, height=np.max(hist)*0.2, distance=max(10, bins//20))
    if pk.size < 2:
        # fallback: global min/max centers as two modes approximation
        c1, c2 = np.percentile(ys, 25), np.percentile(ys, 75)
        delta = float(c2 - c1)
        noise_std = float(np.std(ys - np.median(ys), ddof=1))
        return dict(delta=delta, off_mean=float(c1), noise_std=noise_std,
                    centers=(float(c1), float(c2)), widths=(np.nan, np.nan),
                    band_counts=(0,0))

    # take two largest peaks by height
    top2 = pk[np.argsort(hist[pk])[-2:]]
    top2.sort()
    w, wh, left_i, right_i = peak_widths(hist, top2, rel_height=0.5)

    c1, c2 = centers[top2[0]], centers[top2[1]]
    # Convert FWHM in index units to value units:
    left_x  = centers[np.clip(left_i.astype(int), 0, centers.size-1)]
    right_x = centers[np.clip(right_i.astype(int),0, centers.size-1)]
    fwhm1 = float(abs(right_x[0] - left_x[0]))
    fwhm2 = float(abs(right_x[1] - left_x[1]))

    # Gaussian: FWHM = 2*sqrt(2*ln2)*sigma ≈ 2.355 σ
    sigma1 = fwhm1 / 2.355 if np.isfinite(fwhm1) and fwhm1>0 else np.nan
    sigma2 = fwhm2 / 2.355 if np.isfinite(fwhm2) and fwhm2>0 else np.nan
    sigma  = np.nanmean([sigma1, sigma2])

    delta = float(c2 - c1)
    off_mean = float(min(c1, c2))
    noise_std = float(sigma if np.isfinite(sigma) else np.std(ys - np.median(ys), ddof=1))
    return dict(delta=delta, off_mean=off_mean, noise_std=noise_std,
                centers=(float(c1), float(c2)), widths=(float(fwhm1), float(fwhm2)),
                band_counts=(0,0))

def method15_peak_bands(y: np.ndarray, bins: int = 1024, band_k: float = 0.5) -> Dict[str, float]:
    """
    Combined approach: use Method 1 to locate peaks and FWHM, then form narrow bands
    around each peak center (± k * FWHM). Use only samples falling within these bands
    to compute Δ = mean(high_band) - mean(low_band), and pooled σ as noise.
    This explicitly excludes rising/falling points.
    """
    base = method1_histogram_peaks(y, bins=bins)
    c = base['centers']; w = base['widths']
    y = detrend_linear(np.asarray(y, float).ravel())
    ys = moving_average(y, max(1, y.size // 200))

    if any(np.isnan(w)):
        return base  # fallback to method1 if widths unavailable

    c1, c2 = sorted(c)
    w1, w2 = w
    band1 = (ys >= (c1 - band_k*w1)) & (ys <= (c1 + band_k*w1))
    band2 = (ys >= (c2 - band_k*w2)) & (ys <= (c2 + band_k*w2))

    if band1.sum() < 10 or band2.sum() < 10:
        return base

    low_mean  = float(np.mean(ys[band1]))
    high_mean = float(np.mean(ys[band2]))
    delta     = float(high_mean - low_mean)

    # pooled σ
    s1 = float(np.std(ys[band1], ddof=1))
    s2 = float(np.std(ys[band2], ddof=1))
    n1, n2 = int(band1.sum()), int(band2.sum())
    pooled = float(np.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / max(1, (n1+n2-2))))

    return dict(delta=delta, off_mean=low_mean, noise_std=pooled,
                centers=(float(c1), float(c2)), widths=(float(w1), float(w2)),
                band_counts=(n1, n2))

def method2_cycle_maxmin(y: np.ndarray, fps: float, period_s: float,
                         trim_q: float = 0.1) -> Dict[str, float]:
    """
    Segment into cycles (length ≈ round(fps*period_s)) and compute per-cycle
    amplitude as (q_hi - q_lo) using trimmed quantiles (e.g., 90th - 10th).
    Δ is the median across cycles; noise is median absolute deviation of
    plateau samples within cycles (robust but tends to overestimate Δ).
    """
    y = detrend_linear(np.asarray(y, float).ravel())
    L = max(8, int(round(fps * period_s)))
    if y.size < 2*L:
        # fallback to method1 if too short
        base = method1_histogram_peaks(y, bins=256)
        return base

    n_cycles = y.size // L
    cycles = y[:n_cycles*L].reshape(n_cycles, L)

    qlo = np.percentile(cycles, 100*trim_q, axis=1)
    qhi = np.percentile(cycles, 100*(1-trim_q), axis=1)
    amps = qhi - qlo
    delta = float(np.median(amps))

    # Estimate off_mean from lower quantiles, then Δ/Off later per-block
    off_mean = float(np.median(qlo))

    # Noise as robust within-cycle plateau scatter
    # take central region to avoid edges
    mid_lo, mid_hi = int(0.2*L), int(0.8*L)
    mids = cycles[:, mid_lo:mid_hi]
    noise = float(np.median(np.std(mids, axis=1, ddof=1)))

    return dict(delta=delta, off_mean=off_mean, noise_std=noise,
                centers=(np.nan, np.nan), widths=(np.nan, np.nan),
                band_counts=(0,0))

def per_block_measure_v5(exp_npz: Dict[str,Any],
                         fps: float = 10.0, period_s: float = 2.0,
                         use_combined: bool = False,
                         bins: int = 1024) -> Dict[str, Any]:
    sizes = np.asarray(exp_npz['box_sizes']).astype(int)
    k = 0  # only one size stored in this pipeline
    traces = np.asarray(exp_npz['trace_boxes'])[k]   # (B,T)
    rows, cols = int(exp_npz['rows']), int(exp_npz['cols'])
    B, T = traces.shape

    Delta = np.zeros(B, float)
    Noise = np.zeros(B, float)
    Off   = np.zeros(B, float)
    SNR   = np.zeros(B, float)
    band_counts = np.zeros((B,2), int)
    C_low = np.full(B, np.nan, float)
    C_high = np.full(B, np.nan, float)

    for i in range(B):
        y = traces[i]
        if use_combined:
            m = method15_peak_bands(y, bins=bins)
            if not np.isfinite(m['delta']) or m['delta']<=0:
                m = method1_histogram_peaks(y, bins=bins)
        else:
            m = method1_histogram_peaks(y, bins=bins)

        Delta[i] = float(m['delta'])
        Noise[i] = max(1e-12, float(m['noise_std']))
        Off[i]   = float(m['off_mean'])
        SNR[i]   = float(Delta[i]/Noise[i])
        band_counts[i] = m.get('band_counts', (0,0))

        # store peak centers for plotting horizontal lines (lower, upper)
        c = m.get('centers', (np.nan, np.nan))
        try:
            c1, c2 = float(c[0]), float(c[1])
            lo, hi = (c1, c2) if c1 <= c2 else (c2, c1)
        except Exception:
            lo, hi = np.nan, np.nan
        C_low[i] = lo
        C_high[i] = hi

    DeltaFrac = Delta / (Off + 1e-12)
    A = Delta / 2.0

    return {
        'A': A.reshape(rows, cols),
        'SNR': SNR.reshape(rows, cols),
        'DeltaFrac': DeltaFrac.reshape(rows, cols),
        'Delta': Delta.reshape(rows, cols),
        'Noise': Noise.reshape(rows, cols),
        'band_counts': band_counts.reshape(rows, cols, 2),
        'C_low': C_low.reshape(rows, cols),
        'C_high': C_high.reshape(rows, cols),
        'box_used': int(sizes[k]),
        'method': "combined_peak_bands" if use_combined else "histogram_peaks"
    }

# ------------------------------ Plotting ------------------------------

def fig_roi_overlay(exp_npz: Dict[str,Any], title: str = "ROI selection"):
    img = exp_npz['first_frame']
    polys = np.asarray(exp_npz['polygons'])
    fig, ax = plt.subplots(figsize=(6.2,6.2))
    ax.imshow(img, cmap='gray'); ax.set_axis_off()
    coll = PolyCollection(polys, facecolor='none', edgecolor='cyan', linewidth=0.6, alpha=0.9)
    ax.add_collection(coll)
    ax.set_title(title)
    fig.tight_layout()
    return fig

def fig_heatmap(grid: np.ndarray, title: str, center_zero: bool, cmap='RdBu_r'):
    fig, ax = plt.subplots(figsize=(5.8,4.6))
    if center_zero:
        v = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
        im = ax.imshow(grid, cmap=cmap, norm=TwoSlopeNorm(vcenter=0, vmin=-v, vmax=v), origin='upper')
    else:
        v = np.nanmax(grid) if np.isfinite(grid).any() else 1.0
        im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=v, origin='upper')
    cb = fig.colorbar(im, ax=ax, shrink=0.9); cb.set_label(title)
    ax.set_xlabel("col"); ax.set_ylabel("row"); ax.set_title(title)
    fig.tight_layout()
    return fig

def fig_traces_grid(exp_npz: Dict[str,Any], fits: Dict[str,Any], box_for_plot: int = 32):
    """All-block traces with optional scaling and peak-center lines.
    - We plot **detrended + lightly smoothed** traces to align with the
      histogram/peak methods.
    - Two **red dotted** lines per block show the lower/upper peak centers.
    - In-axes text shows **Δ** and **σ** (noise) for that block.
    - Scaling: multiply traces (and lines) by factor k = 100 / mV so that 100 mV → 1×,
      10 mV → 10×, 1 mV → 100×, etc.
    """
    sizes = np.asarray(exp_npz['box_sizes']).astype(int)
    ksize = 0
    traces = np.asarray(exp_npz['trace_boxes'])[ksize]  # [B,T]
    rows, cols = int(exp_npz['rows']), int(exp_npz['cols'])
    B,T = traces.shape

    mV = float(exp_npz.get('perturb_mV', 100.0))
    scale = 100.0 / max(1.0, mV)

    # Prepare display traces (detrend + smooth, then scale)
    win = max(1, T // 200)
    disp = np.empty_like(traces, dtype=float)
    for i in range(B):
        y = detrend_linear(traces[i])
        ys = moving_average(y, win)
        disp[i] = ys * scale

    # Global y-range for shared y-axis
    y_min = float(np.min(disp))
    y_max = float(np.max(disp))
    pad = 0.05 * (y_max - y_min + 1e-9)
    y_lo, y_hi = y_min - pad, y_max + pad

    C_lo = np.asarray(fits.get('C_low'))
    C_hi = np.asarray(fits.get('C_high'))
    Delta = np.asarray(fits.get('Delta'))
    Noise = np.asarray(fits.get('Noise'))

    fig_w = max(8, cols*1.2)
    fig_h = max(6, rows*1.0)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)

    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            ax = axes[r,c]
            ax.plot(disp[i], lw=0.4)
            # draw red dotted lines at scaled centers, if finite
            clo = C_lo[r,c]
            chi = C_hi[r,c]
            if np.isfinite(clo):
                ax.axhline(clo * scale, ls=':', color='red', lw=0.7)
            if np.isfinite(chi):
                ax.axhline(chi * scale, ls=':', color='red', lw=0.7)
            # overlay text: Δ and σ (unscaled numeric values)
            dval = float(Delta[r,c]) if np.isfinite(Delta[r,c]) else np.nan
            sval = float(Noise[r,c]) if np.isfinite(Noise[r,c]) else np.nan

            ax.text(
                0.02, 0.80,
                f"Δ={dval:.3g}\nσ={sval:.3g}",
                transform=ax.transAxes, fontsize=7,
                bbox=dict(fc=(1,1,1,0.55), ec='none', pad=0.2)
            )
            ax.text(0.02, 0.05, f"{r},{c}", transform=ax.transAxes, fontsize=7,
                    bbox=dict(fc=(1,1,1,0.45), ec='none', pad=0.2))
            ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel():
        ax.set_ylim(y_lo, y_hi)

    mv_note = f"(mV diff={int(mV)} → ×{scale:.1f} scale)" if mV>0 else ""
    fig.suptitle(f"All block traces — box {int(sizes[ksize])} {mv_note}")
    fig.tight_layout(rect=[0,0,1,0.97])
    return fig

def overlay_on_first(exp_npz: Dict[str,Any],
                     grid: np.ndarray,
                     title: str,
                     cmap: str = 'RdBu_r',
                     vmax: float | None = None):
    vals = np.asarray(grid, dtype=float).ravel()
    img = exp_npz['first_frame']
    polys = np.asarray(exp_npz['polygons'])
    if vmax is None:
        vmax = np.nanmax(np.abs(vals)) if vals.size else 1.0
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(6.3, 6.3))
    ax.imshow(img, cmap='gray')
    ax.set_axis_off()
    coll = PolyCollection(polys, array=vals, cmap=cmap, norm=norm,
                          alpha=0.35, edgecolor='none')
    ax.add_collection(coll)
    cb = plt.colorbar(coll, ax=ax, shrink=0.85)
    cb.set_label(title)
    ax.set_title(f"{title} — overlay as color (centered at 0)")
    fig.tight_layout()
    return fig

def fig_info_page(E: Dict[str,Any], fits: Dict[str,Any],
                  fps: float, period_s: float, method: str) -> plt.Figure:
    rows, cols = int(E['rows']), int(E['cols'])
    B = rows*cols
    Delta = np.asarray(fits['Delta']).reshape(-1)
    Noise = np.asarray(fits['Noise']).reshape(-1)
    DeltaFrac = np.asarray(fits['DeltaFrac']).reshape(-1)
    SNR = np.asarray(fits['SNR']).reshape(-1)

    # robust aggregates
    med = lambda x: float(np.nanmedian(x))
    mu  = lambda x: float(np.nanmean(x))

    text = []
    text.append("Experiment Report (V5) — INFO")
    text.append("")
    text.append(f"Experiment dir: {E['experiment_dir']}")
    text.append(f"First frame:    {E['first_frame_path']}")
    text.append(f"Frames (T):     {E['frame_count']}")
    text.append(f"ROI grid:       {rows} rows × {cols} cols; box {fits['box_used']} px")
    text.append(f"LED mA / mV:    {E['led_mA']} mA / {E['perturb_mV']} mV")
    text.append(f"Method:         {method}")
    text.append(f"fps / period s: {fps} / {period_s}")
    text.append("")
    text.append("Aggregate metrics across blocks:")
    text.append(f"  Δ (median / mean):      {med(Delta):.4g} / {mu(Delta):.4g}")
    text.append(f"  Noise σ (median / mean): {med(Noise):.4g} / {mu(Noise):.4g}")
    text.append(f"  SNR (median / mean):     {med(SNR):.3g} / {mu(SNR):.3g}")
    text.append(f"  Δ/Off (median / mean):   {med(DeltaFrac):.4g} / {mu(DeltaFrac):.4g}")

    fig = plt.figure(figsize=(8.5, 11.0))
    plt.axis('off')
    plt.text(0.05, 0.95, "\n".join(text), va='top', family='monospace', fontsize=9)
    return fig

# ------------------------------ Pipeline ------------------------------

def ensure_roi_and_maybe_extract(meta: ExperimentPath) -> Dict[str,Any]:
    roi_json = meta.base / "roi_grid_config.json"
    proceed = False
    if not roi_json.exists():
        first = load_bmp_gray(find_first_bmp(meta.base))
        ui = ROIGridUI(first, roi_json)
        print("\n[ROI UI] F (4-pt), drag handles, W/S rows, D/A cols, G boxes, ENTER extract, Q save-only.\n")
        plt.show()
        proceed = ui.proceed
    else:
        print(f"[ROI] Found existing ROI: {roi_json}")
        first = load_bmp_gray(find_first_bmp(meta.base))
        ui = ROIGridUI(first, roi_json)
        cfg = json.loads(roi_json.read_text())
        if 'corners_xy' in cfg:
            ui.corners = np.array(cfg['corners_xy'], float)
            ui._install_handles(); ui._redraw()
        print("\n[ROI UI] (Existing ROI preloaded) — ENTER extract, or adjust then ENTER; Q save-only.\n")
        plt.show()
        proceed = ui.proceed

    E = extract_traces(meta, roi_json, box_sizes=(32,)) if proceed else None
    if E is None:
        print("[Info] Proceeding to extract with saved ROI…")
        E = extract_traces(meta, roi_json, box_sizes=(32,))
    return E

def analyze_experiment_end_to_end_v5(path_arg: Optional[str] = None,
                                     fps: float = 10.0, period_s: float = 2.0,
                                     use_combined: bool = True):
    t_all0 = time.perf_counter()

    base = Path(path_arg).expanduser() if path_arg else pick_folder_dialog()
    meta = choose_experiment(base)
    _set_last_folder(meta.base)

    # ROI + Extract
    t_ex0 = time.perf_counter()
    E = ensure_roi_and_maybe_extract(meta)
    t_ex1 = time.perf_counter()

    # Measure (V5 statistics)
    t_fit0 = time.perf_counter()
    fits = per_block_measure_v5(E, fps=fps, period_s=period_s,
                                use_combined=use_combined, bins=1024)
    t_fit1 = time.perf_counter()

    # Figures
    t_fig0 = time.perf_counter()
    figs: List[plt.Figure] = []
    figs.append(fig_info_page(E, fits, fps=fps, period_s=period_s, method=fits['method']))
    figs.append(fig_roi_overlay(E, "ROI grid selection"))
    figs.append(overlay_on_first(E, 2 * fits['A'], "delta (2A = Δ)"))
    figs.append(fig_heatmap(2 * fits['A'],         f"delta (box {fits['box_used']})", center_zero=True))
    figs.append(fig_heatmap(fits['SNR'],           f"SNR (box {fits['box_used']})",   center_zero=False))
    figs.append(fig_heatmap(fits['DeltaFrac'],     f"Δ/Off (box {fits['box_used']})", center_zero=False))
    figs.append(fig_traces_grid(E, fits, box_for_plot=32))

    # Save PDF
    pdf_path = meta.base / "experiment_report_v5.pdf"
    with PdfPages(pdf_path) as pdf:
        for F in figs:
            pdf.savefig(F)
    print(f"[Report] Saved → {pdf_path}")
    try:
        import subprocess, sys, os
        if sys.platform == 'darwin': subprocess.Popen(['open', str(pdf_path)])
        elif sys.platform.startswith('win'): os.startfile(str(pdf_path))  # type: ignore
        else: subprocess.Popen(['xdg-open', str(pdf_path)])
    except Exception as _e:
        print(f"[WARN] Could not open PDF automatically: {_e}")
    t_fig1 = time.perf_counter()

    # ---------- Timing ----------
    t_all1 = time.perf_counter()
    print("\n[Timing]")
    print(f"  Extract: {t_ex1 - t_ex0:.3f} s")
    print(f"  Measure: {t_fit1 - t_fit0:.3f} s")
    print(f"  Figures: {t_fig1 - t_fig0:.3f} s")
    print(f"  Total:   {t_all1 - t_all0:.3f} s")

# ------------------------------ CLI ------------------------------
if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv)>1 else None
    analyze_experiment_end_to_end_v5(path_arg)
