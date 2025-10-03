#!/usr/bin/env python3
r"""
Analyze Experiment End-to-End V7 (GUI-first, ROI from V6-style)
---------------------------------------------------------------
• Launch with NO arguments → folder picker → small params dialog →
  4-point selection (TL, TR, BL, BR) with draggable corners → live grid adjust UI
  (W/S rows, A/D cols, box +/- with R/F) → extract traces from BMP **or** RAW →
  save exp_block_data.npz → optional plot_npz_dialog_v4.py.

Key UI (after first frame shows):
  - Click 4 corners in order TL → TR → BL → BR.
  - Drag yellow corner handles to fine-tune.
  - Press ENTER to continue to the Grid Adjuster.
Grid Adjuster UI (mirrors V6 behavior):
  - Drag the same 4 corners.
  - Keys: W/S = rows +/-, A/D = cols +/-, R/F = box px +/-, G = toggle boxes, ENTER = accept.

Default RAW settings (per request):
  - dtype = uint8 (8-bit)
  - frames_per_file = 10

Default folder opened by the picker:
C:\\Users\\user\\Documents\\Github\\Squid_LEAP\\software_add emergent\\output
"""
from __future__ import annotations
import argparse
import importlib
import sys
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# -------- GUI --------
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# -------- Arrays / Images / Plots --------
import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

# ======================== BMP discovery & loader ==========================
BMP_RE = re.compile(r"^(\d+)_(\d+)\.bmp$", re.IGNORECASE)

def load_bmp_gray(path: Path) -> np.ndarray:
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.ndim == 3 and arr.shape[2] == 3:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        arr = 0.299 * r + 0.587 * g + 0.114 * b
    return arr.astype(np.float32)

def discover_bmps(exp: Path) -> List[Path]:
    tuples = []
    subs = [d for d in exp.iterdir() if d.is_dir() and d.name.isdigit()]
    subs.sort(key=lambda p: int(p.name))
    search = subs if subs else [exp]
    for d in search:
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() == ".bmp":
                m = BMP_RE.match(f.name)
                if m:
                    frame, glob = int(m.group(1)), int(m.group(2))
                    tuples.append((glob, frame, f))
    tuples.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in tuples]

# ============================= RAW helpers ================================
@dataclass
class RawLayout:
    H: int
    W: int
    dtype: str        # 'uint8' or 'uint16'
    endianness: str   # 'little' or 'big'
    frames_per_file: int
    header_bytes: int = 0
    row_stride_bytes: int = 0  # 0 → auto = W * bytes_per_pixel

def _compose_uint16(bytes_hw: np.ndarray, endianness: str) -> np.ndarray:
    lo = bytes_hw[..., 0].astype(np.uint16)
    hi = bytes_hw[..., 1].astype(np.uint16)
    return (lo | (hi << 8)) if endianness == "little" else ((lo << 8) | hi)

def _u8_to_frames(buf: memoryview, n: int, H: int, W: int, row_stride_bytes: int,
                  bytes_per_pix: int, dtype: str, endianness: str) -> np.ndarray:
    arr = np.frombuffer(buf, dtype=np.uint8)
    frame_stride = H * row_stride_bytes
    if arr.size < n * frame_stride:
        n = arr.size // frame_stride
    if n <= 0:
        return np.zeros((0, H, W), dtype=np.float32)
    arr = as_strided(arr, shape=(n, H, row_stride_bytes),
                     strides=(frame_stride, row_stride_bytes, 1))
    data_bytes = W * bytes_per_pix
    arr = arr[:, :, :data_bytes]
    if dtype == "uint8":
        frames = arr.reshape(n, H, W).astype(np.float32)
    else:
        bytes2 = arr.reshape(n, H, W, 2)
        frames = _compose_uint16(bytes2, endianness).astype(np.float32)
    return frames

# ========================== ROI & reduction ===============================
class FourPointSelector:
    """Draggable 4-point ROI:
    1) Left-click TL, TR, BL, BR
    2) Drag yellow handles to refine
    3) Press Enter to accept; Right-click to undo last BEFORE 4 placed
    """
    def __init__(self, frame: np.ndarray):
        self.frame = frame
        self.points: list[Tuple[float, float]] = []
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(frame, cmap='gray')
        self.ax.set_title('Click 4 corners (TL, TR, BL, BR). Drag to adjust. Press Enter to finish.')
        # Events
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.drag_idx: Optional[int] = None
        self.accepted = False
        self.redraw()

    def _within_handle(self, event, x, y, tol=8):
        if event.x is None or event.y is None:
            return False
        disp = self.ax.transData.transform((x, y))
        dx = event.x - disp[0]; dy = event.y - disp[1]
        return (dx*dx + dy*dy) ** 0.5 <= tol

    def pick_handle(self, event) -> Optional[int]:
        for i, (x, y) in enumerate(self.points):
            if self._within_handle(event, x, y):
                return i
        return None

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 3 and len(self.points) < 4:
            if self.points:
                self.points.pop(); self.redraw(); return
        if event.button != 1:
            return
        if len(self.points) < 4:
            self.points.append((event.xdata, event.ydata))
            self.redraw()
        else:
            idx = self.pick_handle(event)
            if idx is not None:
                self.drag_idx = idx

    def on_motion(self, event):
        if self.drag_idx is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x = np.clip(event.xdata, 0, self.frame.shape[1]-1)
        y = np.clip(event.ydata, 0, self.frame.shape[0]-1)
        self.points[self.drag_idx] = (x, y)
        self.redraw()

    def on_release(self, event):
        self.drag_idx = None

    def on_key(self, event):
        if event.key == 'enter' and len(self.points) == 4:
            self.accepted = True
            plt.close(self.fig)

    def redraw(self):
        self.ax.clear(); self.ax.imshow(self.frame, cmap='gray')
        labels = ['TL', 'TR', 'BL', 'BR']
        for (x, y) in self.points:
            circ = Circle((x, y), radius=5, facecolor='none', edgecolor='yellow', lw=1.2)
            self.ax.add_patch(circ)
        for i, (x, y) in enumerate(self.points):
            self.ax.text(x+4, y+4, labels[i] if i < 4 else str(i+1), color='y', fontsize=9)
        if len(self.points) == 4:
            (x0,y0),(x1,y1),(x2,y2),(x3,y3) = self.points
            self.ax.plot([x0,x1],[y0,y1],'y-'); self.ax.plot([x2,x3],[y2,y3],'y-')
            self.ax.plot([x0,x2],[y0,y2],'y-'); self.ax.plot([x1,x3],[y1,y3],'y-')
        self.ax.figure.canvas.draw_idle()

    def go(self) -> np.ndarray:
        plt.show()
        if not self.accepted or len(self.points) != 4:
            raise SystemExit('ROI selection cancelled or incomplete. Need 4 points, then press Enter.')
        return np.array(self.points, dtype=np.float32)

class GridAdjuster:
    """V6-like grid adjuster: drag corners, adjust rows/cols/box, preview rectangles, Enter accepts."""
    def __init__(self, frame: np.ndarray, quad: np.ndarray, rows: int, cols: int, box: int):
        self.frame = frame
        self.quad = quad.astype(float)  # TL, TR, BL, BR
        self.rows = int(rows)
        self.cols = int(cols)
        self.box  = int(box)
        self.show_boxes = True
        self.drag_idx: Optional[int] = None

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(frame, cmap='gray')
        self.ax.set_title("Grid Adjuster — Drag corners; W/S rows, A/D cols, R/F box ±; G show; ENTER accept")
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.redraw()

    def centers_from_quad(self) -> np.ndarray:
        tl, tr, bl, br = self.quad[0], self.quad[1], self.quad[2], self.quad[3]
        centers = []
        for r in range(self.rows):
            v = (r + 0.5) / self.rows
            for c in range(self.cols):
                u = (c + 0.5) / self.cols
                top = (1 - u) * tl + u * tr
                bot = (1 - u) * bl + u * br
                p = (1 - v) * top + v * bot
                centers.append(p)
        return np.rint(np.array(centers)).astype(np.int32)

    def on_key(self, e):
        k = (e.key or '').lower()
        if k == 'w': self.rows += 1
        elif k == 's': self.rows = max(1, self.rows - 1)
        elif k == 'd': self.cols += 1
        elif k == 'a': self.cols = max(1, self.cols - 1)
        elif k == 'r': self.box += 1
        elif k == 'f': self.box = max(1, self.box - 1)
        elif k == 'g': self.show_boxes = not self.show_boxes
        elif k == 'enter':
            plt.close(self.fig); return
        self.redraw(lite=True)

    def _within_handle(self, event, x, y, tol=8):
        if event.x is None or event.y is None:
            return False
        disp = self.ax.transData.transform((x, y))
        dx = event.x - disp[0]; dy = event.y - disp[1]
        return (dx*dx + dy*dy) ** 0.5 <= tol

    def pick_handle(self, event) -> Optional[int]:
        for i, (x, y) in enumerate(self.quad):
            if self._within_handle(event, x, y):
                return i
        return None

    def on_click(self, event):
        if event.inaxes != self.ax: return
        idx = self.pick_handle(event)
        if idx is not None: self.drag_idx = idx

    def on_motion(self, event):
        if self.drag_idx is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x = np.clip(event.xdata, 0, self.frame.shape[1]-1)
        y = np.clip(event.ydata, 0, self.frame.shape[0]-1)
        self.quad[self.drag_idx] = [x, y]
        self.redraw(lite=True)

    def on_release(self, event): self.drag_idx = None

    def redraw(self, lite=False):
        self.ax.clear(); self.ax.imshow(self.frame, cmap='gray')
        # draw quad
        TL, TR, BL, BR = self.quad
        self.ax.plot([TL[0],TR[0]],[TL[1],TR[1]],'y-',lw=1)
        self.ax.plot([BL[0],BR[0]],[BL[1],BR[1]],'y-',lw=1)
        self.ax.plot([TL[0],BL[0]],[TL[1],BL[1]],'y-',lw=1)
        self.ax.plot([TR[0],BR[0]],[TR[1],BR[1]],'y-',lw=1)
        # draw handles
        for (x,y) in self.quad:
            circ = Circle((x,y), radius=5, facecolor='none', edgecolor='yellow', lw=1.2)
            self.ax.add_patch(circ)
        # boxes
        if self.show_boxes:
            centers = self.centers_from_quad()
            h = self.box // 2
            for (cx, cy) in centers:
                rect = Rectangle((cx - h, cy - h), self.box, self.box, fill=False, linewidth=0.8, edgecolor='cyan')
                self.ax.add_patch(rect)
        self.ax.set_title(f"Grid Adjuster — rows={self.rows} cols={self.cols} box={self.box} (W/S, A/D, R/F). ENTER to accept.")
        self.ax.figure.canvas.draw_idle()

# ===================== Extraction (BMP and RAW fast) ======================

def build_packed_indices(H: int, W: int, centers: np.ndarray, box: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = box // 2
    cx, cy = centers[:, 0], centers[:, 1]
    x0 = np.clip(cx - h, 0, W)
    y0 = np.clip(cy - h, 0, H)
    x1 = np.clip(x0 + box, 0, W)
    y1 = np.clip(y0 + box, 0, H)
    B = centers.shape[0]
    pos_list = []
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
            pos_list.append(p)
            cur += p.size
            areas[b] = p.size
        else:
            areas[b] = 0
    pos = np.concatenate(pos_list) if pos_list else np.array([], dtype=np.int64)
    areas = np.maximum(areas, 1)
    return pos.astype(np.int64), starts, areas.astype(np.float32)

def extract_traces_bmps(bmp_list: List[Path], H: int, W: int,
                        pos: np.ndarray, starts: np.ndarray, areas: np.ndarray,
                        B: int) -> Tuple[np.ndarray, str, np.ndarray]:
    T = len(bmp_list)
    traces = np.empty((B, T), dtype=np.float32)
    first_frame = None
    for t, fpath in enumerate(bmp_list):
        frame = load_bmp_gray(fpath)
        if first_frame is None:
            first_frame = frame.copy()
        flat = frame.reshape(-1)[pos].astype(np.uint32, copy=False)
        sums = np.add.reduceat(flat, starts)
        traces[:, t] = (sums / areas)
    return traces, str(bmp_list[0]), first_frame

def extract_traces_raw_fast(exp_dir: Path, layout: RawLayout,
                            pos: np.ndarray, starts: np.ndarray, areas: np.ndarray,
                            H: int, W: int, B: int,
                            chunk_frames: int = 256) -> Tuple[np.ndarray, str, np.ndarray]:
    bytes_per_pix = 1 if layout.dtype == 'uint8' else 2
    row_stride = layout.row_stride_bytes if layout.row_stride_bytes > 0 else layout.W * bytes_per_pix
    files = sorted(exp_dir.glob('batch_*.raw')) or sorted(exp_dir.glob('*.raw'))
    if not files:
        raise SystemExit(f"No RAW files in {exp_dir}")

    total_frames = len(files) * layout.frames_per_file
    traces = np.empty((B, total_frames), dtype=np.float32)
    filled = 0
    first_frame = None
    frame_stride = layout.H * row_stride

    for f in files:
        mm_u8 = np.memmap(f, dtype=np.uint8, mode='r')
        n_in_file = min(layout.frames_per_file, total_frames - filled)
        start_idx = 0
        while start_idx < n_in_file:
            n = min(chunk_frames, n_in_file - start_idx)
            offset = layout.header_bytes + start_idx * frame_stride
            buf = memoryview(mm_u8)[offset: offset + n * frame_stride]
            frames = _u8_to_frames(buf, n, H, W, row_stride, bytes_per_pix, layout.dtype, layout.endianness)
            if first_frame is None and frames.shape[0] > 0:
                first_frame = frames[0].copy()
            flat = frames.reshape(n, -1)[:, pos].astype(np.uint32, copy=False)  # (n, P)
            sums = np.add.reduceat(flat, starts, axis=1)                        # (n, B)
            means = (sums / areas[None, :]).astype(np.float32)                  # (n, B)
            traces[:, filled:filled+n] = means.T                                # FIX: transpose to (B, n)
            filled += n
            start_idx += n

    if first_frame is None:
        first_frame = np.zeros((H, W), dtype=np.float32)
    return traces[:, :filled], str(exp_dir / 'batch_*.raw'), first_frame

def extract_traces_raw_safe(exp_dir: Path, layout: RawLayout,
                            pos: np.ndarray, starts: np.ndarray, areas: np.ndarray,
                            H: int, W: int, B: int) -> Tuple[np.ndarray, str, np.ndarray]:
    """
    Simpler, robust reader:
    - No as_strided; uses fromfile on each RAW
    - Verifies expected byte count
    - Enforces (B, T) output
    Assumes: row_stride_bytes == W * bytes_per_pixel, header_bytes == 0
             (set these correctly in the V7 dialog if your files differ)
    """
    files = sorted(exp_dir.glob('batch_*.raw')) or sorted(exp_dir.glob('*.raw'))
    if not files:
        raise SystemExit(f"No RAW files in {exp_dir}")

    bytes_per_pix = 1 if layout.dtype == 'uint8' else 2
    row_stride = layout.row_stride_bytes if layout.row_stride_bytes > 0 else (W * bytes_per_pix)
    assert row_stride == W * bytes_per_pix, \
        f"Row stride ({row_stride}) must equal W*bytes_per_pix ({W*bytes_per_pix}) for this safe reader."

    frame_bytes = H * row_stride
    expected_per_file = layout.frames_per_file * frame_bytes

    # Count total frames up-front (skip incomplete files)
    valid_files = []
    total_frames = 0
    for f in files:
        size = f.stat().st_size - int(layout.header_bytes or 0)
        if size < expected_per_file or size % frame_bytes != 0:
            print(f"[WARN] Skipping {f.name}: size {size} not a multiple of frame size {frame_bytes}")
            continue
        n_frames = min(size // frame_bytes, layout.frames_per_file)
        valid_files.append((f, n_frames))
        total_frames += n_frames

    if total_frames == 0:
        raise SystemExit("No valid RAW frames after size checks.")

    traces = np.empty((B, total_frames), dtype=np.float32)
    first_frame = None
    filled = 0

    for f, n_frames in valid_files:
        # Read the exact number of frames we’ll use from this file
        with open(f, 'rb') as fp:
            # Seek to header if any
            hdr = int(layout.header_bytes or 0)
            if hdr:
                fp.seek(hdr, os.SEEK_SET)
            # Read one file worth of data
            buf = np.fromfile(fp, dtype=np.uint8, count=n_frames * frame_bytes)
        # reshape to (n, H, row_stride) then crop to W
        if layout.dtype == 'uint8':
            frames_u8 = buf.reshape(n_frames, H, row_stride)[:, :, :W]
            frames = frames_u8.astype(np.float32)
        else:
            bytes2 = buf.reshape(n_frames, H, row_stride)[:, :, :W*2].reshape(n_frames, H, W, 2)
            if layout.endianness == 'little':
                frames_u16 = (bytes2[...,0].astype(np.uint16) | (bytes2[...,1].astype(np.uint16) << 8))
            else:
                frames_u16 = ((bytes2[...,0].astype(np.uint16) << 8) | bytes2[...,1].astype(np.uint16))
            frames = frames_u16.astype(np.float32)

        if first_frame is None and frames.shape[0] > 0:
            first_frame = frames[0].copy()

        # Reduce to block means: (n, P) → reduceat → (n, B) → transpose to (B, n)
        flat = frames.reshape(n_frames, -1)[:, pos].astype(np.uint32, copy=False)     # (n, P)
        sums = np.add.reduceat(flat, starts, axis=1)                                   # (n, B)
        means = (sums / areas[None, :]).astype(np.float32)                             # (n, B)
        traces[:, filled:filled+n_frames] = means.T                                    # (B, n)

        filled += n_frames

    if first_frame is None:
        first_frame = np.zeros((H, W), dtype=np.float32)
    return traces[:, :filled], str(exp_dir / 'batch_*.raw'), first_frame

# =============================== NPZ / Plot ===============================

def try_invoke_plotter(npz_path: Path) -> None:
    try:
        spec = importlib.util.find_spec('plot_npz_dialog_v4')
        if spec is not None:
            mod = importlib.import_module('plot_npz_dialog_v4')
            if hasattr(mod, 'main'):
                print('[Plotter] Running plot_npz_dialog_v4.main(...)')
                mod.main([str(npz_path)])
                return
    except Exception as e:
        print(f"[Plotter] Import path failed: {e}")
    here = Path(__file__).resolve().parent
    cand = here / 'plot_npz_dialog_v4.py'
    if cand.exists():
        try:
            print('[Plotter] Launching subprocess: plot_npz_dialog_v4.py')
            subprocess.run([sys.executable, str(cand), str(npz_path)], check=True)
            return
        except Exception as e:
            print(f"[Plotter] Subprocess failed: {e}")
    print('[Plotter] plot_npz_dialog_v4 not found; skipping PDF generation.')

# ============================== GUI: Dialog ===============================
class ParamDialog(tk.Toplevel):
    def __init__(self, master, defaults: Dict[str, Any], has_bmps: bool):
        super().__init__(master)
        self.title('LEAP End-to-End Parameters (V7)')
        self.resizable(False, False)
        self.result = None

        frm = ttk.Frame(self, padding=10)
        frm.grid(row=0, column=0, sticky='nsew')

        def add_row(r, label, var):
            ttk.Label(frm, text=label).grid(row=r, column=0, sticky='e', padx=6, pady=4)
            ent = ttk.Entry(frm, textvariable=var, width=14)
            ent.grid(row=r, column=1, sticky='w')
            return ent

        self.rows_v = tk.StringVar(value=str(defaults.get('rows', 8)))
        self.cols_v = tk.StringVar(value=str(defaults.get('cols', 9)))
        self.box_v  = tk.StringVar(value=str(defaults.get('box', 32)))
        self.fps_v  = tk.StringVar(value=str(defaults.get('fps', 5000.0)))

        add_row(0, 'Rows', self.rows_v)
        add_row(1, 'Cols', self.cols_v)
        add_row(2, 'Box (px)', self.box_v)
        add_row(3, 'FPS', self.fps_v)

        ttk.Separator(frm).grid(row=4, column=0, columnspan=3, sticky='ew', pady=(8,4))

        ttk.Label(frm, text='RAW settings (used only if no BMPs found):').grid(row=5, column=0, columnspan=3, sticky='w', pady=(2,2))

        self.raw_h_v = tk.StringVar(value=str(defaults.get('raw_h', 608)))
        self.raw_w_v = tk.StringVar(value=str(defaults.get('raw_w', 1024)))
        self.fpf_v   = tk.StringVar(value=str(defaults.get('frames_per_file', 10)))  # default 10
        self.dtype_v = tk.StringVar(value=defaults.get('raw_dtype', 'uint8'))        # default 8-bit
        self.end_v   = tk.StringVar(value=defaults.get('endianness', 'little'))
        self.hdr_v   = tk.StringVar(value=str(defaults.get('header_bytes', 0)))
        self.row_v   = tk.StringVar(value=str(defaults.get('row_stride_bytes', 0)))

        add_row(6, 'RAW Height (px)', self.raw_h_v)
        add_row(7, 'RAW Width (px)', self.raw_w_v)
        add_row(8, 'Frames per RAW file', self.fpf_v)

        ttk.Label(frm, text='RAW dtype').grid(row=9, column=0, sticky='e', padx=6, pady=4)
        ttk.Combobox(frm, textvariable=self.dtype_v, values=['uint8','uint16'], width=12, state='readonly').grid(row=9, column=1, sticky='w')
        ttk.Label(frm, text='Endianness').grid(row=10, column=0, sticky='e', padx=6, pady=4)
        ttk.Combobox(frm, textvariable=self.end_v, values=['little','big'], width=12, state='readonly').grid(row=10, column=1, sticky='w')
        add_row(11, 'Header bytes', self.hdr_v)
        add_row(12, 'Row stride (bytes, 0=auto)', self.row_v)

        self.plot_v = tk.BooleanVar(value=defaults.get('invoke_plotter', True))
        ttk.Checkbutton(frm, text='Invoke plot_npz_dialog_v4 after NPZ', variable=self.plot_v).grid(row=13, column=0, columnspan=2, sticky='w', padx=6, pady=(6,0))

        btns = ttk.Frame(frm); btns.grid(row=14, column=0, columnspan=2, pady=(10,0))
        ttk.Button(btns, text='OK', command=self.on_ok).grid(row=0, column=0, padx=5)
        ttk.Button(btns, text='Cancel', command=self.on_cancel).grid(row=0, column=1, padx=5)

        if has_bmps:
            pass

        self.grab_set(); self.protocol('WM_DELETE_WINDOW', self.on_cancel)
        self.wait_visibility(); self.focus_set()

    def on_ok(self):
        try:
            rows = int(self.rows_v.get()); cols = int(self.cols_v.get()); box = int(self.box_v.get()); fps = float(self.fps_v.get())
            raw_h = int(self.raw_h_v.get()); raw_w = int(self.raw_w_v.get()); fpf = int(self.fpf_v.get())
            hdr = int(self.hdr_v.get()); rowb = int(self.row_v.get())
        except Exception:
            messagebox.showerror('Invalid input', 'Please enter valid numeric values.')
            return
        self.result = dict(rows=rows, cols=cols, box=box, fps=fps,
                           raw_h=raw_h, raw_w=raw_w, frames_per_file=fpf,
                           raw_dtype=self.dtype_v.get(), endianness=self.end_v.get(),
                           header_bytes=hdr, row_stride_bytes=rowb,
                           invoke_plotter=self.plot_v.get())
        self.destroy()

    def on_cancel(self):
        self.result = None; self.destroy()

# ================================ Driver =================================

def run_with_gui():
    root = tk.Tk(); root.withdraw()
    exp = filedialog.askdirectory(
        title='Select experiment folder (BMPs or RAW batches)',
        initialdir=r'C:\\Users\\user\\Documents\\Github\\Squid_LEAP\\software_add emergent\\output'
    )
    if not exp:
        sys.exit('No folder selected.')
    exp_dir = Path(exp)
    if not exp_dir.exists():
        sys.exit(f'Experiment folder not found: {exp_dir}')

    bmp_list = discover_bmps(exp_dir)
    has_bmps = len(bmp_list) > 0

    defaults = dict(rows=8, cols=9, box=32, fps=5000.0,
                    raw_h=608, raw_w=1024, frames_per_file=10,   # default 10
                    raw_dtype='uint8', endianness='little',      # default 8-bit
                    header_bytes=0, row_stride_bytes=0,
                    invoke_plotter=True)

    dlg = ParamDialog(root, defaults, has_bmps)
    root.wait_window(dlg)
    if dlg.result is None:
        sys.exit('Cancelled.')

    params = dlg.result

    # Prepare first frame for point-picking
    if has_bmps:
        first = load_bmp_gray(bmp_list[0]); H, W = first.shape
    else:
        H, W = int(params['raw_h']), int(params['raw_w'])
        # --- RAW auto-detect & decode first frame for preview ---
        raw_files = sorted(exp_dir.glob('batch_*.raw')) or sorted(exp_dir.glob('*.raw'))
        if not raw_files:
            raise SystemExit(f'No RAW files in {exp_dir}')
        first_size = raw_files[0].stat().st_size
        raw_dtype = params.get('raw_dtype', 'uint8')
        # Heuristic example for 608x1024x10 @ 8-bit
        bpp = 1 if raw_dtype in ('u8','uint8') else 2
        row_stride = int(params.get('row_stride_bytes', 0)) or (W * bpp)
        frame_bytes = H * row_stride
        if frame_bytes == 0:
            raise SystemExit('RAW frame_bytes is zero; check H/W/stride.')
        # If file is an exact multiple of frame_bytes, derive frames_per_file
        if not params.get('frames_per_file') or first_size % frame_bytes == 0:
            params['frames_per_file'] = max(1, first_size // frame_bytes)
        mm = np.memmap(raw_files[0], dtype=np.uint8, mode='r')
        offset = int(params.get('header_bytes', 0))
        buf = memoryview(mm)[offset: offset + frame_bytes]
        if bpp == 1:
            first = np.frombuffer(buf, dtype=np.uint8).reshape(H, row_stride)[:, :W].astype(np.float32)
        else:
            b = np.frombuffer(buf, dtype=np.uint8).reshape(H, row_stride)[:, :W*2].reshape(H, W, 2)
            little = (params.get('endianness','little')=='little')
            if little:
                first = (b[...,0].astype(np.uint16) | (b[...,1].astype(np.uint16) << 8)).astype(np.float32)
            else:
                first = ((b[...,0].astype(np.uint16) << 8) | b[...,1].astype(np.uint16)).astype(np.float32)

    # 4-point selection
    fp_selector = FourPointSelector(first)
    quad = fp_selector.go()  # TL, TR, BL, BR

    # Live grid adjust like V6
    ga = GridAdjuster(first, quad=quad, rows=int(params['rows']), cols=int(params['cols']), box=int(params['box']))
    plt.show()  # blocks until ENTER in adjuster
    rows, cols, box = ga.rows, ga.cols, ga.box
    quad = ga.quad
    centers = ga.centers_from_quad()
    pos, starts, areas = build_packed_indices(H, W, centers, box)
    B = rows * cols

    # Extract traces
    if has_bmps:
        traces, first_path, first_frame = extract_traces_bmps(bmp_list, H, W, pos, starts, areas, B)
    else:
        layout = RawLayout(H=H, W=W, dtype=params['raw_dtype'], endianness=params['endianness'],
                           frames_per_file=int(params['frames_per_file']),
                           header_bytes=int(params['header_bytes']), row_stride_bytes=int(params['row_stride_bytes']))
        #traces, first_path, first_frame = extract_traces_raw_fast(exp_dir, layout, pos, starts, areas, H, W, B)
        traces, first_path, first_frame = extract_traces_raw_safe(exp_dir, layout, pos, starts, areas, H, W, B)

    # Pack NPZ payload
    payload = dict(
        experiment_dir=str(exp_dir),
        first_frame_path=first_path,
        first_frame=first_frame.astype(np.float32),
        rows=int(rows), cols=int(cols),
        cell_w_um=float((quad[1,0] - quad[0,0]) / max(1, int(cols))),  # placeholders
        cell_h_um=float((quad[2,1] - quad[0,1]) / max(1, int(rows))),
        row_gap_um=0.0, col_gap_um=0.0,
        box_sizes=np.array([int(box)], dtype=np.int32),
        centers=centers,
        polygons=np.zeros((B, 4, 2), dtype=np.float32),
        trace_boxes=traces[None, ...],
        frame_count=traces.shape[1],
        job_id='NA', dose=-1, led_mA=0, perturb_mV=0.0,
        timestamp=exp_dir.name,
        block_labels=np.array([f"r{r}_c{c}" for r in range(int(rows)) for c in range(int(cols))], dtype=object),
        block_rc=np.array([[r, c] for r in range(int(rows)) for c in range(int(cols))], dtype=int),
        fps=float(params['fps']),
    )

    npz_path = exp_dir / 'exp_block_data.npz'
    np.savez_compressed(npz_path, **payload)
    print(f"[NPZ] Saved → {npz_path}")

    if bool(params.get('invoke_plotter', True)):
        try_invoke_plotter(npz_path)

def main(argv: Optional[List[str]] = None):
    run_with_gui()

if __name__ == '__main__':
    main()
