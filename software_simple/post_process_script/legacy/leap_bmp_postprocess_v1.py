#!/usr/bin/env python3
"""
LEAP BMP Post-Process (v1)
- BMP-only (no RAW)
- If --exp is provided, runs that folder; else opens a folder dialog
- Four-point ROI appears immediately (auto-seeded at image borders); drag handles and press Enter
- Grid Adjuster: W/S rows +/- , A/D cols +/- , R/F box +/- , G toggle boxes, Enter accept
- Saves: <exp_dir>/exp_block_data.npz

CLI:
  python leap_bmp_postprocess_v1.py --exp "D:\path\to\experiment" --rows 8 --cols 9 --box 32 --fps 5000 --init-quad auto
Or import and call:
  from leap_bmp_postprocess_v1 import process_experiment
  npz_path = process_experiment("D:/exp", rows=8, cols=9, box=32, fps=5000, init_quad="auto")
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

# GUI
import tkinter as tk
from tkinter import filedialog

# Arrays / Images / Plots
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


# -------------------------- BMP discovery & loader --------------------------

def load_bmp_gray(path: Path) -> np.ndarray:
    """Load a BMP as float32 grayscale."""
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
    """Find BMPs. If subfolders named '0','1','2' exist, search within them; else search exp root."""
    tuples = []
    subs = [d for d in exp.iterdir() if d.is_dir() and d.name.isdigit()]
    search_dirs = subs if subs else [exp]
    for d in search_dirs:
        for f in sorted(d.glob("*.bmp")):
            # Try to parse leading numbers if the filenames are like "<frame>_<global>.bmp"
            name = f.stem
            parts = name.split("_")
            try:
                if len(parts) >= 2:
                    frame = int(parts[0]); glob = int(parts[1])
                else:
                    # Fallback: sort just by name
                    frame = 0; glob = 0
                tuples.append((glob, frame, f))
            except Exception:
                tuples.append((0, 0, f))
    tuples.sort(key=lambda t: (t[0], t[1], t[2].name))
    return [t[2] for t in tuples]


# -------------------------- ROI selector (4 points) -------------------------

class FourPointSelector:
    """Draggable 4-point ROI (TL, TR, BL, BR). Press Enter to accept."""
    def __init__(self, frame: np.ndarray, init_points: Optional[np.ndarray] = None):
        self.frame = frame
        self.points: list[tuple[float, float]] = []
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(frame, cmap="gray")
        self.ax.set_title("Click 4 corners TL,TR,BL,BR. Drag to adjust. Press Enter to finish.")
        self.drag_idx: Optional[int] = None
        self.accepted = False

        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        if init_points is not None and len(init_points) == 4:
            self.points = [(float(x), float(y)) for x, y in init_points]

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
        if event.key == "enter" and len(self.points) == 4:
            self.accepted = True
            plt.close(self.fig)

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.frame, cmap="gray")
        labels = ["TL", "TR", "BL", "BR"]
        for (x, y) in self.points:
            self.ax.add_patch(Circle((x, y), radius=5, facecolor="none", edgecolor="yellow", lw=1.2))
        for i, (x, y) in enumerate(self.points):
            self.ax.text(x+4, y+4, labels[i] if i < 4 else str(i+1), color="y", fontsize=9)
        if len(self.points) == 4:
            (x0,y0),(x1,y1),(x2,y2),(x3,y3) = self.points
            self.ax.plot([x0,x1],[y0,y1],"y-"); self.ax.plot([x2,x3],[y2,y3],"y-")
            self.ax.plot([x0,x2],[y0,y2],"y-"); self.ax.plot([x1,x3],[y1,y3],"y-")
        self.ax.figure.canvas.draw_idle()

    def go(self) -> np.ndarray:
        plt.show()
        if not self.accepted or len(self.points) != 4:
            raise SystemExit("ROI selection cancelled or incomplete. Need 4 points, then press Enter.")
        return np.array(self.points, dtype=np.float32)


# ----------------------------- Grid Adjuster -------------------------------

class GridAdjuster:
    """Drag corners; W/S rows, A/D cols, R/F box; G toggle boxes; Enter accept."""
    def __init__(self, frame: np.ndarray, quad: np.ndarray, rows: int, cols: int, box: int):
        self.frame = frame
        self.quad = quad.astype(float)  # TL, TR, BL, BR
        self.rows = int(rows)
        self.cols = int(cols)
        self.box  = int(box)
        self.show_boxes = True
        self.drag_idx: Optional[int] = None

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(frame, cmap="gray")
        self.ax.set_title("Grid Adjuster — Drag corners; W/S rows, A/D cols, R/F box; G show; ENTER accept")
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)
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
        k = (e.key or "").lower()
        if k == "w": self.rows += 1
        elif k == "s": self.rows = max(1, self.rows - 1)
        elif k == "d": self.cols += 1
        elif k == "a": self.cols = max(1, self.cols - 1)
        elif k == "r": self.box += 1
        elif k == "f": self.box = max(1, self.box - 1)
        elif k == "g": self.show_boxes = not self.show_boxes
        elif k == "enter":
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
        self.ax.clear(); self.ax.imshow(self.frame, cmap="gray")
        TL, TR, BL, BR = self.quad
        self.ax.plot([TL[0],TR[0]],[TL[1],TR[1]],"y-",lw=1)
        self.ax.plot([BL[0],BR[0]],[BL[1],BR[1]],"y-",lw=1)
        self.ax.plot([TL[0],BL[0]],[TL[1],BL[1]],"y-",lw=1)
        self.ax.plot([TR[0],BR[0]],[TR[1],BR[1]],"y-",lw=1)
        for (x,y) in self.quad:
            self.ax.add_patch(Circle((x,y), radius=5, facecolor="none", edgecolor="yellow", lw=1.2))
        if self.show_boxes:
            centers = self.centers_from_quad()
            h = self.box // 2
            for (cx, cy) in centers:
                rect = Rectangle((cx - h, cy - h), self.box, self.box, fill=False, linewidth=0.8, edgecolor="cyan")
                self.ax.add_patch(rect)
        self.ax.set_title(f"Grid Adjuster — rows={self.rows} cols={self.cols} box={self.box} (W/S, A/D, R/F). ENTER to accept.")
        self.ax.figure.canvas.draw_idle()


# ----------------------- Indices and trace extraction ----------------------

def build_packed_indices(H: int, W: int, centers: np.ndarray, box: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (pos, starts, areas) for block means over a flat frame."""
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
                        B: int) -> tuple[np.ndarray, str, np.ndarray]:
    """Return (traces[B,T], first_path, first_frame)."""
    T = len(bmp_list)
    if T == 0:
        raise SystemExit("No BMPs found in the selected folder.")
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


# ----------------------------- Core processor ------------------------------

def process_experiment(exp_dir: str | Path, rows: int = 8, cols: int = 9, box: int = 32, fps: float = 5000.0,
                       init_quad: str | np.ndarray = "auto") -> Path:
    """Run the full BMP pipeline. Returns path to the saved NPZ."""
    exp_dir = Path(exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {exp_dir}")

    bmp_list = discover_bmps(exp_dir)
    if not bmp_list:
        raise SystemExit(f"No BMP files found under: {exp_dir}")

    # First frame + size
    first = load_bmp_gray(bmp_list[0])
    H, W = first.shape

    # Build initial quad
    if isinstance(init_quad, str):
        q = init_quad.strip().lower()
        if q == "auto" or q == "":
            inset = 10
            init_pts = np.array([[inset, inset],
                                 [W - inset, inset],
                                 [inset, H - inset],
                                 [W - inset, H - inset]], dtype=np.float32)
        else:
            # parse "x1,y1;x2,y2;x3,y3;x4,y4"
            try:
                pairs = [p.strip() for p in q.split(";")]
                if len(pairs) != 4: raise ValueError
                xy = []
                for p in pairs:
                    x, y = p.split(",")
                    xy.append([float(x), float(y)])
                init_pts = np.array(xy, dtype=np.float32)
            except Exception:
                raise ValueError("init_quad format invalid. Use 'auto' or 'x1,y1;x2,y2;x3,y3;x4,y4'.")
    else:
        init_pts = np.array(init_quad, dtype=np.float32)

    # 4-point selector
    fp_selector = FourPointSelector(first, init_points=init_pts)
    quad = fp_selector.go()

    # Grid adjust
    ga = GridAdjuster(first, quad=quad, rows=int(rows), cols=int(cols), box=int(box))
    plt.show()
    rows, cols, box = ga.rows, ga.cols, ga.box
    quad = ga.quad
    centers = ga.centers_from_quad()
    pos, starts, areas = build_packed_indices(H, W, centers, box)
    B = rows * cols

    # Extract traces
    traces, first_path, first_frame = extract_traces_bmps(bmp_list, H, W, pos, starts, areas, B)

    # Pack NPZ
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
        fps=float(fps),
    )

    npz_path = exp_dir / "exp_block_data.npz"
    np.savez_compressed(npz_path, **payload)
    print(f"[NPZ] Saved → {npz_path}")
    return npz_path


# ----------------------------------- CLI -----------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LEAP BMP-only post-processor")
    p.add_argument("--exp", type=str, default="", help="Experiment folder. If given, skip the folder picker.")
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--box",  type=int, default=32)
    p.add_argument("--fps",  type=float, default=5000.0)
    p.add_argument("--init-quad", type=str, default="auto",
                   help='Seed TL,TR,BL,BR. "auto" or "x1,y1;x2,y2;x3,y3;x4,y4". Default: auto.')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.exp:
        exp_dir = Path(args.exp)
        if not exp_dir.exists():
            raise SystemExit(f"Experiment folder not found: {exp_dir}")
    else:
        root = tk.Tk(); root.withdraw()
        chosen = filedialog.askdirectory(title="Select experiment folder (BMPs)")
        if not chosen:
            raise SystemExit("No folder selected.")
        exp_dir = Path(chosen)
    process_experiment(exp_dir, rows=args.rows, cols=args.cols, box=args.box, fps=args.fps,
                       init_quad=args.init_quad)


if __name__ == "__main__":
    main()
