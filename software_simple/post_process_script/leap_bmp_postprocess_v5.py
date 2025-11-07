#!/usr/bin/env python3
"""
leap_bmp_postprocess_v4.py (updated)
Changes vs previous v4:
- Page 1: traces are centered by μ and scaled by the **global max Δ = P75−P25** so neighbors don't touch.
  Lines are **color-coded by Δ** with a colorbar.
- Page 2: traces are centered by μ and scaled by the **global max (Δ/μ)** (percentage change).
  Panel titles show μ, Δ, and Δ% = 100*(Δ/μ). Red dotted lines mark P25/P75 in the same normalized units.
- Drag & drop: if --exp is not provided, you can **drag & drop a folder into the terminal** and press Enter;
  if left blank, a GUI picker opens. (leap_roi_tools.py remains unchanged)
"""
from __future__ import annotations
import argparse, os, sys, platform, subprocess
from pathlib import Path
from typing import List, Tuple

# GUI folder picker
import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from leap_roi_tools import FourPointSelector, GridAdjuster

# -------------------------- BMP discovery & loader --------------------------

def load_bmp_uint8(path: Path) -> np.ndarray:
    """Load a BMP as uint8 grayscale; returns HxW uint8 array."""
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim == 2:
        return arr.astype(np.uint8, copy=False)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.ndim == 3 and arr.shape[2] == 3:
        r, g, b = arr[..., 0].astype(np.float32), arr[..., 1].astype(np.float32), arr[..., 2].astype(np.float32)
        y = 0.299 * r + 0.587 * g + 0.114 * b
        return np.clip(y, 0, 255).astype(np.uint8)
    return arr.astype(np.uint8, copy=False)

def discover_bmps(exp: Path) -> List[Path]:
    """Find BMPs. If subfolders named '0','1','2' exist, search within them; else search exp root."""
    tuples = []
    subs = [d for d in exp.iterdir() if d.is_dir() and d.name.isdigit()]
    search_dirs = subs if subs else [exp]
    for d in search_dirs:
        for f in sorted(d.glob("*.bmp")):
            name = f.stem
            parts = name.split("_")
            try:
                if len(parts) >= 2:
                    frame = int(parts[0]); glob = int(parts[1])
                else:
                    frame = 0; glob = 0
                tuples.append((glob, frame, f))
            except Exception:
                tuples.append((0, 0, f))
    tuples.sort(key=lambda t: (t[0], t[1], t[2].name))
    return [t[2] for t in tuples]

# ----------------------- Indices and trace extraction ----------------------

def build_packed_indices(H: int, W: int, centers: np.ndarray, box: int):
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
                        B: int):
    """Return (traces[B,T] float32, first_frame uint8)."""
    T = len(bmp_list)
    if T == 0:
        raise SystemExit("No BMPs found in the selected folder.")
    traces = np.empty((B, T), dtype=np.float32)
    first_frame = None
    for t, fpath in enumerate(bmp_list):
        frame = load_bmp_uint8(fpath)
        if first_frame is None:
            first_frame = frame.copy()
        flat_u8 = frame.reshape(-1)
        gathered = flat_u8[pos].astype(np.uint32, copy=False)
        sums = np.add.reduceat(gathered, starts, dtype=np.uint32)
        traces[:, t] = (sums / areas).astype(np.float32)
    return traces, first_frame

# -------------------------- Percentile levels -------------------------------

def percentile_levels(trace: np.ndarray, p_low: float = 25.0, p_high: float = 75.0) -> Tuple[float, float, float, float]:
    """Returns (mu, p25, p75, delta)."""
    x = np.asarray(trace, dtype=np.float32)
    mu = float(np.mean(x))
    p25 = float(np.percentile(x, p_low))
    p75 = float(np.percentile(x, p_high))
    return mu, p25, p75, float(p75 - p25)

# ----------------------------- PDF rendering ------------------------------

def render_pdf_pages(first_frame: np.ndarray, centers: np.ndarray, box: int,
                     traces: np.ndarray, fps: float, out_pdf: Path,
                     rows: int, cols: int,
                     mus: np.ndarray, p25s: np.ndarray, p75s: np.ndarray,
                     deltas: np.ndarray, delta_pcts: np.ndarray,
                     global_delta_max: float, global_delta_pct_max: float):
    """Two pages: (1) image + colored overlays  (2) small multiples with % normalization."""
    H, W = first_frame.shape
    B, T = traces.shape
    h = box // 2

    with PdfPages(out_pdf) as pdf:
        # ---------------- Page 1 ----------------
        fig1, ax1 = plt.subplots(figsize=(W/100, H/100), dpi=100)
        ax1.imshow(first_frame, cmap="gray")
        ax1.set_axis_off()
        cmap = plt.cm.inferno
        norm = plt.Normalize(vmin=0.0, vmax=max(global_delta_max, 1e-6))
        for b in range(B):
            cx, cy = centers[b]
            x0, x1 = int(cx - h), int(cx + h)
            y0, y1 = int(cy - h), int(cy + h)
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(W-1, x1); y1 = min(H-1, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            tr = traces[b]
            mu = mus[b]
            tr_norm = (tr - mu) / max(global_delta_max, 1e-6)
            y_ext0 = max(0, y0 - (y1 - y0))
            y_ext1 = min(H-1, y1 + (y1 - y0))
            ys_line = cy - tr_norm * (y_ext1 - y_ext0) * 0.5
            roi_w = (x1 - x0)
            stretch = max(1, int(round(2.5 * roi_w)))
            xL = int(cx - stretch/2); xR = int(cx + stretch/2)
            xL = max(0, xL); xR = min(W-1, xR)
            if xR <= xL: xL, xR = x0, x1
            xs_line = np.linspace(xL, xR, num=T)
            ax1.plot(xs_line, ys_line, linewidth=0.9, color=cmap(norm(deltas[b])))
            ax1.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, linewidth=0.6, edgecolor="yellow"))
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig1.colorbar(mappable, ax=ax1, fraction=0.046, pad=0.02)
        cbar.set_label("Δ = P75−P25 (intensity)")
        fig1.tight_layout(pad=0)
        pdf.savefig(fig1, bbox_inches="tight", pad_inches=0)
        plt.close(fig1)

        # ---------------- Page 2 ----------------
        fig2 = plt.figure(figsize=(cols*2.5, rows*1.9), dpi=160)
        gs = GridSpec(rows, cols, figure=fig2, wspace=0.25, hspace=0.35)
        for b in range(B):
            r = b // cols; c = b % cols
            ax = fig2.add_subplot(gs[r, c])
            tr = traces[b]
            mu = mus[b]
            denom_mu = max(abs(mu), 1e-6)
            tr_pct = (tr - mu) / (global_delta_pct_max * denom_mu)
            p25n = (p25s[b] - mu) / (global_delta_pct_max * denom_mu)
            p75n = (p75s[b] - mu) / (global_delta_pct_max * denom_mu)
            delta = deltas[b]
            delta_pct = delta / denom_mu * 100.0
            ax.plot(tr_pct, linewidth=0.8)
            ax.axhline(p25n, linestyle=":", linewidth=0.9, color="red")
            ax.axhline(p75n, linestyle=":", linewidth=0.9, color="red")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylim(-1.2, 1.2)
            ax.set_title(f"μ={mu:.3g}  Δ={delta:.3g}  Δ%={delta_pct:.2f}%", fontsize=8)
        fig2.suptitle(
            f"Percent-normalized traces; fps={fps:.3f} Hz (scale by max Δ/μ = {global_delta_pct_max:.3f})",
            fontsize=10
        )
        pdf.savefig(fig2, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig2)

# ----------------------------- Core processor ------------------------------

def open_file(path: Path):
    try:
        if platform.system() == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass

def process_experiment(exp_dir: str | Path, rows: int = 8, cols: int = 9, box: int = 32, fps: float = 5000.0,
                       init_quad: str | np.ndarray = "auto", open_report: bool = True) -> tuple[Path, Path]:
    """Run the full BMP pipeline. Returns (npz_path, pdf_path)."""
    exp_dir = Path(exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment folder not found: {exp_dir}")

    bmp_list = discover_bmps(exp_dir)
    if not bmp_list:
        raise SystemExit(f"No BMP files found under: {exp_dir}")

    # First frame + size
    first = load_bmp_uint8(bmp_list[0])
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
    centers = ga.centers_from_quad()

    # Traces
    pos, starts, areas = build_packed_indices(H, W, centers, box)
    B = rows * cols
    traces, first_frame = extract_traces_bmps(bmp_list, H, W, pos, starts, areas, B)

    # Precompute stats and global maxima
    mus = np.empty(B, dtype=np.float32)
    p25s = np.empty(B, dtype=np.float32)
    p75s = np.empty(B, dtype=np.float32)
    deltas = np.empty(B, dtype=np.float32)
    delta_pcts = np.empty(B, dtype=np.float32)
    for b in range(B):
        mu, p25, p75, d = percentile_levels(traces[b], 25.0, 75.0)
        mus[b], p25s[b], p75s[b], deltas[b] = mu, p25, p75, d
        denom = max(abs(mu), 1e-6)
        delta_pcts[b] = d / denom
    global_delta_max = float(np.max(deltas)) if B>0 else 1.0
    global_delta_pct_max = float(np.max(delta_pcts)) if B>0 else 1.0

    # NPZ minimal
    npz_name = f"{exp_dir.name}_exp_block_data.npz"
    npz_path = exp_dir / npz_name
    np.savez_compressed(npz_path,
                        experiment_dir=str(exp_dir),
                        first_frame=first_frame.astype(np.uint8, copy=False),
                        centers=centers.astype(np.int32, copy=False),
                        traces=traces.astype(np.float32, copy=False))
    print(f"[NPZ] Saved → {npz_path}")

    # PDF
    pdf_path = exp_dir / f"{exp_dir.name}_report.pdf"
    render_pdf_pages(first_frame, centers, box, traces, fps, pdf_path, rows=rows, cols=cols,
                     mus=mus, p25s=p25s, p75s=p75s,
                     deltas=deltas, delta_pcts=delta_pcts,
                     global_delta_max=global_delta_max,
                     global_delta_pct_max=global_delta_pct_max)
    print(f"[PDF] Saved → {pdf_path}")

    if open_report:
        open_file(pdf_path)

    return npz_path, pdf_path

# ----------------------------------- CLI -----------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LEAP BMP-only post-processor (v4 updated)")
    p.add_argument("--exp", type=str, default="", help="Experiment folder. If given, skip the folder picker.")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--box",  type=int, default=32)
    p.add_argument("--fps",  type=float, default=5000.0)
    p.add_argument("--init-quad", type=str, default="auto",
                   help='Seed TL,TR,BL,BR. \"auto\" or \"x1,y1;x2,y2;x3,y3;x4,y4\". Default: auto.')
    p.add_argument("--no-open", action="store_true", help="Do not auto-open the PDF after saving.")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    if args.exp:
        exp_dir = Path(args.exp)
        if not exp_dir.exists():
            raise SystemExit(f"Experiment folder not found: {exp_dir}")
    else:
        # Console drag & drop first
        try:
            print("Drag & drop the experiment FOLDER into this terminal, then press Enter.\n"
                  "(Leave blank to open a folder picker.)")
            dropped = input("> ").strip().strip('"').strip("'")
        except Exception:
            dropped = ''
        if dropped:
            exp_dir = Path(dropped)
            if not exp_dir.exists():
                raise SystemExit("Experiment folder not found: {exp_dir}")
        else:
            root = tk.Tk(); root.withdraw()
            chosen = filedialog.askdirectory(title="Select experiment folder (BMPs)")
            if not chosen:
                raise SystemExit("No folder selected.")
            exp_dir = Path(chosen)
    process_experiment(exp_dir, rows=args.rows, cols=args.cols, box=args.box, fps=args.fps,
                       init_quad=args.init_quad, open_report=not args.no_open)

if __name__ == "__main__":
    main()
