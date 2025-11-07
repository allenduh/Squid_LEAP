
#!/usr/bin/env python3
"""
leap_bmp_postprocess_v3.py
- BMP-only; imports FourPointSelector & GridAdjuster from leap_roi_tools
- Frames kept uint8; traces float32
- NPZ now ONLY: experiment_dir (str), first_frame (uint8), centers (int32), traces (float32)
  and filename is "<exp_dir.name>_exp_block_data.npz" saved into exp_dir
- Two-page PDF:
  * Page 1: first frame + raw traces overlaid in RED, scaled to extend beyond each ROI box (2x height)
  * Page 2: grid of per-ROI raw traces; dotted lines at two histogram peaks (levels);
            text showing mean, Δ (high-low), σ (pooled), and S/N = Δ/σ
- Auto-opens PDF (can be disabled)
"""
from __future__ import annotations
import argparse, os, sys, platform, subprocess
from pathlib import Path
from typing import List, Optional, Tuple

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
    """Return (traces[B,T] float32, first_frame uint8). Sums in uint32 for safety."""
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

# -------------------------- Level estimation & stats ------------------------

def two_level_stats(trace: np.ndarray, bins: int = 128):
    """
    Find two histogram peaks (low/high), return:
    (mu, mu_low, mu_high, delta, sigma_pooled, sn)
    """
    x = np.asarray(trace, dtype=np.float32)
    mu = float(np.mean(x))

    hist, edges = np.histogram(x, bins=bins)
    k = np.array([1,2,3,2,1], dtype=np.float32); k /= k.sum()
    hist_s = np.convolve(hist.astype(np.float32), k, mode="same")

    lm = []
    for i in range(1, len(hist_s)-1):
        if hist_s[i] >= hist_s[i-1] and hist_s[i] >= hist_s[i+1]:
            lm.append(i)
    if not lm:
        mu_low = mu_high = mu
    else:
        idx_sorted = list(np.argsort(hist_s[lm])[::-1])
        chosen = []
        for idx in idx_sorted:
            if not chosen:
                chosen.append(lm[idx])
            else:
                if abs(lm[idx] - chosen[0]) >= 5:
                    chosen.append(lm[idx]); break
        if len(chosen) < 2:
            # fallback to extreme far bin from the strongest peak
            far = 0 if lm[idx_sorted[0]] < len(hist_s)/2 else len(hist_s)-1
            chosen = sorted([lm[idx_sorted[0]], far])
        chosen = sorted(chosen)
        c0, c1 = chosen[0], chosen[1]
        centers = 0.5 * (edges[:-1] + edges[1:])
        mu_low  = float(centers[c0])
        mu_high = float(centers[c1])

    thr = 0.5*(mu_low + mu_high)
    low_pts  = x[x <= thr]
    high_pts = x[x >  thr]
    std_low  = float(np.std(low_pts))  if low_pts.size  else 0.0
    std_high = float(np.std(high_pts)) if high_pts.size else 0.0
    sigma = 0.5*(std_low + std_high)
    delta = float(mu_high - mu_low)
    sn = (delta / sigma) if sigma > 0 else np.inf
    return mu, mu_low, mu_high, delta, sigma, sn

# ----------------------------- PDF rendering ------------------------------

def render_pdf_pages(first_frame: np.ndarray, centers: np.ndarray, box: int,
                     traces: np.ndarray, fps: float, out_pdf: Path):
    """Two pages: (1) image + red overlays  (2) small multiples + dotted levels + stats."""
    H, W = first_frame.shape
    B, T = traces.shape
    h = box // 2

    with PdfPages(out_pdf) as pdf:
        # Page 1
        fig1, ax1 = plt.subplots(figsize=(W/100, H/100), dpi=100)
        ax1.imshow(first_frame, cmap="gray")
        ax1.set_axis_off()
        for b in range(B):
            cx, cy = centers[b]
            x0, x1 = int(cx - h), int(cx + h)
            y0, y1 = int(cy - h), int(cy + h)
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(W-1, x1); y1 = min(H-1, y1)
            if x1 <= x0 or y1 <= y0: continue
            tr = traces[b]
            mn, mx = float(np.min(tr)), float(np.max(tr))
            span = (mx - mn) if mx > mn else 1.0
            tr_norm = (tr - mn) / span
            y_ext0 = y0 - (y1 - y0)
            y_ext1 = y1 + (y1 - y0)
            y_ext0 = max(0, y_ext0); y_ext1 = min(H-1, y_ext1)
            xs_line = np.linspace(x0, x1, num=T)
            ys_line = y_ext1 - tr_norm * (y_ext1 - y_ext0)
            ax1.plot(xs_line, ys_line, linewidth=0.9, color="red")
            ax1.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, linewidth=0.6, edgecolor="yellow"))
        fig1.tight_layout(pad=0)
        pdf.savefig(fig1, bbox_inches="tight", pad_inches=0)
        plt.close(fig1)

        # Page 2
        # Layout as close to square as possible
        cols = int(np.ceil(np.sqrt(B)))
        rows = int(np.ceil(B / cols))
        fig2 = plt.figure(figsize=(cols*2.4, rows*1.8), dpi=150)
        gs = GridSpec(rows, cols, figure=fig2, wspace=0.25, hspace=0.35)
        for b in range(B):
            r = b // cols; c = b % cols
            ax = fig2.add_subplot(gs[r, c])
            tr = traces[b]
            mu, mu_low, mu_high, delta, sigma, sn = two_level_stats(tr)
            ax.plot(tr, linewidth=0.8)
            ax.axhline(mu_low,  linestyle=":", linewidth=0.8)
            ax.axhline(mu_high, linestyle=":", linewidth=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"Δ={delta:.4g}  σ={sigma:.4g}  S/N={sn:.3g}\nμ={mu:.3g}", fontsize=8)
        fig2.suptitle(f"RAW traces; fps={fps:.3f} Hz (Page 2: levels & stats)", fontsize=10)
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

    # NPZ minimal; filename includes experiment_dir name
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
    render_pdf_pages(first_frame, centers, box, traces, fps, pdf_path)
    print(f"[PDF] Saved → {pdf_path}")

    if open_report:
        open_file(pdf_path)

    return npz_path, pdf_path

# ----------------------------------- CLI -----------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LEAP BMP-only post-processor (v3)")
    p.add_argument("--exp", type=str, default="", help="Experiment folder. If given, skip the folder picker.")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=9)
    p.add_argument("--box",  type=int, default=32)
    p.add_argument("--fps",  type=float, default=5000.0)
    p.add_argument("--init-quad", type=str, default="auto",
                   help='Seed TL,TR,BL,BR. "auto" or "x1,y1;x2,y2;x3,y3;x4,y4". Default: auto.')
    p.add_argument("--no-open", action="store_true", help="Do not auto-open the PDF after saving.")
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
                       init_quad=args.init_quad, open_report=not args.no_open)

if __name__ == "__main__":
    main()
