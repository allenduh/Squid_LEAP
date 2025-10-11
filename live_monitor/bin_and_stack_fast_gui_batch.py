#!/usr/bin/env python3
"""
bin_and_stack_fast_gui_batch.py — Batched binning with reduceat (fast), folder picker, frame range, **binned TIFF** + v4 NPZ

Changes vs previous:
- TIFF stack now contains **binned frames** (shape = rows×cols) instead of raw H×W frames.
- Keeps fast path: OpenCV decode + batched reduceat + threaded prefetch.
- GUI folder picker if --exp omitted; range controls --start/--end for quick tests.
- Option --tiff-mode {binned,raw} (default: binned) if you still want raw.

Outputs
- v4-compatible NPZ with keys: trace_boxes (1,B,T), box_sizes, rows, cols, centers, polygons, first_frame, frame_count, block_labels, block_rc, frame_start, frame_end
- Multipage TIFF stack (ImageJ-friendly). Auto BigTIFF if needed (rare since binned is small).

Usage
  python bin_and_stack_fast_gui_batch.py  # pick folder via GUI
  python bin_and_stack_fast_gui_batch.py --exp D:/exp --box 32 --start 0 --end 2000 --batch 64 --workers 4
  python bin_and_stack_fast_gui_batch.py --exp D:/exp --box 32 --tiff-mode raw  # if you want raw TIFF

Requires: numpy, opencv-python, tifffile, pillow, tkinter (std on Windows)
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2
import tifffile as tiff
from concurrent.futures import ThreadPoolExecutor

# ---------------- GUI picker -----------------
def pick_folder_dialog(title: str = "Select experiment folder") -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.wm_attributes("-topmost", True)
        p = filedialog.askdirectory(title=title)
        root.destroy()
        return Path(p) if p else None
    except Exception:
        return None

# --------------- BMP ordering ----------------
NUM_ONLY = re.compile(r"(?i)^(\d+)\.bmp$")
PAIR_UND = re.compile(r"(?i)^(\d+)[_-](\d+)\.bmp$")
PAIR_BF  = re.compile(r"(?i)^batch[_-]?(\d+)[_-]frame[_-]?(\d+)\.bmp$")

def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]

def list_all_frames(exp: Path, frames_per_file: int = 10) -> List[Path]:
    bases: List[Path] = []
    if (exp / "bmp_export").exists(): bases.append(exp / "bmp_export")
    if (exp / "0").exists(): bases.append(exp / "0")
    nums = [d for d in exp.iterdir() if d.is_dir() and d.name.isdigit()]
    bases.extend(sorted(nums, key=lambda p: int(p.name)))
    bases.append(exp)

    items: List[Tuple[int, Path]] = []
    for base in bases:
        for f in sorted(base.rglob("*.bmp"), key=lambda p: _natural_key(p.name)):
            n = f.name
            m = NUM_ONLY.match(n)
            if m:
                items.append((int(m.group(1)), f)); continue
            m = PAIR_UND.match(n)
            if m:
                b, fr = int(m.group(1)), int(m.group(2))
                items.append((b * frames_per_file + fr, f)); continue
            m = PAIR_BF.match(n)
            if m:
                b, fr = int(m.group(1)), int(m.group(2))
                items.append((b * frames_per_file + fr, f)); continue
    if not items:
        raise SystemExit(f"No BMP frames found under {exp}")
    items.sort(key=lambda t: t[0])
    return [p for _, p in items]

# -------------- ROI packing ------------------

def _build_packed_union_indices(H: int, W: int, centers: np.ndarray, bs: int):
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
            pos_list.append(p)
            cur += p.size
            areas[b] = p.size
        else:
            areas[b] = 0
    pos = np.concatenate(pos_list) if pos_list else np.array([], dtype=np.int64)
    areas = np.maximum(areas, 1)
    return pos.astype(np.int64), starts, areas.astype(np.float32)

# -------------- Prefetch batches -------------

def _read_gray_cv2(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"Failed to read {path}")
    return arr

def iter_batches(paths: List[Path], batch: int, start: int, end: Optional[int], prefetch: int = 4):
    N = len(paths)
    s = max(0, start)
    e = min(N, end) if (end is not None and end >= 0) else N
    sub = paths[s:e]
    if not sub:
        return
    with ThreadPoolExecutor(max_workers=prefetch) as ex:
        it = iter(sub)
        inflight = []
        for _ in range(min(prefetch * batch, len(sub))):
            try:
                p = next(it)
                inflight.append(ex.submit(_read_gray_cv2, p))
            except StopIteration:
                break
        buf: List[np.ndarray] = []
        written = 0
        while inflight:
            fut = inflight.pop(0)
            buf.append(fut.result())
            try:
                p = next(it)
                inflight.append(ex.submit(_read_gray_cv2, p))
            except StopIteration:
                pass
            if len(buf) == batch:
                yield written, np.stack(buf, axis=0)
                written += len(buf); buf.clear()
        if buf:
            yield written, np.stack(buf, axis=0)

# ----------------- Core ----------------------

def run(exp_dir: Path, box_size: int, frames_per_file: int = 10,
        tiff_out: Optional[Path] = None, batch: int = 64, start: int = 0, end: Optional[int] = None,
        prefetch_workers: int = 4, tiff_mode: str = "binned"):
    exp_dir = Path(exp_dir)
    if tiff_out is None:
        tiff_out = exp_dir / ("stack_binned.tif" if tiff_mode == "binned" else "stack_raw.tif")

    roi_json = exp_dir / "roi_grid_config.json"
    if not roi_json.exists():
        raise SystemExit(f"Missing ROI json: {roi_json} (run roi_select_gui.py first)")

    cfg = json.loads(roi_json.read_text())
    rows, cols = int(cfg["rows"]), int(cfg["cols"])
    centers = np.asarray(cfg.get("centers_xy_f") or cfg.get("centers") or cfg.get("cell_centers"))
    if centers is None or centers.shape != (rows * cols, 2):
        raise RuntimeError("centers missing or wrong shape in ROI json")
    centers = np.rint(centers).astype(np.int32)

    polys = cfg.get("cell_polygons", None)
    if polys is not None:
        polys = np.array(polys, dtype=np.float32)

    frames = list_all_frames(exp_dir, frames_per_file=frames_per_file)
    T_total = len(frames)

    # First frame (for shape/dtype)
    first = _read_gray_cv2(frames[0])
    H, W = int(first.shape[0]), int(first.shape[1])

    pos, starts, areas = _build_packed_union_indices(H, W, centers, int(box_size))
    B = centers.shape[0]

    s = max(0, start)
    e = min(T_total, end) if (end is not None and end >= 0) else T_total
    T = max(0, e - s)

    traces = np.empty((B, T), dtype=np.float32)

    # TIFF dims & dtype
    if tiff_mode == "binned":
        # Each page is rows×cols float32 (mean intensity per box)
        page_shape = (rows, cols)
        tiff_dtype = np.float32
        # Binned is tiny; BigTIFF virtually never needed
        use_bigtiff = False
    else:
        page_shape = (H, W)
        tiff_dtype = np.uint8
        bytes_est = int(T) * int(H) * int(W)
        use_bigtiff = bytes_est >= (4 * 1024**3)

    with tiff.TiffWriter(str(tiff_out), bigtiff=use_bigtiff) as tw:
        written = 0
        for offset, batch_arr in iter_batches(frames, batch=batch, start=s, end=e, prefetch=prefetch_workers):
            # ---- Batched binning (this is the key part) ----
            Bf = batch_arr.shape[0]
            flat_batch = batch_arr.reshape(Bf, -1)           # (Bf, H*W)
            gathered   = flat_batch[:, pos]                  # (Bf, |pos|)
            sums_batch = np.add.reduceat(gathered, starts, axis=1)  # (Bf, B)
            means_batch = (sums_batch / areas)[..., None]     # (Bf, B, 1)
            traces[:, written:written + Bf] = (sums_batch / areas).T
            # ----------------------------------------------

            # TIFF pages (binned or raw)
            if tiff_mode == "binned":
                # reshape binned frames (B) -> (rows, cols)
                binned_frames = means_batch.reshape(Bf, rows, cols).astype(tiff_dtype, copy=False)
                for k in range(Bf):
                    tw.write(binned_frames[k], contiguous=True, photometric='minisblack')
            else:
                for k in range(Bf):
                    tw.write(batch_arr[k], contiguous=True, photometric='minisblack')

            written += Bf
            if written % 200 == 0:
                print(f"[{written}/{T}] frames processed…")

    out = dict(
        rows=rows,
        cols=cols,
        box_sizes=np.array([int(box_size)], dtype=np.int32),
        centers=centers,
        polygons=polys if polys is not None else np.zeros((B, 4, 2), dtype=np.float32),
        trace_boxes=traces[None, ...],  # (1,B,T)
        frame_count=T,
        first_frame=first.astype(np.float32),
        first_frame_path=str(frames[s]),
        block_labels=np.array([f"r{r}_c{c}" for r in range(rows) for c in range(cols)], dtype=object),
        block_rc=np.array([[r, c] for r in range(rows) for c in range(cols)], dtype=np.int32),
        frame_start=int(s), frame_end=int(e),
    )
    np.savez_compressed(exp_dir / "exp_block_data.npz", **out)
    print(f"[DONE] NPZ → {exp_dir / 'exp_block_data.npz'} | TIFF → {tiff_out} | traces {traces.shape} | TIFF pages {T} ({'binned' if tiff_mode=='binned' else 'raw'})")

# ---------------- CLI ------------------------

def main():
    ap = argparse.ArgumentParser(description="Batched fast NPZ+TIFF with GUI picker and range; TIFF binned by default")
    ap.add_argument('--exp', help='Experiment folder (if omitted, GUI picker will prompt)')
    ap.add_argument('--box', type=int, default=32)
    ap.add_argument('--frames-per-file', type=int, default=10)
    ap.add_argument('--tiff-out', default=None)
    ap.add_argument('--tiff-mode', choices=['binned','raw'], default='binned')
    ap.add_argument('--batch', type=int, default=64, help='Batch size for reduceat (e.g., 64)')
    ap.add_argument('--start', type=int, default=0, help='Start frame index (after sorting)')
    ap.add_argument('--end', type=int, default=None, help='End frame index (exclusive)')
    ap.add_argument('--workers', type=int, default=4, help='Prefetch threads for I/O')
    args = ap.parse_args()

    exp = Path(args.exp) if args.exp else pick_folder_dialog()
    if not exp:
        print("No folder selected.")
        return

    tiff_out = Path(args.tiff_out) if args.tiff_out else None
    run(exp, box_size=args.box, frames_per_file=args.frames_per_file,
        tiff_out=tiff_out, batch=args.batch, start=args.start, end=args.end,
        prefetch_workers=args.workers, tiff_mode=args.tiff_mode)

if __name__ == '__main__':
    main()
