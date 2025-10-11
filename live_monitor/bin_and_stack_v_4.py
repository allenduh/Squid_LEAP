#!/usr/bin/env python3
"""
bin_and_stack_v4.py — One-shot extractor that writes BOTH:
  1) v4-compatible NPZ (includes 'trace_boxes')
  2) ImageJ-compatible multi-page TIFF stack

Features
- Robust BMP ordering: supports 96095.bmp, 6_005.bmp, batch12_frame34.bmp
- Streaming binning (uint8 → float32 means per block); minimal RAM
- v4-compatible keys: trace_boxes (1,B,T), box_sizes (1,), centers, polygons, first_frame, frame_count...
- ImageJ stack uses tifffile (pip install tifffile)

Usage
  python bin_and_stack_v4.py --exp "D:/CM_record/A_2025-10-10_11-54-49.128879" --box 32 --frames-per-file 10
  # Optional: --tiff-out "D:/stack_imagej.tif"

Deps: numpy, pillow, tifffile
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
import tifffile as tiff

# ---------- BMP ordering ----------
NUM_ONLY = re.compile(r"(?i)^(\d+)\.bmp$")
PAIR_UND = re.compile(r"(?i)^(\d+)[_-](\d+)\.bmp$")
PAIR_BF  = re.compile(r"(?i)^batch[_-]?(\d+)[_-]frame[_-]?(\d+)\.bmp$")

def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]

def list_all_frames(exp: Path, frames_per_file: int = 10) -> List[Path]:
    bases: List[Path] = []
    if (exp / "bmp_export").exists():
        bases.append(exp / "bmp_export")
    if (exp / "0").exists():
        bases.append(exp / "0")
    nums = [d for d in exp.iterdir() if d.is_dir() and d.name.isdigit()]
    bases.extend(sorted(nums, key=lambda p: int(p.name)))
    bases.append(exp)

    items: List[Tuple[int, Path]] = []
    for base in bases:
        for f in sorted(base.rglob("*.bmp"), key=lambda p: _natural_key(p.name)):
            n = f.name
            m = NUM_ONLY.match(n)
            if m:
                items.append((int(m.group(1)), f))
                continue
            m = PAIR_UND.match(n)
            if m:
                b, fr = int(m.group(1)), int(m.group(2))
                items.append((b * frames_per_file + fr, f))
                continue
            m = PAIR_BF.match(n)
            if m:
                b, fr = int(m.group(1)), int(m.group(2))
                items.append((b * frames_per_file + fr, f))
                continue
    if not items:
        raise SystemExit(f"No BMP frames found under {exp}")
    items.sort(key=lambda t: t[0])
    return [p for _, p in items]

# ---------- ROI packing ----------

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
    areas = np.maximum(areas, 1)  # guard divide-by-zero
    return pos.astype(np.int64), starts, areas.astype(np.float32)

# ---------- core ----------

def run(exp_dir: Path, box_size: int, frames_per_file: int = 10, tiff_out: Path | None = None):
    exp_dir = Path(exp_dir)
    roi_json = exp_dir / "roi_grid_config.json"
    if not roi_json.exists():
        raise SystemExit(f"Missing ROI json: {roi_json} (run roi_select_gui.py first)")

    cfg = json.loads(roi_json.read_text())
    rows, cols = int(cfg["rows"]), int(cfg["cols"])
    centers = np.asarray(
        cfg.get("centers_xy_f") or cfg.get("centers") or cfg.get("cell_centers")
    )
    if centers is None or centers.shape != (rows * cols, 2):
        raise RuntimeError("centers missing or wrong shape in ROI json")
    centers = np.rint(centers).astype(np.int32)

    polys = cfg.get("cell_polygons", None)
    if polys is not None:
        polys = np.array(polys, dtype=np.float32)

    frames = list_all_frames(exp_dir, frames_per_file=frames_per_file)
    T = len(frames)

    # First frame
    first = Image.open(frames[0]).convert("L")
    first_arr = np.array(first, dtype=np.float32)
    H, W = first_arr.shape

    pos, starts, areas = _build_packed_union_indices(H, W, centers, int(box_size))

    B = centers.shape[0]
    traces = np.empty((B, T), dtype=np.float32)



        # after computing: first_arr, pos, starts, areas, traces, frames, T ...
    # choose output
    if tiff_out is None:
        tiff_out = exp_dir / "stack_imagej.tif"

    first_u8 = first_arr.astype(np.uint8)

    with tiff.TiffWriter(str(tiff_out), bigtiff=False) as tw:
        # write first page
        tw.write(first_u8, contiguous=True, photometric='minisblack')

        # correct binning for frame 0
        flat0 = first_u8.reshape(-1)
        gathered0 = flat0[pos].astype(np.uint32, copy=False)
        sums0 = np.add.reduceat(gathered0, starts)
        traces[:, 0] = sums0 / areas

        # remaining frames
        for t_idx, fp in enumerate(frames[1:], start=1):
            arr = np.array(Image.open(fp).convert("L"))  # uint8
            tw.write(arr, contiguous=True, photometric='minisblack')

            flat = arr.reshape(-1)
            gathered = flat[pos].astype(np.uint32, copy=False)
            sums = np.add.reduceat(gathered, starts)
            traces[:, t_idx] = sums / areas

            if (t_idx + 1) % 200 == 0:
                print(f"[{t_idx + 1}/{T}] frames processed…")


    # Save NPZ (v4-compatible)
    out = dict(
        rows=rows,
        cols=cols,
        box_sizes=np.array([int(box_size)], dtype=np.int32),
        centers=centers,
        polygons=polys if polys is not None else np.zeros((B, 4, 2), dtype=np.float32),
        trace_boxes=traces[None, ...],  # (1,B,T)
        frame_count=T,
        first_frame=first_arr,
        first_frame_path=str(frames[0]),
        block_labels=np.array([f"r{r}_c{c}" for r in range(rows) for c in range(cols)], dtype=object),
        block_rc=np.array([[r, c] for r in range(rows) for c in range(cols)], dtype=np.int32),
    )
    np.savez_compressed(exp_dir / "exp_block_data.npz", **out)
    print(f"[DONE] NPZ → {exp_dir / 'exp_block_data.npz'} | TIFF → {tiff_out} | traces {traces.shape}")


def main():
    ap = argparse.ArgumentParser(description="One-shot NPZ + ImageJ TIFF stack")
    ap.add_argument("--exp", required=True, help="Experiment folder with ROI json + BMPs")
    ap.add_argument("--box", type=int, default=32, help="Box size (pixels)")
    ap.add_argument("--frames-per-file", type=int, default=10, help="Frames per RAW file (for two-number names)")
    ap.add_argument("--tiff-out", default=None, help="Optional TIFF output path; default inside exp folder")
    args = ap.parse_args()

    tiff_out = Path(args.tiff_out) if args.tiff_out else None
    run(Path(args.exp), box_size=args.box, frames_per_file=args.frames_per_file, tiff_out=tiff_out)


if __name__ == "__main__":
    main()
