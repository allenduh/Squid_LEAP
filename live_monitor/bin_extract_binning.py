#!/usr/bin/env python3
"""
bin_extract_binning.py — Fast binning using saved ROI (CLEAN VERSION)

- Reads roi_grid_config.json (created by roi_select_gui.py)
- Loads BMP frames in correct order (numeric-only like 96095.bmp, 6_005.bmp, or batch12_frame34.bmp)
- Streams frames (uint8) and computes per-block means without storing full video
- Saves compact exp_block_data.npz with traces (B × T) + metadata

Deps: numpy, pillow
"""
from __future__ import annotations
import json
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image

# ---------- BMP ordering ----------
NUM_ONLY = re.compile(r"(?i)^(\d+)\.bmp$")
PAIR_UND = re.compile(r"(?i)^(\d+)[_-](\d+)\.bmp$")
PAIR_BF  = re.compile(r"(?i)^batch[_-]?(\d+)[_-]frame[_-]?(\d+)\.bmp$")


def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def list_all_frames(exp: Path, frames_per_file: int = 10) -> List[Path]:
    """Return BMP frames sorted numerically; supports several naming styles."""
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


# ---------- ROI to packed indices ----------

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


# ---------- core extraction ----------

def extract(exp_dir: Path, box_size: int, frames_per_file: int = 10) -> Dict[str, Any]:
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

    frames = list_all_frames(exp_dir, frames_per_file=frames_per_file)
    T = len(frames)

    # First frame: infer H,W and store preview
    first = Image.open(frames[0]).convert("L")
    first_arr = np.array(first)  # uint8
    H, W = first_arr.shape

    # Packed indices (flat indexing into H*W)
    pos, starts, areas = _build_packed_union_indices(H, W, centers, int(box_size))

    B = centers.shape[0]
    traces = np.empty((B, T), dtype=np.float32)  # output (B,T)

    # Streaming pass over frames
    for t, fp in enumerate(frames):
        arr = np.array(Image.open(fp).convert("L"))  # uint8 (H,W)
        flat = arr.reshape(-1)  # uint8 (H*W)
        gathered = flat[pos].astype(np.uint32, copy=False)  # avoid overflow in sum
        sums = np.add.reduceat(gathered, starts)  # (B,)
        traces[:, t] = sums / areas  # mean per block

        if (t + 1) % 100 == 0:
            print(f"[{t + 1}/{T}] frames processed…")

    out = dict(
        experiment_dir=str(exp_dir),
        first_frame_path=str(frames[0]),
        first_frame=first_arr,
        rows=rows,
        cols=cols,
        box_size=int(box_size),
        centers=centers,
        frame_count=T,
        traces=traces,  # (B,T) float32
        block_labels=np.array([f"r{r}_c{c}" for r in range(rows) for c in range(cols)], dtype=object),
        block_rc=np.array([[r, c] for r in range(rows) for c in range(cols)], dtype=np.int32),
    )
    np.savez_compressed(exp_dir / "exp_block_data.npz", **out)
    print(f"[Extract] Saved traces → {exp_dir / 'exp_block_data.npz'}  (shape {traces.shape})")
    return out


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Fast binning extractor (streams frames; saves B×T traces)"
    )
    ap.add_argument(
        "--exp", required=True, help="Experiment folder containing ROI json and BMPs"
    )
    ap.add_argument("--box", type=int, default=32, help="Box size (pixels)")
    ap.add_argument(
        "--frames-per-file",
        type=int,
        default=10,
        help="Frames per RAW file (for two-number names)",
    )
    args = ap.parse_args()
    extract(Path(args.exp), box_size=args.box, frames_per_file=args.frames_per_file)


if __name__ == "__main__":
    main()
