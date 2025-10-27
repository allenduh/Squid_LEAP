#!/usr/bin/env python3
"""
Streamlined RAW -> (Binned TIFF + NPZ) pipeline
- Combines: (1) RAW reader + (2) fast bin+stack — no BMP intermediates.
- Keeps order stable using a sequential memmap reader with batched yields (prefetch-free by default).
- Adds a CompressionManager to centralize parameters (fps, test-file limit, etc.).

Assumptions (tweak via CLI):
- Each RAW contains N frames of uint8 with shape (H, W), laid out contiguously (N*H*W bytes).
- Frame index used for ordering: combined = batch_index * frames_per_file + i; batch_index inferred from digits in filename.

Outputs:
- Multipage TIFF stack:
    * binned mode (default): each page is rows×cols (float32 mean per ROI box)
    * raw mode: optional, each page is H×W (uint8)
- NPZ (v4-compatible) with keys:
    rows, cols, box_sizes, centers, polygons, trace_boxes(1,B,T),
    first_frame(float32), first_frame_path, frame_start, frame_end,
    frame_count, block_labels, block_rc

Usage
  python raw2binned_stack.py --in D:/raw --exp D:/exp --height 608 --width 1024 --n 10 --frames-per-file 10 --box 32 --fps 500
  # quick test on first 3 RAW files and 1200 frames total:
  python raw2binned_stack.py --in D:/raw --exp D:/exp --max-files 3 --end 1200

Notes
- Requires: numpy, tifffile
- ROI config expected at: {exp}/roi_grid_config.json (same shape as your current workflow)
"""
from __future__ import annotations
import argparse, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import numpy as np
import tifffile as tiff

# ------------------------------ Utilities ------------------------------

DIGITS = re.compile(r"(\d+)")

def infer_batch_index(p: Path) -> int:
    """Infer batch index from digits in filename (e.g., 'batch_9609.raw' -> 9609)."""
    m = DIGITS.findall(p.stem)
    return int(m[-1]) if m else 0

def expected_size_bytes(H: int, W: int, N: int) -> int:
    return int(H) * int(W) * int(N)

def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]

# ------------------------------ Compression Manager ------------------------------

@dataclass
class CompressionManager:
    # geometry & raw layout
    height: int
    width: int
    frames_per_raw: int          # N frames in each RAW file
    frames_per_file: int         # FRPF used to derive combined index
    # binning/ROI
    box_size: int
    # I/O & control
    in_dir: Path
    exp_dir: Path
    tiff_out: Optional[Path]
    tiff_mode: str               # 'binned' or 'raw'
    batch: int                   # batch size for processing frames
    start: int                   # first frame index in combined-ordered stream
    end: Optional[int]           # exclusive end index
    max_files: Optional[int]     # limit raw files for quick tests
    fps: Optional[float]         # stored in NPZ
    # misc
    bigtiff_threshold_gb: float = 4.0

    def roi_config_path(self) -> Path:
        return self.exp_dir / "roi_grid_config.json"

    def list_raw_files(self, pattern: str = "*.raw") -> List[Tuple[int, Path]]:
        files = sorted(self.in_dir.glob(pattern), key=lambda p: natural_key(p.name))
        if self.max_files is not None and self.max_files >= 0:
            files = files[: self.max_files]
        jobs = [(infer_batch_index(p), p) for p in files]
        jobs.sort(key=lambda t: t[0])
        return jobs

# ------------------------------ ROI helpers ------------------------------

def load_roi(cfg_path: Path):
    """Load rows, cols, centers (B,2), polygons (B,4,2 or zeros)."""
    if not cfg_path.exists():
        raise SystemExit(f"Missing ROI json: {cfg_path} (run roi_select_gui.py first)")

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
    """Precompute flattened indices for fast reduceat binning."""
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
            areas[b] = 1
    pos = np.concatenate(pos_list) if pos_list else np.array([], dtype=np.int64)
    return pos.astype(np.int64), starts, areas.astype(np.float32)

# ------------------------------ RAW reader (ordered stream) ------------------------------

def iter_frames_from_raws(jobs, H: int, W: int, N: int):
    """Yield frames as uint8 arrays in combined order across RAW files.
    Order: for each sorted (batch_idx, path), frames i=0..N-1.
    """
    for batch_idx, rp in jobs:
        need = expected_size_bytes(H, W, N)
        sz = rp.stat().st_size
        if sz < need:
            print(f"[SKIP] {rp.name} is short: {sz} < {need}")
            continue
        mm = np.memmap(rp, mode='r', dtype=np.uint8, shape=(N, H, W))
        try:
            for i in range(N):
                yield mm[i]  # ndarray view
        finally:
            del mm  # ensure memmap closed

def iter_batches_from_stream(stream, batch: int, start: int, end: Optional[int]):
    """Accumulate frames from a stream into batches of shape (Bf, H, W) honoring [start, end)."""
    buf: List[np.ndarray] = []
    idx = 0
    s = max(0, start)
    e = None if (end is None or end < 0) else int(end)
    for frame in stream:
        if idx < s:
            idx += 1
            continue
        if e is not None and idx >= e:
            break
        buf.append(np.asarray(frame, copy=False))
        idx += 1
        if len(buf) == batch:
            yield np.stack(buf, axis=0)
            buf.clear()
    if buf:
        yield np.stack(buf, axis=0)

# ------------------------------ Core runner ------------------------------

def run(cm: CompressionManager):
    rows, cols, centers, polys = load_roi(cm.roi_config_path())
    jobs = cm.list_raw_files()
    if not jobs:
        raise SystemExit(f"No RAW files found under {cm.in_dir}")

    # Peek first frame cheaply (no need to read everything)
    first_path = jobs[0][1]
    mm0 = np.memmap(first_path, mode='r', dtype=np.uint8, shape=(cm.frames_per_raw, cm.height, cm.width))
    first_frame = np.array(mm0[0], copy=True).astype(np.float32)
    del mm0

    B = centers.shape[0]
    pos, starts, areas = build_packed_union_indices(cm.height, cm.width, centers, cm.box_size)

    # Determine T from [start,end) relative to total available frames
    total_frames = len(jobs) * cm.frames_per_raw
    s = max(0, cm.start)
    e = min(total_frames, cm.end) if (cm.end is not None and cm.end >= 0) else total_frames
    T = max(0, e - s)
    if T == 0:
        raise SystemExit("Frame range empty: nothing to process.")

    traces = np.empty((B, T), dtype=np.float32)

    # TIFF init
    if cm.tiff_out is None:
        cm.tiff_out = cm.exp_dir / ("stack_binned.tif" if cm.tiff_mode == "binned" else "stack_raw.tif")
    if cm.tiff_mode == "binned":
        page_shape = (rows, cols)
        tiff_dtype = np.float32
        use_bigtiff = False
    else:
        page_shape = (cm.height, cm.width)
        tiff_dtype = np.uint8
        bytes_est = T * cm.height * cm.width
        use_bigtiff = (bytes_est >= int(cm.bigtiff_threshold_gb * (1024**3)))

    # Writers
    cm.exp_dir.mkdir(parents=True, exist_ok=True)
    with tiff.TiffWriter(str(cm.tiff_out), bigtiff=use_bigtiff) as tw:
        written = 0
        stream = iter_frames_from_raws(jobs, cm.height, cm.width, cm.frames_per_raw)
        for batch_arr in iter_batches_from_stream(stream, cm.batch, cm.start, cm.end):
            Bf, H, W = batch_arr.shape
            flat_batch = batch_arr.reshape(Bf, -1)            # (Bf, H*W)
            gathered   = flat_batch[:, pos]                   # (Bf, |pos|)
            sums_batch = np.add.reduceat(gathered, starts, axis=1)  # (Bf, B)
            means_batch = (sums_batch / areas)                # (Bf, B)
            # traces fill
            end_idx = written + Bf
            traces[:, written:end_idx] = means_batch.T

            # TIFF pages
            if cm.tiff_mode == "binned":
                binned_frames = means_batch.reshape(Bf, rows, cols).astype(tiff_dtype, copy=False)
                for k in range(Bf):
                    tw.write(binned_frames[k], contiguous=True, photometric='minisblack')
            else:
                for k in range(Bf):
                    tw.write(batch_arr[k], contiguous=True, photometric='minisblack')

            written = end_idx
            if written % 200 == 0 or written == T:
                print(f"[{written}/{T}] frames processed…")

    # NPZ payload
    out = dict(
        rows=rows,
        cols=cols,
        box_sizes=np.array([int(cm.box_size)], dtype=np.int32),
        centers=centers,
        polygons=polys,
        trace_boxes=traces[None, ...],  # (1,B,T)
        frame_count=T,
        first_frame=first_frame,
        first_frame_path=str(first_path),
        block_labels=np.array([f"r{r}_c{c}" for r in range(rows) for c in range(cols)], dtype=object),
        block_rc=np.array([[r, c] for r in range(rows) for c in range(cols)], dtype=np.int32),
        frame_start=int(s), frame_end=int(e),
        fps=float(cm.fps) if cm.fps is not None else np.nan,
    )
    npz_path = cm.exp_dir / "exp_block_data.npz"
    np.savez_compressed(npz_path, **out)
    print(f"[DONE] NPZ -> {npz_path} | TIFF -> {cm.tiff_out} | traces {traces.shape} | pages {T} ({cm.tiff_mode})")

# ------------------------------ CLI ------------------------------

def main():
    ap = argparse.ArgumentParser(description="RAW -> (Binned TIFF + NPZ) without BMP intermediates")
    # input layout
    ap.add_argument('--in', dest='inp', required=True, help='Input folder of RAW files')
    ap.add_argument('--exp', required=True, help='Experiment folder (expects roi_grid_config.json inside)')
    ap.add_argument('--height', type=int, required=True, help='Frame height')
    ap.add_argument('--width', type=int, required=True, help='Frame width')
    ap.add_argument('--n', type=int, dest='frames_per_raw', required=True, help='Frames per RAW file (N)')
    ap.add_argument('--frames-per-file', type=int, default=10, help='FRPF for combined index (kept for compatibility)')
    # roi/bin
    ap.add_argument('--box', type=int, default=32, help='Square box size for binning')
    # outputs/mode
    ap.add_argument('--tiff-out', default=None, help='Output TIFF path (default: {exp}/stack_binned.tif)')
    ap.add_argument('--tiff-mode', choices=['binned','raw'], default='binned', help='Binned pages or raw pages')
    # batching & range
    ap.add_argument('--batch', type=int, default=64, help='Batch size for processing frames (e.g., 64)')
    ap.add_argument('--start', type=int, default=0, help='Start frame index (combined ordering)')
    ap.add_argument('--end', type=int, default=None, help='End frame index (exclusive)')
    # limits/testing
    ap.add_argument('--max-files', type=int, default=None, help='Limit how many RAW files to read (quick tests)')
    ap.add_argument('--fps', type=float, default=None, help='Optional FPS metadata stored in NPZ')

    args = ap.parse_args()

    cm = CompressionManager(
        height=args.height,
        width=args.width,
        frames_per_raw=args.frames_per_raw,
        frames_per_file=args.frames_per_file,
        box_size=args.box,
        in_dir=Path(args.inp),
        exp_dir=Path(args.exp),
        tiff_out=Path(args.tiff_out) if args.tiff_out else None,
        tiff_mode=args.tiff_mode,
        batch=args.batch,
        start=args.start,
        end=args.end,
        max_files=args.max_files,
        fps=args.fps,
    )
    run(cm)

if __name__ == "__main__":
    main()
