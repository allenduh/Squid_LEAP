#!/usr/bin/env python3
r"""
Binned-frame viewer with ROI normalization by global mean (flicker removal).

- Click image to add/remove ROI traces (up to 8). Selected bins are marked.
- ROI traces are normalized by the per-frame global mean:
    divide   → ROI / global         (default; removes multiplicative flicker)
    subtract → ROI - global         (removes additive common-mode)
    none     → raw ROI
- Fast loader: numpy.memmap + chunked processing.
- RAW options: dtype/endianness/header_bytes/row_stride_bytes.

Dependencies:  pip install pyqtgraph PyQt5
"""

import argparse, sys, time
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import as_strided

# Qt + PyQtGraph imports (prefer PyQt5; fallback to PySide6 if needed)
try:
    from PyQt5 import QtCore, QtWidgets, QtGui
    QT_BINDING = "PyQt5"
except Exception:
    from PySide6 import QtCore, QtWidgets, QtGui  # type: ignore
    QT_BINDING = "PySide6"

import pyqtgraph as pg
pg.setConfigOptions(useOpenGL=True, enableExperimental=True, antialias=False)

# ---------------- RAW helpers ----------------

def _compose_uint16(bytes_hw: np.ndarray, endianness: str) -> np.ndarray:
    lo = bytes_hw[..., 0].astype(np.uint16)
    hi = bytes_hw[..., 1].astype(np.uint16)
    return (lo | (hi << 8)) if endianness == "little" else ((lo << 8) | hi)

def _u8_to_frames(buf: memoryview, n: int, H: int, W: int, row_stride_bytes: int,
                  bytes_per_pix: int, dtype: str, endianness: str) -> np.ndarray:
    """Convert raw bytes to (n,H,W) float32, honoring row padding and dtype."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    frame_stride = H * row_stride_bytes
    if arr.size < n * frame_stride:
        n = arr.size // frame_stride
    if n <= 0:
        return np.zeros((0, H, W), dtype=np.float32)

    arr = as_strided(arr, shape=(n, H, row_stride_bytes),
                     strides=(frame_stride, row_stride_bytes, 1))
    data_bytes = W * bytes_per_pix
    arr = arr[:, :, :data_bytes]  # drop padding

    if dtype == "uint8":
        frames = arr.reshape(n, H, W).astype(np.float32)
    else:
        bytes2 = arr.reshape(n, H, W, 2)
        frames = _compose_uint16(bytes2, endianness).astype(np.float32)
    return frames

def list_raw_files(run_dir: Path, min_bytes: int):
    files = sorted(f for f in run_dir.glob('batch_*.raw') if f.stat().st_size >= min_bytes)
    if not files:
        raise FileNotFoundError("No batch_*.raw files with sufficient size found.")
    return files

# --------------- Load + bin ------------------

def load_and_bin(run_dir: Path, W: int, H: int, FPF: int, expected_frames: int,
                 by: int, bx: int, dtype: str, endianness: str,
                 header_bytes: int, row_stride_bytes: int,
                 chunk_frames: int = 256):
    bytes_per_pix = 1 if dtype == "uint8" else 2
    min_bytes = header_bytes + FPF * H * row_stride_bytes
    files = list_raw_files(run_dir, min_bytes)

    total_frames_avail = len(files) * FPF
    total_frames = total_frames_avail if expected_frames <= 0 else min(expected_frames, total_frames_avail)

    Hc = (H // by) * by
    Wc = (W // bx) * bx
    gh, gw = Hc // by, Wc // bx

    print(f"[Info] Found {len(files)} files, up to {total_frames_avail} frames; loading {total_frames}")
    print(f"[Info] Cropping ({H},{W}) -> ({Hc},{Wc}); bin grid: {gh} x {gw}")
    print(f"[Info] dtype={dtype} bytes/pix={bytes_per_pix} endianness={endianness} header={header_bytes} row_stride={row_stride_bytes}")
    print(f"[Info] chunk_frames={chunk_frames}")

    out = np.empty((total_frames, gh, gw), dtype=np.float32)

    t0 = time.perf_counter()
    io_time = 0.0
    bin_time = 0.0
    filled = 0

    for f in files:
        if filled >= total_frames:
            break
        remain = total_frames - filled
        n_in_file = min(FPF, remain)

        # map the whole file once
        io_t0 = time.perf_counter()
        mm_u8 = np.memmap(f, dtype=np.uint8, mode='r')
        io_time += time.perf_counter() - io_t0

        start = 0
        while start < n_in_file:
            n = min(chunk_frames, n_in_file - start)
            # bytes → frames
            io_t0 = time.perf_counter()
            offset = header_bytes + start * H * row_stride_bytes
            frames = _u8_to_frames(memoryview(mm_u8)[offset: offset + n * H * row_stride_bytes],
                                   n, H, W, row_stride_bytes, bytes_per_pix, dtype, endianness)
            io_time += time.perf_counter() - io_t0

            # crop and bin
            bin_t0 = time.perf_counter()
            fr = frames[:, :Hc, :Wc]
            binned = fr.reshape(fr.shape[0], gh, by, gw, bx).mean(axis=(2, 4), dtype=np.float32)
            bin_time += time.perf_counter() - bin_t0

            out[filled:filled+n] = binned
            filled += n
            start += n

    t1 = time.perf_counter()
    total_elapsed = t1 - t0
    fps = filled / max(1e-9, total_elapsed)
    bytes_total = filled * H * row_stride_bytes
    mib_per_s = (bytes_total / (1024*1024)) / max(1e-9, total_elapsed)

    print(f"[Bench] Frames processed : {filled}")
    print(f"[Bench] Total time       : {total_elapsed:0.2f} s  ({fps:0.0f} fps)")
    print(f"[Bench]   IO time        : {io_time:0.2f} s")
    print(f"[Bench]   Bin time       : {bin_time:0.2f} s")
    print(f"[Bench] Effective BW     : {mib_per_s:0.0f} MiB/s (incl. compute)")
    return out  # (T, gh, gw)

# --------------- GUI ------------------------

class ClickableImage(pg.ImageItem):
    clicked = QtCore.pyqtSignal(int, int)  # y, x
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gh = None; self.gw = None
    def setGridShape(self, gh, gw):
        self.gh = gh; self.gw = gw
    def mouseClickEvent(self, ev):
        if self.gh is None or self.gw is None: return
        if ev.button() != QtCore.Qt.LeftButton: return
        pos = ev.pos()
        x = int(np.clip(np.floor(pos.x()), 0, self.gw - 1))
        y = int(np.clip(np.floor(pos.y()), 0, self.gh - 1))
        self.clicked.emit(y, x); ev.accept()

def _decimate(y: np.ndarray, max_points: int = 20000):
    n = y.shape[0]
    if n <= max_points:
        return np.arange(n), y
    step = int(np.ceil(n / max_points))
    idx = np.arange(0, n, step, dtype=int)
    return idx, y[idx]

class Viewer(QtWidgets.QWidget):
    def __init__(self, binned: np.ndarray, playback_fps: int = 200,
                 norm_mode: str = 'divide', parent=None):
        super().__init__(parent)
        self.binned = binned
        self.T, self.gh, self.gw = binned.shape
        self.playback_fps = playback_fps
        self.cur = 0; self.playing = True
        self.max_traces = 8; self.selected = []
        self.norm_mode = norm_mode; self.eps = 1e-6

        # precompute global mean per frame
        self.global_trace = binned.mean(axis=(1,2)).astype(np.float32)

        self._build_ui(); self._connect()
        self._refresh_frame()
        self.timer.start(int(1000 / max(1, self.playback_fps)))

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)

        # left: image + slider
        left_panel = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left_panel); left_v.setContentsMargins(0,0,0,0); left_v.setSpacing(6)

        self.gw_widget = pg.GraphicsLayoutWidget()
        self.vb = self.gw_widget.addViewBox(lockAspect=True)
        self.img = ClickableImage(); self.img.setImage(self.binned[0], autoLevels=True)
        self.img.setGridShape(self.gh, self.gw)
        self.vb.addItem(self.img); self.vb.invertY(True)

        # overlay markers for selected bins
        self.overlay = pg.ScatterPlotItem(size=12, pen=pg.mkPen(width=2), brush=None)
        self.vb.addItem(self.overlay)

        control_bar = QtWidgets.QWidget(); cb = QtWidgets.QHBoxLayout(control_bar)
        cb.setContentsMargins(0,0,0,0); cb.setSpacing(6)
        self.frame_label = QtWidgets.QLabel(f"Frame: 0 / {self.T-1}")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.T - 1); self.slider.setValue(0); self.slider.setSingleStep(1)
        cb.addWidget(self.frame_label); cb.addWidget(self.slider)

        left_v.addWidget(self.gw_widget, 1); left_v.addWidget(control_bar, 0)
        root.addWidget(left_panel, 2)

        # right: traces
        right_panel = pg.GraphicsLayoutWidget(); root.addWidget(right_panel, 3)
        self.trace_plot = right_panel.addPlot(title="ROI Traces (normalized)")
        self.trace_plot.showGrid(x=True, y=True, alpha=0.3)
        self.trace_items = []
        for _ in range(self.max_traces):
            it = pg.PlotCurveItem(); it.hide(); self.trace_plot.addItem(it); self.trace_items.append(it)

        right_panel.nextRow()
        self.global_plot = right_panel.addPlot(title="Global Mean (raw)")
        self.global_plot.showGrid(x=True, y=True, alpha=0.3)
        gx, gy = _decimate(self.global_trace); self.global_item = pg.PlotCurveItem(gx, gy)
        self.global_plot.addItem(self.global_item)

        # timer + shortcuts
        self.timer = QtCore.QTimer(self); self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.toggle_play)
        QtWidgets.QShortcut(QtGui.QKeySequence("C"), self, activated=self.clear_traces)

        root.addWidget(QtWidgets.QLabel("Click image to toggle ROI traces (÷ global). Space=Play/Pause, C=Clear."))

    def _connect(self):
        self.slider.valueChanged.connect(self.set_frame_index)
        self.img.clicked.connect(self.toggle_trace_at)
        self.timer.timeout.connect(self._tick)

    def _tick(self):
        if not self.playing: return
        self.cur += 1
        if self.cur >= self.T: self.cur = 0
        self.slider.blockSignals(True); self.slider.setValue(self.cur); self.slider.blockSignals(False)
        self._refresh_frame()

    def set_frame_index(self, idx: int):
        self.cur = int(idx); self._refresh_frame()

    def toggle_play(self):
        self.playing = not self.playing

    def clear_traces(self):
        self.selected.clear()
        for it in self.trace_items: it.hide()
        self._update_overlay()

    def toggle_trace_at(self, y: int, x: int):
        key = (y, x)
        if key in self.selected: self.selected.remove(key)
        else:
            if len(self.selected) >= self.max_traces:
                return
            self.selected.append(key)
        self._update_traces(); self._update_overlay()

    def _refresh_frame(self):
        self.img.setImage(self.binned[self.cur], autoLevels=False)
        self.frame_label.setText(f"Frame: {self.cur} / {self.T-1}")
        self._update_traces()

    def _norm(self, roi_trace: np.ndarray) -> np.ndarray:
        g = self.global_trace
        if self.norm_mode == 'divide':
            return roi_trace / (g + self.eps)
        elif self.norm_mode == 'subtract':
            return roi_trace - g
        else:
            return roi_trace

    def _update_overlay(self):
        if not self.selected: self.overlay.setData([], []); return
        xs = [x + 0.5 for (_, x) in self.selected]; ys = [y + 0.5 for (y, _) in self.selected]
        self.overlay.setData(xs, ys)

    def _update_traces(self):
        for i, it in enumerate(self.trace_items):
            if i < len(self.selected):
                y, x = self.selected[i]
                tr = self.binned[:, y, x]
                trn = self._norm(tr)
                # simple decimation for long sequences
                n = trn.size
                if n > 20000:
                    step = int(np.ceil(n / 20000)); idx = np.arange(0, n, step, dtype=int); it.setData(idx, trn[idx])
                else:
                    it.setData(trn)
                it.show()
            else:
                it.hide()
        self.trace_plot.setXRange(max(0, self.cur - 2000), min(self.T, self.cur + 2000), padding=0)

# --------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--frame_w", required=True, type=int)
    ap.add_argument("--frame_h", required=True, type=int)
    ap.add_argument("--frames_per_file", required=True, type=int)
    ap.add_argument("--expected_frames", default=0, type=int)
    ap.add_argument("--bin_y", default=32, type=int)
    ap.add_argument("--bin_x", default=32, type=int)
    ap.add_argument("--playback_fps", default=200, type=int)
    # RAW parsing options
    ap.add_argument("--dtype", choices=["uint8", "uint16"], default="uint8")
    ap.add_argument("--endianness", choices=["little", "big"], default="little")
    ap.add_argument("--header_bytes", type=int, default=0)
    ap.add_argument("--row_stride_bytes", type=int, default=0,
                    help="Bytes per row incl. padding; default = frame_w * bytes/pixel")
    # Performance
    ap.add_argument("--chunk_frames", type=int, default=256)
    # Normalization
    ap.add_argument("--norm_mode", choices=["divide", "subtract", "none"], default="divide")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    bytes_per_pix = 1 if args.dtype == "uint8" else 2
    row_stride = args.row_stride_bytes if args.row_stride_bytes > 0 else args.frame_w * bytes_per_pix

    binned = load_and_bin(run_dir, args.frame_w, args.frame_h, args.frames_per_file,
                          args.expected_frames, args.bin_y, args.bin_x,
                          args.dtype, args.endianness, args.header_bytes, row_stride,
                          chunk_frames=args.chunk_frames)

    app = QtWidgets.QApplication(sys.argv)
    w = Viewer(binned, playback_fps=args.playback_fps, norm_mode=args.norm_mode)
    w.setWindowTitle(f"Binned Frames Viewer v6 ({QT_BINDING}) — ROI normalized: {args.norm_mode}")
    w.resize(1400, 800); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
