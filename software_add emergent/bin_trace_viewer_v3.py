#!/usr/bin/env python3
"""
bin_trace_viewer_v3.py
----------------------
Robust PyQtGraph viewer for binned camera frames with:
- PlotCurveItem (no setDownsampling calls)
- Manual decimation for long traces
- Click-to-select ROI traces + visible overlay markers
- QWidget layout for controls (fixes QSlider issues)

Usage:
python bin_trace_viewer_v3.py --run_dir output/EVT_Py_convert --frame_w 1024 --frame_h 608 --frames_per_file 10 --expected_frames 50000 --bin_y 32 --bin_x 32 --playback_fps 200
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Qt + PyQtGraph imports (prefer PyQt5; fallback to PySide6 if needed)
try:
    from PyQt5 import QtCore, QtWidgets, QtGui
    QT_BINDING = "PyQt5"
except Exception:
    from PySide6 import QtCore, QtWidgets, QtGui  # type: ignore
    QT_BINDING = "PySide6"

import pyqtgraph as pg
pg.setConfigOptions(useOpenGL=True, enableExperimental=True, antialias=False)

# ------------------- Data Loading & Binning -------------------

def list_raw_files(run_dir: Path, needed_bytes: int):
    files = sorted(f for f in run_dir.glob('batch_*.raw') if f.stat().st_size >= needed_bytes)
    if not files:
        raise FileNotFoundError("No batch_*.raw files with sufficient size found.")
    return files

def bin_chunk(frames_chunk: np.ndarray, Hc: int, Wc: int, gh: int, gw: int, by: int, bx: int) -> np.ndarray:
    """frames_chunk: (n, H, W) uint8; returns (n, gh, gw) float32"""
    chunk = frames_chunk[:, :Hc, :Wc]
    binned = chunk.reshape(chunk.shape[0], gh, by, gw, bx).mean(axis=(2, 4), dtype=np.float32)
    return binned

def load_and_bin(run_dir: Path, W: int, H: int, FPF: int, expected_frames: int, by: int, bx: int, chunk_frames: int = 1000):
    needed_bytes = W * H * FPF  # uint8
    files = list_raw_files(run_dir, needed_bytes)
    total_frames_avail = len(files) * FPF
    total_frames = total_frames_avail if expected_frames <= 0 else min(expected_frames, total_frames_avail)

    Hc = (H // by) * by
    Wc = (W // bx) * bx
    gh, gw = Hc // by, Wc // bx

    print(f"[Info] Found {len(files)} files, up to {total_frames_avail} frames; loading {total_frames}")
    print(f"[Info] Cropping ({H},{W}) -> ({Hc},{Wc}); bin grid: {gh} x {gw}")

    out = np.empty((total_frames, gh, gw), dtype=np.float32)

    t0 = time.perf_counter()
    filled = 0
    for f in files:
        if filled >= total_frames:
            break

        mm = np.memmap(f, dtype=np.uint8, mode='r', shape=(FPF, H, W))
        remain = total_frames - filled

        start = 0
        while start < min(FPF, remain):
            n = min(chunk_frames, FPF - start, remain)
            frames_chunk = mm[start:start+n]
            binned = bin_chunk(frames_chunk, Hc, Wc, gh, gw, by, bx)
            out[filled:filled+n] = binned
            filled += n
            start += n

    t1 = time.perf_counter()
    fps = filled / max(1e-9, (t1 - t0))
    print(f"[Done] Binned {filled} frames in {t1 - t0:.2f}s ({fps:.0f} fps)")
    return out  # (T, gh, gw)

# ------------------- GUI -------------------

class ClickableImage(pg.ImageItem):
    clicked = QtCore.pyqtSignal(int, int)  # y, x indices in binned grid

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gh = None
        self.gw = None

    def setGridShape(self, gh, gw):
        self.gh = gh
        self.gw = gw

    def mouseClickEvent(self, ev):
        if self.gh is None or self.gw is None:
            return
        if ev.button() != QtCore.Qt.LeftButton:
            return
        pos = ev.pos()
        x = int(np.clip(np.floor(pos.x()), 0, self.gw - 1))
        y = int(np.clip(np.floor(pos.y()), 0, self.gh - 1))
        self.clicked.emit(y, x)
        ev.accept()

def _decimate(y: np.ndarray, max_points: int = 20000):
    """Return (x, y_decimated) with at most max_points samples for plotting."""
    n = y.shape[0]
    if n <= max_points:
        return np.arange(n), y
    step = int(np.ceil(n / max_points))
    idx = np.arange(0, n, step, dtype=int)
    return idx, y[idx]

class Viewer(QtWidgets.QWidget):
    def __init__(self, binned: np.ndarray, playback_fps: int = 200, parent=None):
        super().__init__(parent)
        self.binned = binned  # (T, gh, gw)
        self.T, self.gh, self.gw = binned.shape
        self.playback_fps = playback_fps
        self.cur = 0
        self.playing = True
        self.max_traces = 8
        self.selected = []  # list of (y,x)

        self.global_trace = binned.mean(axis=(1,2)).astype(np.float32)

        self._build_ui()
        self._connect()
        self._refresh_frame()
        self.timer.start(int(1000 / max(1, self.playback_fps)))

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)

        # LEFT PANEL: image + controls in a vertical layout
        left_panel = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left_panel)
        left_v.setContentsMargins(0,0,0,0)
        left_v.setSpacing(6)

        self.gw_widget = pg.GraphicsLayoutWidget()
        self.vb = self.gw_widget.addViewBox(lockAspect=True)
        self.img = ClickableImage()
        self.img.setImage(self.binned[0], autoLevels=True)
        self.img.setGridShape(self.gh, self.gw)
        self.vb.addItem(self.img)
        self.vb.invertY(True)

        # Scatter overlay to mark selected ROIs
        self.overlay = pg.ScatterPlotItem(size=12, pen=pg.mkPen(width=2), brush=None)
        self.vb.addItem(self.overlay)

        # control bar under the image
        control_bar = QtWidgets.QWidget()
        cb_layout = QtWidgets.QHBoxLayout(control_bar)
        cb_layout.setContentsMargins(0,0,0,0)
        cb_layout.setSpacing(6)

        self.frame_label = QtWidgets.QLabel(f"Frame: 0 / {self.T-1}")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.T - 1)
        self.slider.setValue(0)
        self.slider.setSingleStep(1)

        cb_layout.addWidget(self.frame_label)
        cb_layout.addWidget(self.slider)

        left_v.addWidget(self.gw_widget, 1)
        left_v.addWidget(control_bar, 0)

        root.addWidget(left_panel, 2)

        # RIGHT PANEL: traces
        right_panel = pg.GraphicsLayoutWidget()
        root.addWidget(right_panel, 3)

        self.trace_plot = right_panel.addPlot(title="ROI Traces")
        self.trace_plot.showGrid(x=True, y=True, alpha=0.3)
        self.trace_items = []
        for _ in range(self.max_traces):
            item = pg.PlotCurveItem()
            item.hide()
            self.trace_plot.addItem(item)
            self.trace_items.append(item)

        right_panel.nextRow()
        self.global_plot = right_panel.addPlot(title="Global Mean")
        self.global_plot.showGrid(x=True, y=True, alpha=0.3)
        gx, gy = _decimate(self.global_trace)
        self.global_item = pg.PlotCurveItem(gx, gy)
        self.global_plot.addItem(self.global_item)

        # Timer
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)

        # Shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.toggle_play)
        QtWidgets.QShortcut(QtGui.QKeySequence("C"), self, activated=self.clear_traces)

        # Status
        self.status = QtWidgets.QLabel(
            "Click binned image to add/remove ROI traces (max 8). Space=Play/Pause, C=Clear."
        )
        root.addWidget(self.status)

    def _connect(self):
        self.slider.valueChanged.connect(self.set_frame_index)
        self.img.clicked.connect(self.toggle_trace_at)
        self.timer.timeout.connect(self._tick)

    def _tick(self):
        if not self.playing:
            return
        self.cur += 1
        if self.cur >= self.T:
            self.cur = 0
        self.slider.blockSignals(True)
        self.slider.setValue(self.cur)
        self.slider.blockSignals(False)
        self._refresh_frame()

    def set_frame_index(self, idx: int):
        self.cur = int(idx)
        self._refresh_frame()

    def toggle_play(self):
        self.playing = not self.playing

    def clear_traces(self):
        self.selected.clear()
        for item in self.trace_items:
            item.hide()
        self._update_overlay()

    def toggle_trace_at(self, y: int, x: int):
        key = (y, x)
        if key in self.selected:
            self.selected.remove(key)
        else:
            if len(self.selected) >= self.max_traces:
                self.status.setText(f"Max {self.max_traces} traces; press C to clear some.")
                return
            self.selected.append(key)
        self._update_traces()
        self._update_overlay()

    def _refresh_frame(self):
        self.img.setImage(self.binned[self.cur], autoLevels=False)
        self.frame_label.setText(f"Frame: {self.cur} / {self.T-1}")
        # do not change overlay centers here (static per grid)
        self._update_traces()

    def _update_overlay(self):
        # Show markers at selected bin centers (x+0.5, y+0.5)
        if not self.selected:
            self.overlay.setData([], [])
            return
        xs = [x + 0.5 for (_, x) in self.selected]
        ys = [y + 0.5 for (y, _) in self.selected]
        self.overlay.setData(xs, ys)

    def _update_traces(self):
        for i, item in enumerate(self.trace_items):
            if i < len(self.selected):
                y, x = self.selected[i]
                trace = self.binned[:, y, x]
                tx, ty = _decimate(trace)
                item.setData(tx, ty)
                item.show()
            else:
                item.hide()

        # Follow time window (±2000 frames)
        self.trace_plot.setXRange(max(0, self.cur - 2000), min(self.T, self.cur + 2000), padding=0)

# ------------------- Main -------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--frame_w", required=True, type=int)
    parser.add_argument("--frame_h", required=True, type=int)
    parser.add_argument("--frames_per_file", required=True, type=int)
    parser.add_argument("--expected_frames", default=0, type=int)
    parser.add_argument("--bin_y", default=32, type=int)
    parser.add_argument("--bin_x", default=32, type=int)
    parser.add_argument("--playback_fps", default=200, type=int)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    binned = load_and_bin(run_dir, args.frame_w, args.frame_h, args.frames_per_file,
                          args.expected_frames, args.bin_y, args.bin_x)

    app = QtWidgets.QApplication(sys.argv)
    w = Viewer(binned, playback_fps=args.playback_fps)
    w.setWindowTitle(f"Binned Frames Viewer ({QT_BINDING}) — Click to add ROI traces (Space=Play/Pause, C=Clear)")
    w.resize(1400, 800)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
