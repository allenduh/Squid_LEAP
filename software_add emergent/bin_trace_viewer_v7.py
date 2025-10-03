#!/usr/bin/env python3
r"""
bin_trace_viewer_v7.py
----------------------
- Browse for a folder (File ▸ Open Folder…) to load batch_*.raw files.
- ROI traces normalized by framewise GLOBAL mean (default: divide).
- Overview plot with a draggable LinearRegionItem controls the visible time window.
- Time axis can be in seconds if you set --fps (>0), otherwise frames.
- Fast chunked loader with IO vs bin (compute) timing.
- Click image to add/remove ROI traces (up to 8). Markers show selected bins.

Dependencies:  pip install pyqtgraph PyQt5
"""

import argparse, sys, time
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import as_strided

# Qt/PyQtGraph
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
                 io_mode: str = "memmap", chunk_frames: int = 256):
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
    print(f"[Info] io_mode={io_mode} chunk_frames={chunk_frames}")

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

        if io_mode == "memmap":
            io_t0 = time.perf_counter()
            mm_u8 = np.memmap(f, dtype=np.uint8, mode='r')
            io_time += time.perf_counter() - io_t0

            start = 0
            while start < n_in_file:
                n = min(chunk_frames, n_in_file - start)
                io_t0 = time.perf_counter()
                offset = header_bytes + start * H * row_stride_bytes
                frames = _u8_to_frames(memoryview(mm_u8)[offset: offset + n * H * row_stride_bytes],
                                       n, H, W, row_stride_bytes, bytes_per_pix, dtype, endianness)
                io_time += time.perf_counter() - io_t0

                bin_t0 = time.perf_counter()
                fr = frames[:, :Hc, :Wc]
                binned = fr.reshape(fr.shape[0], gh, by, gw, bx).mean(axis=(2, 4), dtype=np.float32)
                bin_time += time.perf_counter() - bin_t0

                out[filled:filled+n] = binned
                filled += n
                start += n

        else:  # explicit read()
            with open(f, 'rb', buffering=1024*1024) as fh:
                io_t0 = time.perf_counter()
                fh.seek(header_bytes, 0)
                io_time += time.perf_counter() - io_t0
                start = 0
                while start < n_in_file:
                    n = min(chunk_frames, n_in_file - start)
                    io_t0 = time.perf_counter()
                    fh.seek(start * H * row_stride_bytes, 1) if start > 0 else None
                    buf = fh.read(n * H * row_stride_bytes)
                    io_time += time.perf_counter() - io_t0

                    frames = _u8_to_frames(memoryview(buf), n, H, W, row_stride_bytes, bytes_per_pix, dtype, endianness)

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

def _decimate_xy(x: np.ndarray, y: np.ndarray, max_points: int = 20000):
    n = y.shape[0]
    if n <= max_points:
        return x, y
    step = int(np.ceil(n / max_points))
    idx = np.arange(0, n, step, dtype=int)
    return x[idx], y[idx]

class Viewer(QtWidgets.QMainWindow):
    def __init__(self, binned: np.ndarray, fps: float, norm_mode: str):
        super().__init__()
        self.binned = binned
        self.T, self.gh, self.gw = binned.shape
        self.fps = float(max(0.0, fps))
        self.norm_mode = norm_mode
        self.eps = 1e-6
        self.cur = 0
        self.playing = True
        self.max_traces = 8
        self.selected = []

        # time axis
        if self.fps > 0:
            self.t = np.arange(self.T, dtype=np.float32) / self.fps
            self.x_label = "Time (s)"
            self.default_window = min(self.T, int(self.fps * 1.0))  # ~1 second
        else:
            self.t = np.arange(self.T, dtype=np.float32)
            self.x_label = "Frame"
            self.default_window = min(self.T, 5000)

        # precompute global trace
        self.global_trace = binned.mean(axis=(1,2)).astype(np.float32)

        self._build_ui()
        self._connect()
        self._init_region()
        self._refresh_frame()

        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000 // max(1, 200))  # UI playback ~200 fps

    # ----- UI construction
    def _build_ui(self):
        cw = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(cw)
        self.setCentralWidget(cw)

        # LEFT: image + controls
        left_panel = QtWidgets.QWidget()
        left_v = QtWidgets.QVBoxLayout(left_panel); left_v.setContentsMargins(0,0,0,0); left_v.setSpacing(6)
        self.gw_widget = pg.GraphicsLayoutWidget()
        self.vb = self.gw_widget.addViewBox(lockAspect=True)
        self.img = ClickableImage(); self.img.setImage(self.binned[0], autoLevels=True)
        self.img.setGridShape(self.gh, self.gw)
        self.vb.addItem(self.img); self.vb.invertY(True)
        self.overlay = pg.ScatterPlotItem(size=12, pen=pg.mkPen(width=2), brush=None)
        self.vb.addItem(self.overlay)

        # Controls row
        controls = QtWidgets.QWidget(); hb = QtWidgets.QHBoxLayout(controls)
        hb.setContentsMargins(0,0,0,0); hb.setSpacing(8)
        self.frame_label = QtWidgets.QLabel(f"Frame: 0 / {self.T-1}")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.T - 1); self.slider.setValue(0); self.slider.setSingleStep(1)
        self.norm_combo = QtWidgets.QComboBox(); self.norm_combo.addItems(["divide","subtract","none"])
        self.norm_combo.setCurrentText(self.norm_mode)
        self.open_btn = QtWidgets.QPushButton("Open Folder…")
        hb.addWidget(self.frame_label, 0)
        hb.addWidget(self.slider, 1)
        hb.addWidget(QtWidgets.QLabel("Normalize:"), 0)
        hb.addWidget(self.norm_combo, 0)
        hb.addWidget(self.open_btn, 0)

        left_v.addWidget(self.gw_widget, 1)
        left_v.addWidget(controls, 0)
        root.addWidget(left_panel, 2)

        # RIGHT: plots (ROI traces + overview w/ region)
        right = pg.GraphicsLayoutWidget()
        root.addWidget(right, 3)

        # ROI traces
        self.trace_plot = right.addPlot(title="ROI Traces (normalized)")
        self.trace_plot.showGrid(x=True, y=True, alpha=0.3)
        self.trace_plot.setLabel('bottom', self.x_label)
        self.trace_items = []
        for _ in range(self.max_traces):
            it = pg.PlotCurveItem()
            it.hide()
            self.trace_plot.addItem(it)
            self.trace_items.append(it)

        # Overview (global mean) + region
        right.nextRow()
        self.overview = right.addPlot(title="Overview (Global Mean)")
        self.overview.showGrid(x=True, y=True, alpha=0.3)
        self.overview.setLabel('bottom', self.x_label)
        x, y = _decimate_xy(self.t, self.global_trace)
        self.over_curve = pg.PlotCurveItem(x, y)
        self.overview.addItem(self.over_curve)
        self.region = pg.LinearRegionItem(movable=True)
        self.overview.addItem(self.region)

        # Menubar
        file_menu = self.menuBar().addMenu("&File")
        act_open = QtWidgets.QAction("Open Folder…", self); act_open.triggered.connect(self._open_folder_dialog)
        act_exit = QtWidgets.QAction("Exit", self); act_exit.triggered.connect(self.close)
        file_menu.addAction(act_open); file_menu.addSeparator(); file_menu.addAction(act_exit)

        # Status bar
        self.statusBar().showMessage("Click image to toggle ROI traces; drag region in overview to zoom.")

        # Shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.toggle_play)
        QtWidgets.QShortcut(QtGui.QKeySequence("C"), self, activated=self.clear_traces)

    def _connect(self):
        self.slider.valueChanged.connect(self._set_frame_index)
        self.img.clicked.connect(self._toggle_trace_at)
        self.region.sigRegionChanged.connect(self._region_changed)
        self.norm_combo.currentTextChanged.connect(self._norm_changed)
        self.open_btn.clicked.connect(self._open_folder_dialog)

    # ----- Data / region
    def _init_region(self):
        # initial region width ~1s (or default_window frames)
        start = 0
        stop = min(self.T-1, start + self.default_window)
        self._set_region_indices(start, stop)

    def _get_region_indices(self):
        rgn = self.region.getRegion()  # in same units as overview's x axis (t)
        a, b = float(rgn[0]), float(rgn[1])
        if self.fps > 0:
            i0 = max(0, int(np.floor(a * self.fps)))
            i1 = min(self.T-1, int(np.ceil(b * self.fps)))
        else:
            i0 = max(0, int(np.floor(a)))
            i1 = min(self.T-1, int(np.ceil(b)))
        if i1 <= i0:
            i1 = min(self.T-1, i0 + 1)
        return i0, i1

    def _set_region_indices(self, i0, i1):
        # set region in axis units
        if self.fps > 0:
            self.region.setRegion([i0 / self.fps, i1 / self.fps])
        else:
            self.region.setRegion([i0, i1])
        self._region_changed()

    def _region_changed(self):
        i0, i1 = self._get_region_indices()
        # update x-range of trace plot
        if self.fps > 0:
            self.trace_plot.setXRange(i0 / self.fps, i1 / self.fps, padding=0)
        else:
            self.trace_plot.setXRange(i0, i1, padding=0)
        self._update_traces(i0, i1)

    # ----- Playback / selection
    def toggle_play(self):
        self.playing = not self.playing

    def clear_traces(self):
        self.selected.clear()
        for it in self.trace_items: it.hide()
        self._update_overlay()

    def _tick(self):
        if not self.playing: return
        self.cur += 1
        if self.cur >= self.T: self.cur = 0
        self.slider.blockSignals(True); self.slider.setValue(self.cur); self.slider.blockSignals(False)
        # keep region anchored to current frame? optional: comment out if not desired
        # i0, i1 = self._get_region_indices()
        # width = i1 - i0
        # if self.cur >= i1 - 100:
        #     self._set_region_indices(self.cur - width + 1, self.cur + 1)
        self._refresh_frame()

    def _set_frame_index(self, idx: int):
        self.cur = int(idx); self._refresh_frame()

    def _toggle_trace_at(self, y: int, x: int):
        key = (y, x)
        if key in self.selected:
            self.selected.remove(key)
        else:
            if len(self.selected) >= self.max_traces:
                self.statusBar().showMessage(f"Max {self.max_traces} traces; press C to clear.", 3000)
                return
            self.selected.append(key)
        self._update_traces(*self._get_region_indices())
        self._update_overlay()

    # ----- Normalization
    def _norm_changed(self, s: str):
        self.norm_mode = s
        self._update_traces(*self._get_region_indices())

    def _norm(self, y: np.ndarray) -> np.ndarray:
        g = self.global_trace
        if self.norm_mode == "divide":
            return y / (g + self.eps)
        elif self.norm_mode == "subtract":
            return y - g
        else:
            return y

    # ----- Updates
    def _refresh_frame(self):
        self.img.setImage(self.binned[self.cur], autoLevels=False)
        self.frame_label.setText(f"Frame: {self.cur} / {self.T-1}")

    def _update_overlay(self):
        if not self.selected:
            self.overlay.setData([], [])
            return
        xs = [x + 0.5 for (_, x) in self.selected]
        ys = [y + 0.5 for (y, _) in self.selected]
        self.overlay.setData(xs, ys)

    def _update_traces(self, i0: int, i1: int):
        # slice window
        xwin = self.t[i0:i1]
        for i, it in enumerate(self.trace_items):
            if i < len(self.selected):
                y, x = self.selected[i]
                raw = self.binned[:, y, x]
                y_norm = self._norm(raw)[i0:i1]
                xdec, ydec = _decimate_xy(xwin, y_norm, max_points=20000)
                it.setData(xdec, ydec)
                it.show()
            else:
                it.hide()

    # ----- File open
    def _open_folder_dialog(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder with batch_*.raw")
        if not d:
            return
        # ask for dims
        ok = False
        while True:
            text, ok = QtWidgets.QInputDialog.getText(
                self, "Frame dimensions",
                "Enter W,H,FPF (e.g., 1024,608,10):",
                text="1024,608,10"
            )
            if not ok: return
            try:
                W,H,FPF = [int(s.strip()) for s in text.split(",")]
                break
            except Exception:
                QtWidgets.QMessageBox.warning(self, "Invalid", "Please enter three integers like 1024,608,10")
        # optional fps
        fps, ok2 = QtWidgets.QInputDialog.getDouble(
            self, "FPS (optional)", "Frames per second (0 = show frames):", value=self.fps or 0.0, min=0.0, max=1e6, decimals=2
        )
        if not ok2: fps = self.fps
        # load
        try:
            binned = load_and_bin(Path(d), W, H, FPF,
                                  expected_frames=0,  # load all
                                  by=max(1, self.gh) if False else 32,  # not used: we rebased below; keep default in main
                                  bx=32,
                                  dtype="uint8", endianness="little",
                                  header_bytes=0, row_stride_bytes=W,
                                  io_mode="memmap", chunk_frames=256)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load error", str(e))
            return
        # replace data
        self._replace_data(binned, fps)

    def _replace_data(self, binned: np.ndarray, fps: float):
        self.binned = binned
        self.T, self.gh, self.gw = binned.shape
        self.fps = float(max(0.0, fps))
        if self.fps > 0:
            self.t = np.arange(self.T, dtype=np.float32) / self.fps
            self.x_label = "Time (s)"
            self.default_window = min(self.T, int(self.fps * 1.0))
        else:
            self.t = np.arange(self.T, dtype=np.float32)
            self.x_label = "Frame"
            self.default_window = min(self.T, 5000)
        self.global_trace = binned.mean(axis=(1,2)).astype(np.float32)

        # reset image & slider
        self.cur = 0
        self.slider.setMaximum(max(0, self.T-1))
        self.slider.setValue(0)
        self.img.setGridShape(self.gh, self.gw)
        self._update_overlay()

        # update plots
        self.trace_plot.setLabel('bottom', self.x_label)
        x, y = _decimate_xy(self.t, self.global_trace)
        self.over_curve.setData(x, y)
        self._init_region()
        self._refresh_frame()

# --------------- Main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="", help="Folder with batch_*.raw; if omitted, a dialog opens")
    ap.add_argument("--frame_w", type=int, default=1024)
    ap.add_argument("--frame_h", type=int, default=608)
    ap.add_argument("--frames_per_file", type=int, default=10)
    ap.add_argument("--expected_frames", type=int, default=0)
    ap.add_argument("--bin_y", type=int, default=32)
    ap.add_argument("--bin_x", type=int, default=32)
    ap.add_argument("--fps", type=float, default=0.0, help="Frames per second; 0 = use frame index")
    # RAW parsing options
    ap.add_argument("--dtype", choices=["uint8", "uint16"], default="uint8")
    ap.add_argument("--endianness", choices=["little", "big"], default="little")
    ap.add_argument("--header_bytes", type=int, default=0)
    ap.add_argument("--row_stride_bytes", type=int, default=0,
                    help="Bytes per row incl. padding; default = frame_w * bytes/pixel")
    # Performance
    ap.add_argument("--chunk_frames", type=int, default=256)
    ap.add_argument("--io_mode", choices=["memmap","read"], default="memmap")
    # Normalization
    ap.add_argument("--norm_mode", choices=["divide","subtract","none"], default="divide")

    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    bytes_per_pix = 1 if args.dtype == "uint8" else 2
    row_stride = args.row_stride_bytes if args.row_stride_bytes > 0 else args.frame_w * bytes_per_pix

    if run_dir and run_dir.is_dir():
        binned = load_and_bin(run_dir, args.frame_w, args.frame_h, args.frames_per_file,
                              args.expected_frames, args.bin_y, args.bin_x,
                              args.dtype, args.endianness, args.header_bytes, row_stride,
                              io_mode=args.io_mode, chunk_frames=args.chunk_frames)
    else:
        # start with a tiny placeholder for GUI; user can File→Open Folder…
        binned = np.zeros((100, max(1,args.frame_h//args.bin_y), max(1,args.frame_w//args.bin_x)), dtype=np.float32)

    app = QtWidgets.QApplication(sys.argv)
    w = Viewer(binned, fps=args.fps, norm_mode=args.norm_mode)
    w.setWindowTitle(f"Binned Frames Viewer v7 ({QT_BINDING}) — ROI normalized by global ({args.norm_mode})")
    w.resize(1500, 850)
    w.show()

    # if run_dir wasn't valid, immediately prompt
    if not (run_dir and run_dir.is_dir()):
        QtCore.QTimer.singleShot(100, w._open_folder_dialog)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
