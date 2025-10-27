#!/usr/bin/env python3
"""
plot_cardiomyocyte_v4_1b_gui.py

Same as v4.1 but fixes Matplotlib compatibility:
- Tries SpanSelector(..., props=...), falls back to rectprops=... if needed.
- If both fail, constructs without styling and still supports on-move updates.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt, iirnotch

def butter_filter(data: np.ndarray, fs: float, kind: str, cutoff, order: int = 2) -> np.ndarray:
    nyq = 0.5 * fs
    if kind == 'highpass':
        Wn = np.clip(float(cutoff) / nyq, 1e-6, 0.999999)
        b, a = butter(order, Wn, btype='highpass')
    elif kind == 'lowpass':
        Wn = np.clip(float(cutoff) / nyq, 1e-6, 0.999999)
        b, a = butter(order, Wn, btype='lowpass')
    else:
        lo, hi = cutoff
        Wn = [np.clip(lo / nyq, 1e-6, 0.9999), np.clip(hi / nyq, 1e-6, 0.9999)]
        b, a = butter(order, Wn, btype='bandpass')
    return filtfilt(b, a, data)

def notch_60(data: np.ndarray, fs: float, f0: float = 60.0, Q: float = 30.0) -> np.ndarray:
    nyq = 0.5 * fs
    if f0 >= nyq or f0 <= 0.0:
        return data.copy()
    w0 = f0 / nyq
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, data)

class WindowGUI:
    def __init__(self, traces: np.ndarray, rows: int, cols: int, fps: float,
                 window_len_pts: int = 100, step_small: int = 5, step_big: int = 50,
                 elec_hp: float = 80.0, scale: float = 1.0, outfile_stem: Path | None = None):
        self.traces = traces.astype(float)
        self.traces_dm = self.traces - np.mean(self.traces, axis=1, keepdims=True)
        self.rows = rows; self.cols = cols
        self.fps = float(fps)
        self.T = traces.shape[1]; self.B = traces.shape[0]
        self.scale = float(scale)
        self.outfile_stem = outfile_stem
        self.W = int(window_len_pts)
        self.step_small = int(step_small); self.step_big = int(step_big)
        self.a = 0; self.b = min(self.T, self.W)

        # S/N prep
        self.elec = np.empty_like(self.traces)
        for i in range(self.B):
            r = butter_filter(self.traces[i], fs=self.fps, kind='highpass', cutoff=elec_hp, order=2)
            r = notch_60(r, fs=self.fps, f0=60.0, Q=30.0)
            self.elec[i] = r
        self.global_sigma = np.std(self.elec, axis=1, ddof=1) + 1e-12

        # Overview (demeaned p30)
        self.overview = np.percentile(self.traces_dm, 30.0, axis=0)

        # Figure
        self.fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(3, 1, height_ratios=[1, 0.05, 2.2], hspace=0.25)
        self.ax_over = self.fig.add_subplot(gs[0, 0])
        self.ax_bar  = self.fig.add_subplot(gs[1, 0])
        self.ax_grid = self.fig.add_subplot(gs[2, 0])

        t_all = np.arange(self.T) / self.fps
        (self.line_over,) = self.ax_over.plot(t_all, self.overview * self.scale, lw=0.7)
        self.ax_over.set_ylabel("Overview (p30, demeaned)")
        self.ax_over.set_xlim(t_all[0], t_all[-1])
        self.span = self.ax_over.axvspan(self.a/self.fps, self.b/self.fps, color='orange', alpha=0.2)

        # SpanSelector with version-safe kwargs
        self.selector = None
        try:
            self.selector = SpanSelector(self.ax_over, self.on_select, 'horizontal',
                                         useblit=True, minspan=1,
                                         props=dict(alpha=0.15, facecolor='orange'),
                                         onmove_callback=self.on_move)
        except TypeError:
            try:
                self.selector = SpanSelector(self.ax_over, self.on_select, 'horizontal',
                                             useblit=True, minspan=1,
                                             rectprops=dict(alpha=0.15, facecolor='orange'))
                # Older versions lack onmove_callback; simulate with onselect only
            except TypeError:
                # Last resort: minimal args (no styling)
                self.selector = SpanSelector(self.ax_over, self.on_select, 'horizontal')

        # Grid setup with pre-created artists
        self.ax_grid.cla(); self.ax_grid.set_axis_off()
        self.ax_cells, self.lines, self.lbl_sn = [], [], []
        for r in range(self.rows):
            for c in range(self.cols):
                left = c / self.cols
                bottom = 0.02 + (self.rows - 1 - r) / self.rows * 0.96
                width = 1.0 / self.cols * 0.98
                height = 0.96 / self.rows
                ax = self.ax_grid.inset_axes([left + 0.01, bottom, width - 0.02, height - 0.02])
                ax.set_facecolor('white'); ax.set_xticks([]); ax.set_yticks([])
                (line,) = ax.plot([], [], lw=0.7)
                t_sn = ax.text(0.02, 0.77, "", transform=ax.transAxes, fontsize=7,
                               bbox=dict(fc=(1,1,1,0.65), ec='none', pad=0.2))
                ax.text(0.02, 0.04, f"{r},{c}", transform=ax.transAxes, fontsize=7,
                        bbox=dict(fc=(1,1,1,0.5), ec='none', pad=0.2))
                self.ax_cells.append(ax); self.lines.append(line); self.lbl_sn.append(t_sn)

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.update_window(self.a)

        self.ax_bar.set_axis_off()
        self.ax_bar.text(0.01, 0.5,
            "Drag to position fixed window. ←/→: move 5; Shift+←/→: move 50; s: save PNG; q/ESC: quit.",
            va='center', ha='left')
        self.fig.suptitle("Cardiomyocyte v4.1b – interactive window S/N explorer", fontsize=14)
        plt.show()

    # callbacks
    def on_select(self, xmin, xmax):
        center = 0.5 * (xmin + xmax)
        start = int(round(center * self.fps)) - self.W // 2
        self.update_window(start)

    def on_move(self, xmin, xmax):
        # If the current Matplotlib provided onmove_callback, this will run for continuous updates.
        center = 0.5 * (xmin + xmax)
        start = int(round(center * self.fps)) - self.W // 2
        self.update_window(start)

    def on_key(self, event):
        if event.key in ('left',):
            self.update_window(self.a - self.step_small)
        elif event.key in ('right',):
            self.update_window(self.a + self.step_small)
        elif event.key in ('shift+left',):
            self.update_window(self.a - self.step_big)
        elif event.key in ('shift+right',):
            self.update_window(self.a + self.step_big)
        elif event.key == 's':
            self.save_png()
        elif event.key in ('q', 'escape'):
            plt.close(self.fig)

    # core
    def update_window(self, start: int):
        self.a = max(0, min(int(start), self.T - self.W))
        self.b = self.a + self.W
        try:
            self.span.remove()
        except Exception:
            pass
        self.span = self.ax_over.axvspan(self.a/self.fps, self.b/self.fps, color='orange', alpha=0.2)
        self.redraw_grid()
        self.fig.canvas.draw_idle()

    def redraw_grid(self):
        Y = (self.traces_dm[:, self.a:self.b] * self.scale).astype(float)
        y_min = float(np.min(Y)); y_max = float(np.max(Y))
        pad = 0.05 * max(y_max - y_min, 1e-12)
        y_lo, y_hi = y_min - pad, y_max + pad

        t0, t1 = self.a / self.fps, self.b / self.fps
        t = np.arange(self.a, self.b) / self.fps

        early = (self.a, min(self.b, self.a + int(round(0.040 * self.fps))))
        sn = np.zeros(self.B, dtype=float)
        for i in range(self.B):
            pk = np.max(np.abs(self.elec[i, early[0]:early[1]])) if early[1] > early[0] else 0.0
            sn[i] = pk / self.global_sigma[i]

        for i, ax in enumerate(self.ax_cells):
            yseg = self.traces_dm[i, self.a:self.b] * self.scale
            self.lines[i].set_data(t, yseg)
            ax.set_xlim(t0, t1); ax.set_ylim(y_lo, y_hi)
            ax.set_xticks([t0, t1]); ax.set_xticklabels([f"{t0:.2f}", f"{t1:.2f}"])
            self.lbl_sn[i].set_text(f"S/Nₑ={sn[i]:.2f}")

        mean_sn = float(np.mean(sn))
        self.ax_over.set_title(f"Window [{t0:.3f}s, {t1:.3f}s]  len={(t1-t0):.3f}s   mean S/Nₑ={mean_sn:.2f}")

    def save_png(self):
        if self.outfile_stem is None:
            return
        t0, t1 = self.a / self.fps, self.b / self.fps
        png = self.outfile_stem.with_name(self.outfile_stem.name + f"_win_{t0:.3f}_{t1:.3f}.png")
        self.fig.savefig(png, dpi=200)
        print(f"[Saved] {png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz_path', nargs='?', help='Path to exp_block_data.npz (dialog if omitted)')
    ap.add_argument('--fps', type=float, default=None, help='Sampling rate override (Hz)')
    ap.add_argument('--win_len', type=int, default=100, help='Window length in frames (default 100)')
    ap.add_argument('--step', type=int, default=5, help='Arrow key step in frames')
    ap.add_argument('--big_step', type=int, default=50, help='Shift+Arrow step in frames')
    ap.add_argument('--elec_hp', type=float, default=80.0, help='High-pass for electrical S/N (Hz)')
    args = ap.parse_args()

    if args.npz_path:
        npz_path = Path(args.npz_path)
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        p = filedialog.askopenfilename(title="Select exp_block_data.npz",
                                       filetypes=[("NumPy NPZ","*.npz"), ("All files","*.*")])
        if not p: raise SystemExit("No file selected")
        npz_path = Path(p)

    Z = np.load(npz_path, allow_pickle=True)
    traces = Z['trace_boxes'][0].astype(float)
    rows = int(Z['rows']); cols = int(Z['cols'])
    fps = float(args.fps if args.fps is not None else (Z['fps'].item() if 'fps' in Z else 5000.0))
    stem = npz_path.with_suffix('')

    gui = WindowGUI(traces=traces, rows=rows, cols=cols, fps=fps,
                    window_len_pts=args.win_len, step_small=args.step, step_big=args.big_step,
                    elec_hp=args.elec_hp, scale=1.0, outfile_stem=stem)
    plt.show()

if __name__ == '__main__':
    main()
