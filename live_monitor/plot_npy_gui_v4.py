#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from matplotlib.gridspec import GridSpec
from typing import List, Tuple, Optional

# =======================
# USER-TUNABLE DEFAULTS
# =======================
DEFAULT_ROWS = 5          # <-- change number of points in Y (rows) here
DEFAULT_COLS = 6          # <-- change number of points in X (cols) here
BOTTOM_SCALE = 4       # <-- make bottom traces taller (GridSpec height ratio)

def auto_grid_channels(height: int, width: int, rows: int = 5, cols: int = 6) -> List[Tuple[int, int]]:
    """Pick evenly spaced (y, x) using strides int(height/rows), int(width/cols)."""
    sy = max(1, int(height / rows))
    sx = max(1, int(width / cols))
    sel: List[Tuple[int, int]] = []
    # center each cell a bit
    y0 = min(height - 1, max(0, sy // 2))
    x0 = min(width - 1,  max(0, sx // 2))
    for r in range(rows):
        for c in range(cols):
            y = int(min(height - 1, y0 + r * sy))
            x = int(min(width - 1,  x0 + c * sx))
            sel.append((y, x))
    return sel

class StackGUI:
    def __init__(self, stack: np.ndarray, fps: float = 5000.0, rows: int = DEFAULT_ROWS, cols: int = DEFAULT_COLS):
        assert stack.ndim == 3, "Expected stack of shape (T, Y, X)"
        self.stack = stack.astype(float, copy=False)
        self.T, self.Y, self.X = self.stack.shape
        self.fps = float(fps)

        # grid parameters (rows/cols) and global pixel offsets
        self.rows = int(rows)
        self.cols = int(cols)
        self.off_y = 0
        self.off_x = 0

        # time axis + global average
        self.t_all = np.arange(self.T) / self.fps
        self.global_mean = self.stack.mean(axis=(1, 2))

        # initial window (0 .. ~100 frames or to end)
        self.a = 0
        self.b = min(self.T, 100)

        # auto-pick evenly spaced channels
        self.sel = auto_grid_channels(self.Y, self.X, rows=self.rows, cols=self.cols)
        self._apply_offsets_to_sel()

        # ---- Layout: give MORE space to the bottom traces ----
        self.fig = plt.figure(figsize=(15, 9))
        gs = GridSpec(
            3, 2,
            height_ratios=[1.0, 0.7, BOTTOM_SCALE],
            width_ratios=[1.0, 1.0],
            hspace=0.30, wspace=0.20
        )

        self.ax_img  = self.fig.add_subplot(gs[0, 0])  # top-left: first frame
        self.ax_over = self.fig.add_subplot(gs[0, 1])  # top-right: global mean + ROI
        self.ax_help = self.fig.add_subplot(gs[1, 0])  # middle-left: help text
        self.ax_void = self.fig.add_subplot(gs[1, 1])  # middle-right: empty
        self.ax_grid = self.fig.add_subplot(gs[2, :])  # bottom: grid of traces

        # --- Top-left: first frame image + colorbar + selection points
        first = self.stack[0]
        self.im = self.ax_img.imshow(first, origin='upper', aspect='auto')
        self.ax_img.set_title("First frame (t=0)")
        self.ax_img.set_xlabel("x"); self.ax_img.set_ylabel("y")
        cb = self.fig.colorbar(self.im, ax=self.ax_img, fraction=0.046, pad=0.04)
        cb.set_label("Intensity", rotation=90)

        # overlay selected channel locations (store handles to update later)
        xs = [x for (y, x) in self.sel]
        ys = [y for (y, x) in self.sel]
        self.sc1 = self.ax_img.scatter(xs, ys, s=48, facecolors='none', edgecolors='k', linewidths=1.3)
        self.sc2 = self.ax_img.scatter(xs, ys, s=24, facecolors='none', edgecolors='w', linewidths=1.0)

        # --- Top-right: global average with draggable span
        (self.line_over,) = self.ax_over.plot(self.t_all, self.global_mean, lw=0.9)
        self.ax_over.set_title("Global mean over (Y,X)")
        self.ax_over.set_xlabel("Time (s)"); self.ax_over.set_ylabel("Mean (raw)")
        self.ax_over.set_xlim(self.t_all[0], self.t_all[-1])
        self.span = self.ax_over.axvspan(self.a/self.fps, self.b/self.fps, color='orange', alpha=0.2)
        try:
            self.selector = SpanSelector(self.ax_over, self.on_select, 'horizontal',
                                         useblit=True, minspan=1,
                                         props=dict(alpha=0.15, facecolor='orange'),
                                         onmove_callback=self.on_move)
        except TypeError:
            self.selector = SpanSelector(self.ax_over, self.on_select, 'horizontal')

        # --- Middle-left: help text (now includes grid controls)
        self.ax_help.set_axis_off()
        self.ax_help.text(0.02, 0.62,
                          "Time window: ←/→ shift, Shift+←/→ big shift, [ / ] shrink/expand, q quit\n"
                          "Grid move: Ctrl+Arrow (↑/↓/←/→) to shift all picks\n"
                          "Grid size: R/r increase/decrease rows, C/c increase/decrease cols\n"
                          "Taller bottom: edit BOTTOM_SCALE near top of the file.",
                          va='center', ha='left', fontsize=10)

        # --- Middle-right: empty
        self.ax_void.set_axis_off()

        # --- Bottom: grid of traces, only bottom row shows x-axis
        self.ax_grid.set_axis_off()
        self.ax_cells: List[plt.Axes] = []
        self.lines: List[plt.Line2D] = []
        self.labels: List[plt.Text] = []
        self._build_trace_grid(self.rows, self.cols)

        self.fig.suptitle("Stack Explorer – (first frame + colorbar & picks, global mean, and grid traces)", fontsize=14)
        self._update_window(self.a, self.b)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()

    # ---- selection utilities ----
    def _apply_offsets_to_sel(self):
        new_sel = []
        for (y, x) in auto_grid_channels(self.Y, self.X, rows=self.rows, cols=self.cols):
            yy = int(np.clip(y + self.off_y, 0, self.Y - 1))
            xx = int(np.clip(x + self.off_x, 0, self.X - 1))
            new_sel.append((yy, xx))
        self.sel = new_sel

    def _update_markers(self):
        xs = [x for (y, x) in self.sel]
        ys = [y for (y, x) in self.sel]
        # PathCollection.set_offsets expects Nx2 array
        import numpy as _np
        offs = _np.c_[xs, ys]
        self.sc1.set_offsets(offs)
        self.sc2.set_offsets(offs)

    def _rebuild_trace_grid(self):
        # remove existing inset axes
        for ax in self.ax_cells:
            try: ax.remove()
            except Exception: pass
        self.ax_cells.clear(); self.lines.clear(); self.labels.clear()
        self._build_trace_grid(self.rows, self.cols)
        self._redraw_traces()

    def _build_trace_grid(self, rows: int, cols: int):
        n = rows * cols
        for i in range(n):
            r = i // cols
            c = i % cols
            left = c/cols
            bottom = 0.04 + (rows-1-r)/rows*0.94
            width = 1.0/cols*0.98
            height = 0.94/rows
            ax = self.ax_grid.inset_axes([left+0.01, bottom, width-0.02, height-0.03])
            ax.set_facecolor('white')
            # Defer x ticks to bottom row only (handled in redraw)
            ax.set_yticks([])
            (line,) = ax.plot([], [], lw=0.9)
            label = ax.text(0.02, 0.06, "",
                            transform=ax.transAxes, fontsize=8,
                            bbox=dict(fc=(1,1,1,0.55), ec='none', pad=0.3))
            self.ax_cells.append(ax); self.lines.append(line); self.labels.append(label)

    # ----- ROI events -----
    def on_select(self, xmin, xmax):
        if xmax < xmin: xmin, xmax = xmax, xmin
        a = int(np.clip(np.floor(xmin * self.fps), 0, self.T - 1))
        b = int(np.clip(np.ceil (xmax * self.fps), 1, self.T))
        if b <= a: b = min(self.T, a + 1)
        self._update_window(a, b)

    def on_move(self, xmin, xmax):
        self.on_select(xmin, xmax)

    def on_key(self, event):
        width = self.b - self.a

        # --------- time window controls ---------
        if event.key in ('left','right','shift+left','shift+right'):
            step = 5 if 'shift+' not in event.key else 50
            if 'left' in event.key:
                a = self.a - step; b = self.b - step
            else:
                a = self.a + step; b = self.b + step
            a = int(np.clip(a, 0, max(0, self.T - width))); b = a + width
            self._update_window(a, b)
            return

        if event.key in ('[',']'):
            if event.key == '[':
                new_w = max(1, width - 5); center = (self.a + self.b)//2
            else:
                new_w = min(self.T, width + 5); center = (self.a + self.b)//2
            a = int(np.clip(center - new_w//2, 0, self.T - new_w)); b = a + new_w
            self._update_window(a, b)
            return

        # --------- grid move with Ctrl+Arrows ---------
        if event.key in ('ctrl+left','ctrl+right','ctrl+up','ctrl+down',
                         'ctrl+shift+left','ctrl+shift+right','ctrl+shift+up','ctrl+shift+down'):
            # step size ~ 1/5 of cell size; with shift: 5x larger
            sy = max(1, int(self.Y / max(1, self.rows) / 5))
            sx = max(1, int(self.X / max(1, self.cols) / 5))
            mul = 5 if 'shift+' in event.key else 1
            dy = 0; dx = 0
            if 'left' in event.key:  dx = -sx * mul
            if 'right' in event.key: dx =  sx * mul
            if 'up' in event.key:    dy = -sy * mul
            if 'down' in event.key:  dy =  sy * mul
            self.off_y = int(np.clip(self.off_y + dy, -(self.Y-1), (self.Y-1)))
            self.off_x = int(np.clip(self.off_x + dx, -(self.X-1), (self.X-1)))
            self._apply_offsets_to_sel()
            self._update_markers()
            self._redraw_traces()
            return

        # --------- grid size (rows/cols) ---------
        if event.key in ('R','r','C','c'):
            if event.key == 'R':
                self.rows = min(self.rows + 1, max(1, self.Y))  # practical upper bound
            elif event.key == 'r':
                self.rows = max(1, self.rows - 1)
            elif event.key == 'C':
                self.cols = min(self.cols + 1, max(1, self.X))
            elif event.key == 'c':
                self.cols = max(1, self.cols - 1)

            self._apply_offsets_to_sel()
            self._rebuild_trace_grid()
            self._update_markers()
            return

        if event.key in ('q','escape'):
            plt.close(self.fig)

    # ----- Render -----
    def _update_window(self, a: int, b: int):
        self.a = int(np.clip(a, 0, self.T - 1))
        self.b = int(np.clip(b, self.a + 1, self.T))
        try:
            self.span.remove()
        except Exception:
            pass
        self.span = self.ax_over.axvspan(self.a/self.fps, self.b/self.fps, color='orange', alpha=0.2)
        self._redraw_traces()
        self.fig.canvas.draw_idle()

    def _redraw_traces(self):
        t0, t1 = self.a / self.fps, self.b / self.fps
        t = np.arange(self.a, self.b) / self.fps

        # ensure sel matches rows*cols
        needed = self.rows * self.cols
        if len(self.sel) != needed:
            self._apply_offsets_to_sel()

        for i, ax in enumerate(self.ax_cells):
            if i >= len(self.sel):
                break
            y, x = self.sel[i]
            yseg = self.stack[self.a:self.b, y, x]
            self.lines[i].set_data(t, yseg)

            # label with y,x
            self.labels[i].set_text(f"{y},{x}")

            # per-panel y limits with padding
            y_min = float(np.min(yseg)); y_max = float(np.max(yseg))
            span = y_max - y_min
            if span <= 0:
                pad = max(1e-6, 0.01 * max(1.0, abs(y_max)))
                y_lo, y_hi = y_min - pad, y_max + pad
            else:
                pad = 0.06 * span
                y_lo, y_hi = y_min - pad, y_max + pad

            ax.set_xlim(t0, t1)
            ax.set_ylim(y_lo, y_hi)

            # Only bottom row shows x-axis ticks/labels
            r = i // self.cols
            if r == self.rows - 1:
                ax.set_xticks([t0, (t0+t1)/2, t1])
                ax.set_xticklabels([f"{t0:.2f}", f"{(t0+t1)/2:.2f}", f"{t1:.2f}"], fontsize=8)
                ax.set_xlabel("Time (s)", fontsize=8, labelpad=2)
            else:
                ax.set_xticks([])

        self.ax_over.set_title(
            f"Global mean | Window [{t0:.3f}s, {t1:.3f}s] (len={(t1-t0):.3f}s)"
        )

def _file_dialog_pick() -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select stack .npy (shape (T, Y, X))",
        filetypes=[("NumPy arrays", "*.npy"), ("All files", "*.*")],
    )
    root.update()
    root.destroy()
    if not path:
        return None
    return Path(path)

def main():
    ap = argparse.ArgumentParser(description=".npy (T,Y,X) stack explorer with movable grid & resizable rows/cols.")
    ap.add_argument("npy_path", nargs="?", default=None, help="Path to stack.npy; if omitted, a file dialog will open.")
    ap.add_argument("--fps", type=float, default=5000.0, help="Sampling rate (Hz)")
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Rows of channels (default 5)")
    ap.add_argument("--cols", type=int, default=DEFAULT_COLS, help="Cols of channels (default 6)")
    args = ap.parse_args()

    # Pick file via dialog if not given
    if args.npy_path is None:
        picked = _file_dialog_pick()
        if picked is None:
            print("No file selected.")
            return
        npy_path = picked
    else:
        npy_path = Path(args.npy_path)

    stack = np.load(npy_path)  # Expect (T, Y, X)
    _ = StackGUI(stack, fps=args.fps, rows=args.rows, cols=args.cols)

if __name__ == "__main__":
    main()
