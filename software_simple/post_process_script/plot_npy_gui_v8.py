#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# =======================
# USER-TUNABLE DEFAULTS
# =======================
DEFAULT_ROWS   = 5         # number of grid rows (Y)
DEFAULT_COLS   = 6         # number of grid cols (X)
DEFAULT_KERNEL = 3         # neighborhood size for averaging (odd int: 1,3,5,...)
FIGSIZE        = (22, 11)  # overall window size
HEIGHT_RATIOS  = [0.40, 0.10, 0.50]  # [top 40%, middle 10%, bottom 50%]
TRACE_ROW_SCALE = 1.2      # each small trace panel is 1.2x taller than baseline
FAST_SHIFT_STRIDE = 4      # during Shift-scan, decimate time by this stride for speed
SMALL_LINEWIDTH = 0.6      # thinner traces to reduce visual sticking

def auto_grid_channels(height: int, width: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    """Pick evenly spaced (y, x) using strides int(height/rows), int(width/cols)."""
    sy = max(1, int(height / max(1, rows)))
    sx = max(1, int(width  / max(1, cols)))
    sel: List[Tuple[int, int]] = []
    y0 = min(height - 1, max(0, sy // 2))
    x0 = min(width  - 1, max(0, sx // 2))
    for r in range(rows):
        for c in range(cols):
            y = int(min(height - 1, y0 + r * sy))
            x = int(min(width  - 1, x0 + c * sx))
            sel.append((y, x))
    return sel

class StackGUI:
    def __init__(self, npy_path: Path, stack: np.ndarray, fps: float, rows: int, cols: int, kernel: int):
        assert stack.ndim == 3, "Expected stack of shape (T, Y, X)"
        self.npy_path = npy_path
        self.stack = stack.astype(float, copy=False)
        self.T, self.Y, self.X = self.stack.shape
        self.fps = float(fps)

        # grid parameters and global pixel offsets
        self.rows = int(rows)
        self.cols = int(cols)
        self.off_y = 0
        self.off_x = 0

        # kernel (odd) for neighborhood averaging
        self.kernel = max(1, int(kernel))
        if self.kernel % 2 == 0:  # force odd
            self.kernel += 1

        # time axis + global average
        self.t_all = np.arange(self.T) / self.fps
        self.global_mean = self.stack.mean(axis=(1, 2))

        # initial time window (first ~100 frames)
        self.a = 0
        self.b = min(self.T, 100)

        # auto-pick evenly spaced channels
        self.sel = auto_grid_channels(self.Y, self.X, rows=self.rows, cols=self.cols)
        self._apply_offsets_to_sel()

        # Grid-move modifier: 'm' + arrows (alt: WASD without holding anything)
        self.move_hold = False

        # ---- Layout per spec ----
        self.fig = plt.figure(figsize=FIGSIZE)
        gs = GridSpec(
            3, 2,
            height_ratios=HEIGHT_RATIOS,
            width_ratios=[1.0, 1.0],
            hspace=0.28, wspace=0.22
        )

        self.ax_img  = self.fig.add_subplot(gs[0, 0])  # top-left: first frame
        self.ax_over = self.fig.add_subplot(gs[0, 1])  # top-right: global mean (only)
        self.ax_help = self.fig.add_subplot(gs[1, 0])  # middle-left: help text
        self.ax_void = self.fig.add_subplot(gs[1, 1])  # middle-right: empty
        self.ax_grid = self.fig.add_subplot(gs[2, :])  # bottom: grid of traces

        # --- Top-left: first frame image + colorbar + selection points
        first = self.stack[0]
        self.im = self.ax_img.imshow(first, origin='upper', aspect='auto', cmap='inferno')  # FIRE-like
        self.ax_img.set_title("First frame (t=0) — markers show grid picks; squares show kernel size")
        self.ax_img.set_xlabel("x"); self.ax_img.set_ylabel("y")
        cb = self.fig.colorbar(self.im, ax=self.ax_img, fraction=0.046, pad=0.04)
        cb.set_label("Intensity", rotation=90)

        # overlay selected channel locations (store handles to update later)
        xs = [x for (y, x) in self.sel]
        ys = [y for (y, x) in self.sel]
        self.sc1 = self.ax_img.scatter(xs, ys, s=50, facecolors='none', edgecolors='k', linewidths=1.0)
        self.sc2 = self.ax_img.scatter(xs, ys, s=26, facecolors='none', edgecolors='w', linewidths=0.9)

        # overlay kernel preview rectangles for each pick
        self.rects: List[mpatches.Rectangle] = []
        self._rebuild_kernel_rects()

        # --- Top-right: global mean only, with draggable time window
        (self.line_over,) = self.ax_over.plot(self.t_all, self.global_mean, lw=0.8, label="Global mean")
        self.ax_over.legend(loc="upper right", fontsize=9, frameon=True)
        self.ax_over.set_title("Global mean (drag to zoom window)")
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

        # --- Middle-left: help text (updated)
        self.ax_help.set_axis_off()
        self._render_help_text()

        # --- Middle-right: empty
        self.ax_void.set_axis_off()

        # --- Bottom: grid of traces, only bottom row shows x-axis
        self.ax_grid.set_axis_off()
        self.ax_cells: List[plt.Axes] = []
        self.lines: List[plt.Line2D] = []
        self.labels: List[plt.Text] = []
        self._build_trace_grid(self.rows, self.cols)

        # Window title includes filename
        fname = self.npy_path.name
        self.fig.suptitle(f"Stack Explorer — {fname} — kernel-averaged traces, fast time scan (WASD/m+arrows to move grid)", fontsize=14)
        print(f"Loaded: {self.npy_path}  shape={self.stack.shape}  fps={self.fps}")

        self._update_window(self.a, self.b)

        # Key handler (press + release to support holding 'm')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)

        # Try to maximize the window if backend supports it
        try:
            mng = plt.get_current_fig_manager()
            if hasattr(mng, "window"):
                try:
                    mng.window.state("zoomed")  # TkAgg on Windows
                except Exception:
                    try:
                        mng.window.showMaximized()  # Qt5Agg
                    except Exception:
                        pass
        except Exception:
            pass

        plt.show()

    # ------------------------ utilities ------------------------
    def _render_help_text(self):
        self.ax_help.clear()
        self.ax_help.set_axis_off()
        text = (
            "Time window: ←/→ (±5), Shift+←/→ (±50, decimated), [ / ] shrink/expand, Z zoom-in x0.5, X zoom-out x2\n"
            "Move grid (no lag): WASD (Shift for bigger steps)  •  Or HOLD 'm' + arrows (Shift bigger)\n"
            "Grid size: R/r increase/decrease rows, C/c increase/decrease cols\n"
            f"Kernel: +/- increase/decrease (odd, current = {self.kernel}) — averages N×N around each pick\n"
            "Intensity map: 'inferno' (fire)\n"
        )
        self.ax_help.text(0.02, 0.62, text, va='center', ha='left', fontsize=10)

    def _apply_offsets_to_sel(self):
        new_sel = []
        base_sel = auto_grid_channels(self.Y, self.X, rows=self.rows, cols=self.cols)
        for (y, x) in base_sel:
            yy = int(np.clip(y + self.off_y, 0, self.Y - 1))
            xx = int(np.clip(x + self.off_x, 0, self.X - 1))
            new_sel.append((yy, xx))
        self.sel = new_sel

    def _update_markers(self):
        xs = [x for (y, x) in self.sel]
        ys = [y for (y, x) in self.sel]
        import numpy as _np
        offs = _np.c_[xs, ys]
        self.sc1.set_offsets(offs)
        self.sc2.set_offsets(offs)

    def _rebuild_trace_grid(self):
        for ax in self.ax_cells:
            try: ax.remove()
            except Exception: pass
        self.ax_cells.clear(); self.lines.clear(); self.labels.clear()
        self._build_trace_grid(self.rows, self.cols)
        self._redraw_traces()

    def _build_trace_grid(self, rows: int, cols: int):
        # compute per-row panel height scaled by TRACE_ROW_SCALE, and distribute evenly
        total_area = 0.94  # area inside ax_grid we can use vertically
        base_h = total_area / rows
        h = min(total_area, base_h * TRACE_ROW_SCALE)
        vgap = 0.0 if rows == 1 else max(0.0, (total_area - rows * h) / (rows - 1))
        for i in range(rows * cols):
            r = i // cols
            c = i % cols
            left = c / cols
            # place from top to bottom
            bottom = 0.04 + (rows - 1 - r) * (h + vgap)
            width = 1.0 / cols * 0.98
            height = h * 0.98  # small padding to avoid overlap
            ax = self.ax_grid.inset_axes([left + 0.01, bottom, width - 0.02, height])
            ax.set_facecolor('white')
            ax.set_yticks([])
            (line,) = ax.plot([], [], lw=SMALL_LINEWIDTH)  # thinner lines
            label = ax.text(0.02, 0.06, "",
                            transform=ax.transAxes, fontsize=8,
                            bbox=dict(fc=(1,1,1,0.55), ec='none', pad=0.3))
            self.ax_cells.append(ax); self.lines.append(line); self.labels.append(label)

    def _rebuild_kernel_rects(self):
        # remove existing
        for r in getattr(self, "rects", []):
            try: r.remove()
            except Exception: pass
        self.rects = []
        if self.kernel <= 1:
            self.fig.canvas.draw_idle()
            return

        half = self.kernel // 2
        for (y, x) in self.sel:
            y0 = max(0, y - half); y1 = min(self.Y - 1, y + half)
            x0 = max(0, x - half); x1 = min(self.X - 1, x + half)
            # Rectangle expects [x, y, w, h] in data coords
            rect = mpatches.Rectangle((x0-0.5, y0-0.5),
                                      (x1 - x0 + 1),
                                      (y1 - y0 + 1),
                                      fill=False, lw=0.9, ec='orange', alpha=0.6)
            self.ax_img.add_patch(rect)
            self.rects.append(rect)
        self.fig.canvas.draw_idle()

    # ------------------------ ROI / signals ------------------------
    def _extract_kernel_mean_trace(self, a: int, b: int, y: int, x: int) -> np.ndarray:
        """Return time-segment [a:b] averaged over an N×N kernel centered at (y,x)."""
        if self.kernel <= 1:
            return self.stack[a:b, y, x]

        half = self.kernel // 2
        y0 = max(0, y - half); y1 = min(self.Y - 1, y + half)
        x0 = max(0, x - half); x1 = min(self.X - 1, x + half)
        seg = self.stack[a:b, y0:y1+1, x0:x1+1].mean(axis=(1, 2))
        return seg

    # ------------------------ events ------------------------
    def on_select(self, xmin, xmax):
        if xmax < xmin: xmin, xmax = xmax, xmin
        a = int(np.clip(np.floor(xmin * self.fps), 0, self.T - 1))
        b = int(np.clip(np.ceil (xmax * self.fps), 1, self.T))
        if b <= a: b = min(self.T, a + 1)
        self._update_window(a, b)

    def on_move(self, xmin, xmax):
        self.on_select(xmin, xmax)

    def on_key_press(self, event):
        # track holding 'm' for grid-move
        if event.key == 'm':
            self.move_hold = True
            return

        width = self.b - self.a

        # --------- time window controls ---------
        if event.key in ('left','right','shift+left','shift+right'):
            step = 5 if 'shift+' not in event.key else 50
            if 'left' in event.key:
                a = self.a - step; b = self.b - step
            else:
                a = self.a + step; b = self.b + step
            a = int(np.clip(a, 0, max(0, self.T - width))); b = a + width
            # During Shift, recompute with time decimation for speed
            stride = FAST_SHIFT_STRIDE if 'shift+' in event.key else 1
            self._update_window(a, b, stride=stride)
            return

        # Zoom-in by half (keep zooming)
        if event.key in ('z','Z'):
            new_w = max(1, width // 2 if width > 1 else 1)
            center = (self.a + self.b)//2
            a = int(np.clip(center - new_w//2, 0, self.T - new_w)); b = a + new_w
            self._update_window(a, b)
            return

        # Optional zoom-out by 2x
        if event.key in ('x','X'):
            new_w = min(self.T, max(1, width * 2))
            center = (self.a + self.b)//2
            a = int(np.clip(center - new_w//2, 0, self.T - new_w)); b = a + new_w
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

        # --------- grid size (rows/cols) ---------
        if event.key in ('R','r','C','c'):
            if event.key == 'R':
                self.rows = min(self.rows + 1, max(1, self.Y))
            elif event.key == 'r':
                self.rows = max(1, self.rows - 1)
            elif event.key == 'C':
                self.cols = min(self.cols + 1, max(1, self.X))
            elif event.key == 'c':
                self.cols = max(1, self.cols - 1)

            self._apply_offsets_to_sel()
            self._rebuild_trace_grid()
            self._update_markers()
            self._rebuild_kernel_rects()
            return

        # --------- kernel size ---------
        if event.key in ('+', '=','-','_'):
            if event.key in ('+','='):
                self.kernel += 2  # keep odd
            else:
                self.kernel = max(1, self.kernel - 2)
            if self.kernel % 2 == 0:
                self.kernel += 1
            self._render_help_text()
            self._rebuild_kernel_rects()
            self._redraw_traces()
            return

        # --------- MOVE GRID: WASD (no modifiers needed) ---------
        if event.key in ('w','a','s','d','W','A','S','D'):
            big = event.key.isupper()
            self._grid_move_wasd(event.key.lower(), big=big)
            return

        if event.key in ('q','escape'):
            plt.close(self.fig)

        # Arrow keys while holding 'm' move the grid
        if self.move_hold and event.key in ('up','down','left','right','shift+up','shift+down','shift+left','shift+right'):
            self._grid_move_with_arrows(event.key)
            return

    def on_key_release(self, event):
        if event.key == 'm':
            self.move_hold = False
            return
        # After Shift fast-scan, refresh with full-resolution traces
        if event.key in ('shift',):
            self._redraw_traces(stride=1)
            self.fig.canvas.draw_idle()

    # ---- movement helpers ----
    def _grid_move_wasd(self, key: str, big: bool=False):
        sy = max(1, int(self.Y / max(1, self.rows) / 5))
        sx = max(1, int(self.X / max(1, self.cols) / 5))
        mul = 5 if big else 1
        dy = 0; dx = 0
        if key == 'a': dx = -sx * mul
        if key == 'd': dx =  sx * mul
        if key == 'w': dy = -sy * mul
        if key == 's': dy =  sy * mul
        self.off_y = int(np.clip(self.off_y + dy, -(self.Y-1), (self.Y-1)))
        self.off_x = int(np.clip(self.off_x + dx, -(self.X-1), (self.X-1)))
        self._apply_offsets_to_sel()
        self._update_markers()
        self._rebuild_kernel_rects()
        self._redraw_traces()

    def _grid_move_with_arrows(self, key: str):
        sy = max(1, int(self.Y / max(1, self.rows) / 5))
        sx = max(1, int(self.X / max(1, self.cols) / 5))
        mul = 5 if 'shift+' in key else 1
        dy = 0; dx = 0
        if 'left' in key:  dx = -sx * mul
        if 'right' in key: dx =  sx * mul
        if 'up' in key:    dy = -sy * mul
        if 'down' in key:  dy =  sy * mul
        self.off_y = int(np.clip(self.off_y + dy, -(self.Y-1), (self.Y-1)))
        self.off_x = int(np.clip(self.off_x + dx, -(self.X-1), (self.X-1)))
        self._apply_offsets_to_sel()
        self._update_markers()
        self._rebuild_kernel_rects()
        self._redraw_traces()

    # ------------------------ render ------------------------
    def _update_window(self, a: int, b: int, stride: int = 1):
        self.a = int(np.clip(a, 0, self.T - 1))
        self.b = int(np.clip(b, self.a + 1, self.T))
        try:
            self.span.remove()
        except Exception:
            pass
        self.span = self.ax_over.axvspan(self.a/self.fps, self.b/self.fps, color='orange', alpha=0.2)
        self._redraw_traces(stride=stride)
        self.fig.canvas.draw_idle()

    def _redraw_traces(self, stride: int = 1):
        t_idx = np.arange(self.a, self.b, stride, dtype=int)
        if t_idx.size == 0:
            t_idx = np.array([self.a], dtype=int)
        t = t_idx / self.fps
        # ensure sel matches rows*cols
        needed = self.rows * self.cols
        if len(self.sel) != needed:
            self._apply_offsets_to_sel()

        # update each small panel with kernel-averaged trace
        for i, ax in enumerate(self.ax_cells):
            if i >= len(self.sel):
                break
            y, x = self.sel[i]
            yseg = self._extract_kernel_mean_trace(self.a, self.b, y, x)[::stride]
            self.lines[i].set_data(t, yseg)
            self.labels[i].set_text(f"{y},{x}")

            y_min = float(np.min(yseg)); y_max = float(np.max(yseg))
            span = y_max - y_min
            if span <= 0:
                pad = max(1e-6, 0.01 * max(1.0, abs(y_max)))
                y_lo, y_hi = y_min - pad, y_max + pad
            else:
                pad = 0.06 * span
                y_lo, y_hi = y_min - pad, y_max + pad

            ax.set_xlim(t[0], t[-1] if t.size > 1 else t[0] + 1/self.fps)
            ax.set_ylim(y_lo, y_hi)

            # Only bottom row shows x-axis ticks/labels
            r = i // self.cols
            if r == self.rows - 1:
                mid = (t[0] + t[-1]) / 2.0 if t.size > 1 else t[0]
                ax.set_xticks([t[0], mid, t[-1]])
                ax.set_xticklabels([f"{t[0]:.2f}", f"{mid:.2f}", f"{t[-1]:.2f}"], fontsize=8)
                ax.set_xlabel("Time (s)", fontsize=8, labelpad=2)
            else:
                ax.set_xticks([])

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
    ap = argparse.ArgumentParser(description=".npy (T,Y,X) stack explorer with grid movement, kernel-averaged traces, and fast scanning.")
    ap.add_argument("npy_path", nargs="?", default=None, help="Path to stack.npy; if omitted, a file dialog will open.")
    ap.add_argument("--fps", type=float, default=5000.0, help="Sampling rate (Hz)")
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Rows of channels (default 5)")
    ap.add_argument("--cols", type=int, default=DEFAULT_COLS, help="Cols of channels (default 6)")
    ap.add_argument("--kernel", type=int, default=DEFAULT_KERNEL, help="Odd kernel size for N×N averaging (1,3,5,...)")
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
    _ = StackGUI(npy_path, stack, fps=args.fps, rows=args.rows, cols=args.cols, kernel=args.kernel)

if __name__ == "__main__":
    main()
