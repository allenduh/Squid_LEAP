
#!/usr/bin/env python3
"""
leap_roi_tools.py
FourPointSelector and GridAdjuster
- No project-specific logic; pure UI helpers.
"""
from __future__ import annotations
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

class FourPointSelector:
    """Draggable 4-point ROI (TL, TR, BL, BR). Press Enter to accept."""
    def __init__(self, frame: np.ndarray, init_points: Optional[np.ndarray] = None):
        self.frame = frame
        self.points: list[tuple[float, float]] = []
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(frame, cmap="gray")
        self.ax.set_title("Click 4 corners TL,TR,BL,BR. Drag to adjust. Press Enter to finish.")
        self.drag_idx: Optional[int] = None
        self.accepted = False

        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        if init_points is not None and len(init_points) == 4:
            self.points = [(float(x), float(y)) for x, y in init_points]

        self.redraw()

    def _within_handle(self, event, x, y, tol=8):
        if event.x is None or event.y is None:
            return False
        disp = self.ax.transData.transform((x, y))
        dx = event.x - disp[0]; dy = event.y - disp[1]
        return (dx*dx + dy*dy) ** 0.5 <= tol

    def pick_handle(self, event) -> Optional[int]:
        for i, (x, y) in enumerate(self.points):
            if self._within_handle(event, x, y):
                return i
        return None

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        if len(self.points) < 4:
            self.points.append((event.xdata, event.ydata))
            self.redraw()
        else:
            idx = self.pick_handle(event)
            if idx is not None:
                self.drag_idx = idx

    def on_motion(self, event):
        if self.drag_idx is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x = np.clip(event.xdata, 0, self.frame.shape[1]-1)
        y = np.clip(event.ydata, 0, self.frame.shape[0]-1)
        self.points[self.drag_idx] = (x, y)
        self.redraw()

    def on_release(self, event):
        self.drag_idx = None

    def on_key(self, event):
        if event.key == "enter" and len(self.points) == 4:
            self.accepted = True
            plt.close(self.fig)

    def redraw(self):
        self.ax.clear()
        self.ax.imshow(self.frame, cmap="gray")
        labels = ["TL", "TR", "BL", "BR"]
        for (x, y) in self.points:
            self.ax.add_patch(Circle((x, y), radius=5, facecolor="none", edgecolor="yellow", lw=1.2))
        for i, (x, y) in enumerate(self.points):
            self.ax.text(x+4, y+4, labels[i] if i < 4 else str(i+1), color="y", fontsize=9)
        if len(self.points) == 4:
            (x0,y0),(x1,y1),(x2,y2),(x3,y3) = self.points
            self.ax.plot([x0,x1],[y0,y1],"y-",lw=1)
            self.ax.plot([x2,x3],[y2,y3],"y-",lw=1)
            self.ax.plot([x0,x2],[y0,y2],"y-",lw=1)
            self.ax.plot([x1,x3],[y1,y3],"y-",lw=1)
        self.ax.figure.canvas.draw_idle()

    def go(self) -> np.ndarray:
        plt.show()
        if not self.accepted or len(self.points) != 4:
            raise SystemExit("ROI selection cancelled or incomplete. Need 4 points, then press Enter.")
        return np.array(self.points, dtype=np.float32)


class GridAdjuster:
    """Drag corners; W/S rows, A/D cols, R/F box; G toggle boxes; Enter accept."""
    def __init__(self, frame: np.ndarray, quad: np.ndarray, rows: int, cols: int, box: int):
        self.frame = frame
        self.quad = quad.astype(float)  # TL, TR, BL, BR
        self.rows = int(rows)
        self.cols = int(cols)
        self.box  = int(box)
        self.show_boxes = True
        self.drag_idx: Optional[int] = None

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(frame, cmap="gray")
        self.ax.set_title("Grid Adjuster — Drag corners; W/S rows, A/D cols, R/F box; G show; ENTER accept")
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw()

    def centers_from_quad(self) -> np.ndarray:
        tl, tr, bl, br = self.quad[0], self.quad[1], self.quad[2], self.quad[3]
        centers = []
        for r in range(self.rows):
            v = (r + 0.5) / self.rows
            for c in range(self.cols):
                u = (c + 0.5) / self.cols
                top = (1 - u) * tl + u * tr
                bot = (1 - u) * bl + u * br
                p = (1 - v) * top + v * bot
                centers.append(p)
        return np.rint(np.array(centers)).astype(np.int32)

    def on_key(self, e):
        k = (e.key or "").lower()
        if k == "w": self.rows += 1
        elif k == "s": self.rows = max(1, self.rows - 1)
        elif k == "d": self.cols += 1
        elif k == "a": self.cols = max(1, self.cols - 1)
        elif k == "r": self.box += 1
        elif k == "f": self.box = max(1, self.box - 1)
        elif k == "g": self.show_boxes = not self.show_boxes
        elif k == "enter":
            plt.close(self.fig); return
        self.redraw(lite=True)

    def _within_handle(self, event, x, y, tol=8):
        if event.x is None or event.y is None:
            return False
        disp = self.ax.transData.transform((x, y))
        dx = event.x - disp[0]; dy = event.y - disp[1]
        return (dx*dx + dy*dy) ** 0.5 <= tol

    def pick_handle(self, event) -> Optional[int]:
        for i, (x, y) in enumerate(self.quad):
            if self._within_handle(event, x, y):
                return i
        return None

    def on_click(self, event):
        if event.inaxes != self.ax: return
        idx = self.pick_handle(event)
        if idx is not None: self.drag_idx = idx

    def on_motion(self, event):
        if self.drag_idx is None or event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        x = np.clip(event.xdata, 0, self.frame.shape[1]-1)
        y = np.clip(event.ydata, 0, self.frame.shape[0]-1)
        self.quad[self.drag_idx] = [x, y]
        self.redraw(lite=True)

    def on_release(self, event): self.drag_idx = None

    def redraw(self, lite=False):
        self.ax.clear(); self.ax.imshow(self.frame, cmap="gray")
        TL, TR, BL, BR = self.quad
        self.ax.plot([TL[0],TR[0]],[TL[1],TR[1]],"y-",lw=1)
        self.ax.plot([BL[0],BR[0]],[BL[1],BR[1]],"y-",lw=1)
        self.ax.plot([TL[0],BL[0]],[TL[1],BL[1]],"y-",lw=1)
        self.ax.plot([TR[0],BR[0]],[TR[1],BR[1]],"y-",lw=1)
        for (x,y) in self.quad:
            self.ax.add_patch(Circle((x,y), radius=5, facecolor="none", edgecolor="yellow", lw=1.2))
        if self.show_boxes:
            centers = self.centers_from_quad()
            h = self.box // 2
            for (cx, cy) in centers:
                rect = Rectangle((cx - h, cy - h), self.box, self.box, fill=False, linewidth=0.8, edgecolor="cyan")
                self.ax.add_patch(rect)
        self.ax.set_title(f"Grid Adjuster — rows={self.rows} cols={self.cols} box={self.box} (W/S, A/D, R/F). ENTER to accept.")
        self.ax.figure.canvas.draw_idle()
