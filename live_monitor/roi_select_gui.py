#!/usr/bin/env python3
"""
roi_select_gui.py — Minimal ROI selector

- Pick an experiment folder (Tk dialog)
- Load the first BMP under it (supports bmp_export/, 0/, numeric subfolders)
- Click 4 corners (press F), drag yellow handles, adjust grid:
  W/S rows, D/A cols, G toggle boxes, ENTER save, Q save-only
- Saves roi_grid_config.json in the experiment folder

Deps: numpy, matplotlib, pillow, tkinter (stdlib)
"""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --------- folder picker ---------
def pick_folder_dialog(title: str = "Select experiment folder") -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.wm_attributes("-topmost", True)
        p = filedialog.askdirectory(title=title)
        root.destroy()
        if not p: raise RuntimeError("No folder chosen")
        return Path(p)
    except Exception:
        return Path.cwd()

# --------- robust BMP discovery ----------
BMP_PATTERNS = [
    re.compile(r"(?i)^batch[_-]?(\d+)[_-]frame[_-]?(\d+)\.bmp$"),
    re.compile(r"(?i)^(\d+)[_-](\d+)\.bmp$"),
    re.compile(r"(?i)^(\d+)\.bmp$"),  # numeric-only (combined index)
]

def _natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

def find_first_bmp(exp: Path) -> Path:
    bases = []
    if (exp / "bmp_export").exists(): bases.append(exp / "bmp_export")
    if (exp / "0").exists(): bases.append(exp / "0")
    nums = [d for d in exp.iterdir() if d.is_dir() and d.name.isdigit()]
    bases.extend(sorted(nums, key=lambda p: int(p.name)))
    bases.append(exp)

    cands = []
    for base in bases:
        for f in sorted(base.rglob("*.bmp"), key=lambda p: _natural_key(p.name)):
            for pat in BMP_PATTERNS:
                if pat.match(f.name):
                    cands.append(f); break
    if not cands:
        raise FileNotFoundError(f"No BMP files found under {exp}")
    return cands[0]

# --------- ROI GUI ----------
class ROIGridUI:
    def __init__(self, img: np.ndarray, out_json: Path, rows=8, cols=9,
                 cell_w_um=100.0, cell_h_um=80.0, row_gap_um=20.0, col_gap_um=0.0):
        self.img = img
        self.out_json = out_json
        self.rows, self.cols = int(rows), int(cols)
        self.cell_w_um, self.cell_h_um = float(cell_w_um), float(cell_h_um)
        self.row_gap_um, self.col_gap_um = float(row_gap_um), float(col_gap_um)
        self.corners: Optional[np.ndarray] = None
        self.handles: list[Circle] = []
        self.drag_idx: Optional[int] = None
        self.show_boxes = True
        self._artists = []
        self._temp_clicks: list[tuple[float,float]] = []
        self.proceed = False

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img, cmap="gray"); self.ax.set_axis_off()
        self.ax.set_title("F=4-pt | drag | W/S rows | D/A cols | G boxes | ENTER save | Q save-only")
        self.cid_k = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.cid_p = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.cid_m = self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        self.cid_r = self.fig.canvas.mpl_connect('button_release_event', self._on_release)

    # -- events --
    def _on_key(self, e):
        k = (e.key or "").lower()
        if k=='f':
            self._temp_clicks.clear()
            print("[ROI] 4-point mode: click four corners (any order).")
        elif k=='w':
            self.rows += 1; self._redraw()
        elif k=='s':
            self.rows = max(1, self.rows-1); self._redraw()
        elif k=='d':
            self.cols += 1; self._redraw()
        elif k=='a':
            self.cols = max(1, self.cols-1); self._redraw()
        elif k=='g':
            self.show_boxes = not self.show_boxes; self._redraw()
        elif k=='enter':
            if self.corners is None:
                print("[ROI] Set four corners before saving."); return
            self._save_roi(); self.proceed = True; plt.close(self.fig)
        elif k=='q':
            self._save_roi(); self.proceed = False; plt.close(self.fig)

    def _on_click(self, e):
        if e.inaxes!=self.ax or e.xdata is None: return
        if self.corners is not None and self.handles:
            for i,h in enumerate(self.handles):
                contains,_ = h.contains(e)
                if contains: self.drag_idx = i; return
        if len(self._temp_clicks) < 4:
            self._temp_clicks.append((e.xdata, e.ydata)); self._draw_temp()
            if len(self._temp_clicks)==4:
                self.corners = np.array(self._order_corners(self._temp_clicks), float)
                self._install_handles(); self._temp_clicks.clear(); self._redraw()

    def _on_move(self, e):
        if self.drag_idx is None or self.corners is None: return
        if e.inaxes!=self.ax or e.xdata is None: return
        self.corners[self.drag_idx] = [e.xdata, e.ydata]
        self.handles[self.drag_idx].center = (e.xdata, e.ydata)
        self._redraw(lite=True)

    def _on_release(self, e): self.drag_idx=None

    # -- internals --
    def _draw_temp(self):
        keep=[]; 
        for tag,art in self._artists:
            if tag=='temp':
                try: art.remove()
                except: pass
            else: keep.append((tag,art))
        self._artists=keep
        for (x,y) in self._temp_clicks:
            dot, = self.ax.plot(x,y,'o',ms=4,mec='cyan',mfc='none',ls='')
            self._artists.append(('temp', dot))
        self.fig.canvas.draw_idle()

    @staticmethod
    def _order_corners(points: List[Tuple[float,float]]):
        pts = np.array(points,float)
        cx,cy = pts[:,0].mean(), pts[:,1].mean()
        ang = np.arctan2(pts[:,1]-cy, pts[:,0]-cx)
        idx = np.argsort(ang); pts = pts[idx]
        tl_idx = np.lexsort((pts[:,0], pts[:,1]))[0]
        return np.roll(pts, -tl_idx, axis=0)  # TL,TR,BR,BL

    def _redraw(self, lite=False):
        keep=[]; 
        for tag,art in self._artists:
            if tag=='grid':
                try: art.remove()
                except: pass
            else: keep.append((tag,art))
        self._artists=keep
        if self.corners is None:
            self.fig.canvas.draw_idle(); return
        TL,TR,BR,BL = self.corners; A = TR-TL; B = BL-TL
        BRc = TL + A + B
        xs=[TL[0],TR[0],BRc[0],BL[0],TL[0]]; ys=[TL[1],TR[1],BRc[1],BL[1],TL[1]]
        l, = self.ax.plot(xs,ys,c='cyan',lw=1,alpha=0.9); self._artists.append(('grid', l))

        if self.show_boxes:
            total_w_um = self.cols*self.cell_w_um
            total_h_um = self.rows*self.cell_h_um + (self.rows-1)*self.row_gap_um
            half_w = self.cell_w_um/2; half_h = self.cell_h_um/2
            for r in range(self.rows):
                v_center_um = r*(self.cell_h_um+self.row_gap_um)+half_h
                v0=(v_center_um-half_h)/total_h_um; v1=(v_center_um+half_h)/total_h_um
                for c in range(self.cols):
                    u_center_um = c*(self.cell_w_um+self.col_gap_um)+half_w
                    u0=(u_center_um-half_w)/total_w_um; u1=(u_center_um+half_w)/total_w_um
                    def M(u,v): return TL + u*A + v*B
                    p1,p2,p3,p4 = M(u0,v0),M(u1,v0),M(u1,v1),M(u0,v1)
                    xs=[p1[0],p2[0],p3[0],p4[0],p1[0]]; ys=[p1[1],p2[1],p3[1],p4[1],p1[1]]
                    l, = self.ax.plot(xs,ys,c='cyan',lw=0.8,alpha=0.85)
                    self._artists.append(('grid', l))
        self._install_handles(); self.fig.canvas.draw_idle()

    def _install_handles(self):
        for h in self.handles:
            try: h.remove()
            except: pass
        self.handles.clear()
        if self.corners is None: return
        for (x,y) in self.corners:
            circ = Circle((x,y), radius=5, facecolor='none', edgecolor='yellow', lw=1.2)
            self.ax.add_patch(circ); self.handles.append(circ)

    def _save_roi(self):
        if self.corners is None: return
        TL,TR,BR,BL = self.corners; A=TR-TL; B=BL-TL
        total_w_um = self.cols*self.cell_w_um
        total_h_um = self.rows*self.cell_h_um + (self.rows-1)*self.row_gap_um
        half_w=self.cell_w_um/2; half_h=self.cell_h_um/2
        centers=[]; polys=[]
        for r in range(self.rows):
            v_center_um = r*(self.cell_h_um+self.row_gap_um)+half_h
            v0=(v_center_um-half_h)/total_h_um; v1=(v_center_um+half_h)/total_h_um
            for c in range(self.cols):
                u_center_um = c*(self.cell_w_um+self.col_gap_um)+half_w
                u0=(u_center_um-half_w)/total_w_um; u1=(u_center_um+half_w)/total_w_um
                def M(u,v): return TL + u*A + v*B
                p1,p2,p3,p4 = M(u0,v0),M(u1,v0),M(u1,v1),M(u0,v1)
                cx,cy = M((u0+u1)/2,(v0+v1)/2)
                centers.append([float(cx),float(cy)])
                polys.append([[float(p1[0]),float(p1[1])],
                              [float(p2[0]),float(p2[1])],
                              [float(p3[0]),float(p3[1])],
                              [float(p4[0][0] if hasattr(p4[0],'__len__') else p4[0]), float(p4[1][0] if hasattr(p4[1],'__len__') else p4[1])]])
        payload = dict(
            rows=int(self.rows), cols=int(self.cols),
            cell_w_um=float(self.cell_w_um), cell_h_um=float(self.cell_h_um),
            row_gap_um=float(self.row_gap_um), col_gap_um=float(self.col_gap_um),
            corners_xy=[TL.tolist(),TR.tolist(),BR.tolist(),BL.tolist()],
            centers_xy_f=centers, cell_polygons=polys
        )
        with open(self.out_json,'w',encoding='utf-8') as f:
            json.dump(payload,f,indent=2)
        print(f"[ROI] Saved → {self.out_json}")

def main():
    exp = pick_folder_dialog("Select experiment folder")
    first = Image.open(find_first_bmp(exp)).convert("L")
    img = np.array(first)
    out_json = exp / "roi_grid_config.json"
    ui = ROIGridUI(img, out_json)
    print("[ROI UI] F=4-pt; drag; W/S rows; D/A cols; G toggle boxes; ENTER save; Q save-only")
    plt.show()

if __name__ == "__main__":
    main()
