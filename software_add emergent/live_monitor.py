import os
import sys
import time
import numpy as np
from collections import deque
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
from pathlib import Path
import threading



# ===== CONFIG =====
BIN_SIZE = 128
HISTORY = 5000  # how many frames to store per block (1 sec at 5k fps)
UPDATE_MS = 100   # check for new file every 2 ms

WATCH_FOLDER = Path("benchmark_batches3/raw")


FRAMES_PER_FILE = 10
FRAME_H, FRAME_W = 608, 1024

NUM_FILES = 500
SHAPE_PER_FILE = (FRAMES_PER_FILE, FRAME_H, FRAME_W)
N_BYTES_PER_FILE = np.prod(SHAPE_PER_FILE)
# ===================

class RawStreamViewer(QtWidgets.QMainWindow):
    def __init__(self, folder):

        super().__init__()
        self.setWindowTitle("Live Stream Viewer (1s Rolling Window)")

        self.folder = folder
        self.bin = BIN_SIZE
        self.history = HISTORY
        self.H_grid = FRAME_H // self.bin
        self.W_grid = FRAME_W // self.bin

        self.plots = [[None] * self.W_grid for _ in range(self.H_grid)]

        self.frame_history = deque(maxlen=HISTORY)
        self.frame_idx = 0
        self.finish = False

        self.setup_plot_grid()

        self.reader = threading.Thread(target=self.file_reader_thread, daemon=True)
        self.reader.start()

        self.binning_thread = threading.Thread(target=self.binning_worker, daemon=True)
        self.binning_thread.start()

        # self.timer = QtCore.QTimer()
        # self.timer.timeout.connect(self.check_new_files)
        # self.timer.start(UPDATE_MS)

    def setup_plot_grid(self):
        layout = pg.GraphicsLayoutWidget()
        self.setCentralWidget(layout)

        for i in range(self.H_grid):
            for j in range(self.W_grid):
                p = layout.addPlot(row=i, col=j)
                p.setMenuEnabled(False)
                p.setMouseEnabled(x=False, y=False)
                p.setXRange(0, self.history, padding=0)
                p.setYRange(-10, 10)
                p.hideAxis("bottom")
                p.hideAxis("left")
                self.plots[i][j] = p.plot(pen=pg.mkPen("lime", width=1))

    def file_reader_thread(self):

        t0 = time.perf_counter()
        self.latest_mtime = 0  # in __init__
        files = sorted(WATCH_FOLDER.glob("*.raw"), key=os.path.getmtime)
        buf = np.empty((FRAMES_PER_FILE, FRAME_H, FRAME_W), dtype=np.uint8)
       
        for f in files:
   
            with open(f, 'rb') as fp:
                fp.readinto(buf.view(np.uint8).reshape(-1))
              
                #binned = buf.reshape(FRAMES_PER_FILE, self.H_grid, self.bin, self.W_grid, self.bin).mean(axis=(2, 4))

                self.frame_history.extend(buf)
                #self.frame_history.extend(binned)

               
            self.frame_idx += FRAMES_PER_FILE

            # Print every 1000 frames
            if self.frame_idx % 1000 == 0:
                t_now = time.perf_counter()
                elapsed = t_now - t0
                fps = self.frame_idx / elapsed
                print(f"{self.frame_idx:6d} frames in {elapsed:6.2f} s  → {fps:6.0f} fps")
        self.finish = True

    def binning_worker(self):
        while not self.finish:
            time.sleep(0.05)  # Adjust binning rate (20Hz = 50ms)

            num_frames = len(self.frame_history)
            if num_frames == 0:
                continue

           
            # Step 1: Crop and stack
            H_b = self.H_grid * self.bin
            W_b = self.W_grid * self.bin
            try:
                all_frames = np.array(self.frame_history)[:, :H_b, :W_b]
            except Exception as e:
                print("Frame stack error:", e)
                continue

            if all_frames.shape[0] < 2:
                continue

            # Step 2: Vectorized binning
            try:
                t0 = time.time()

                self.binned = all_frames.reshape(
                    num_frames, self.H_grid, self.bin, self.W_grid, self.bin
                ).sum(axis=(2, 4))
            except Exception as e:
                print("Binning error:", e)
                continue

            print(f"[Binning] Frames: {num_frames}, Time: {time.time() - t0:.4f}s")        
            
    def check_new_files(self):
        t0 = time.time()

        # Step 1: Prepare the input
        num_frames = len(self.frame_history)
        print(num_frames)

        H_b = self.H_grid * self.bin
        W_b = self.W_grid * self.bin

        # Step 2: Stack and crop all frames into one array
        all_frames = np.array(self.frame_history)[:, :H_b, :W_b]  # shape (N, H, W)

        # Step 3: Vectorized binning
        # Reshape to (N, H_grid, bin, W_grid, bin), then average over bin dims
        binned = all_frames.reshape(num_frames, self.H_grid, self.bin, self.W_grid, self.bin).mean(axis=(2, 4))

        t_bin = time.time()

        # Step 4: Update all plots (each grid gets a time-series trace)
        for i in range(self.H_grid):
            for j in range(self.W_grid):
                trace = binned[:, i, j]
                self.plots[i][j].setData(trace)

                if len(trace) > 1:
                    y_min = np.min(trace)
                    y_max = np.max(trace)
                    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                        self.plots[i][j].getViewBox().setYRange(y_min, y_max, padding=0.2)
                    else:
                        center = y_min if np.isfinite(y_min) else 0
                        self.plots[i][j].getViewBox().setYRange(center - 1, center + 1)

        t_disp = time.time()

if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    win = RawStreamViewer(WATCH_FOLDER)
    win.show()
    sys.exit(app.exec())
