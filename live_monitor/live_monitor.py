import os
import glob
import time
import numpy as np
from collections import deque
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg

BIN_SIZE = 128
HISTORY = 20
UPDATE_MS = 200  # ms
WATCH_FOLDER = "output/stream"
FRAME_HEIGHT = 1836
FRAME_WIDTH = 2064
FRAMES_PER_FILE = 10

class RawStreamVisualizer(QtWidgets.QMainWindow):
    def __init__(self, folder, bin_size=128):
        super().__init__()
        self.setWindowTitle("Live Raw Stream Monitor")

        self.folder = folder
        self.bin = bin_size
        self.history = HISTORY
        self.processed_files = set()
        self.initial_frame = None
        self.buffers = None
        self.curves = None

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_stream)
        self.timer.start(UPDATE_MS)

    def setup_plot_grid(self, H_grid, W_grid):
        self.H_grid = H_grid
        self.W_grid = W_grid
        layout = pg.GraphicsLayoutWidget()
        self.setCentralWidget(layout)

        self.buffers = [[deque(maxlen=self.history) for _ in range(W_grid)] for _ in range(H_grid)]
        self.curves = [[None]*W_grid for _ in range(H_grid)]

        for i in range(H_grid):
            for j in range(W_grid):
                p = layout.addPlot(row=i, col=j)
                p.setMenuEnabled(False)
                p.setMouseEnabled(x=False, y=False)
                p.setXRange(0, self.history-1, padding=0)
                p.setYRange(-20, 20)
                p.hideAxis("bottom")
                p.hideAxis("left")
                self.curves[i][j] = p.plot(pen=pg.mkPen("lime", width=1))

    def update_stream(self):
        raw_files = sorted(glob.glob(os.path.join(self.folder, "**", "*.raw"), recursive=True))
        new_files = [f for f in raw_files if f not in self.processed_files]
        if not new_files:
            return

        for path in new_files:
            try:
                with open(path, "rb") as f:
                    data = np.frombuffer(f.read(), dtype=np.uint8)

                expected_size = FRAMES_PER_FILE * FRAME_HEIGHT * FRAME_WIDTH
                if data.size < expected_size:
                    print(f"Skipping {path}: incomplete data")
                    continue

                frames = data.reshape(FRAMES_PER_FILE, FRAME_HEIGHT, FRAME_WIDTH)

                # Trim to multiple of bin size
                H_b = (FRAME_HEIGHT // self.bin) * self.bin
                W_b = (FRAME_WIDTH // self.bin) * self.bin
                frames = frames[:, :H_b, :W_b]

                # Initialize baseline and plotting
                if self.initial_frame is None:
                    self.initial_frame = frames[0]  # anchor first frame
                    self.H_grid = H_b // self.bin
                    self.W_grid = W_b // self.bin
                    self.setup_plot_grid(self.H_grid, self.W_grid)

                # Subtract baseline
                frames_delta = frames - self.initial_frame

                # Bin and update plots frame by frame
                for f in range(FRAMES_PER_FILE):
                    frame = frames_delta[f]
                    binned = frame.reshape(
                        self.H_grid, self.bin,
                        self.W_grid, self.bin
                    ).mean(axis=(1, 3))

                    for i in range(self.H_grid):
                        for j in range(self.W_grid):
                            buf = self.buffers[i][j]
                            buf.append(binned[i, j])
                            y = list(buf)
                            self.curves[i][j].setData(range(len(y)), y)
                            self.curves[i][j].getViewBox().setYRange(min(y), max(y), padding=0.1)

                self.processed_files.add(path)

            except Exception as e:
                print(f"Error processing {path}: {e}")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = RawStreamVisualizer(WATCH_FOLDER, bin_size=BIN_SIZE)
    win.show()
    sys.exit(app.exec())
