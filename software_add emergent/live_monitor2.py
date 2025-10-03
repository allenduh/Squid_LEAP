# # live_monitor_fast.py
# from pathlib import Path
# from collections import deque
# import threading, time, queue, os

# import numpy as np
# import pyqtgraph as pg
# from PyQt5 import QtCore, QtWidgets


# # --------------------------- basic settings --------------------------------
# FRAME_W, FRAME_H   = 1024, 608        # camera resolution
# FRAMES_PER_FILE    = 10               # set 100 if camera allows bigger files
# FPS                = 5000
# HISTORY            = FPS              # 1‑s rolling window (5 000 samples)
# PRINT_EVERY        = 0.25             # seconds between console stats
# POLL_SLEEP         = 0.0005           # 0.5 ms poll interval
# # ---------------------------------------------------------------------------


# # ----------------------- locate newest EVT_* folder ------------------------
# BASE_DIR  = Path("output")
# RUN_DIRS  = sorted(BASE_DIR.glob("EVT*"))
# if not RUN_DIRS:
#     raise FileNotFoundError("No EVT_* folder in ./output")
# RUN_DIR   = RUN_DIRS[-1]
# print("Watching :", RUN_DIR)

# NEEDED_BYTES = FRAME_W * FRAME_H * FRAMES_PER_FILE
# # ---------------------------------------------------------------------------


# # ---------------------- global counters for stats --------------------------
# start_time   = time.perf_counter()
# frames_read  = 0
# frames_drawn = 0
# stat_lock    = threading.Lock()
# # ---------------------------------------------------------------------------


# def fast_reader(q: "queue.Queue[np.ndarray]"):
#     """
#     Faster producer: batch_<n>.raw --> 10 mean values -> queue (1 put per file)
#     """
#     global frames_read
#     idx = 0
#     while True:
#         f = RUN_DIR / f"batch_{idx}.raw"

#         # wait until file exists and is complete
#         if not f.exists() or f.stat().st_size < NEEDED_BYTES:
#             time.sleep(POLL_SLEEP)
#             continue

#         # zero‑copy map → reshape in place
#         mm = np.memmap(f, dtype=np.uint8, mode='r')
#         frames = np.ndarray((FRAMES_PER_FILE, FRAME_H, FRAME_W),
#                             dtype=np.uint8, buffer=mm)
#         vals = frames.mean(axis=(1, 2), dtype=np.float32)  # (10,)

#         q.put_nowait(vals)                # ONE put per file

#         with stat_lock:
#             frames_read += FRAMES_PER_FILE
#         idx += 1


# class LivePlot(QtWidgets.QMainWindow):
#     def __init__(self, q: "queue.Queue[np.ndarray]"):
#         super().__init__()
#         self.q         = q
#         self.buffer    = deque(maxlen=HISTORY)

#         self.plot_w    = pg.PlotWidget()
#         self.plot_w.setXRange(-1, 0)     # 1‑s window
#         self.curve     = self.plot_w.plot(pen="lime", width=1)
#         self.setCentralWidget(self.plot_w)
#         self.resize(800, 400)
#         self.setWindowTitle("Live 5 kfps monitor")
#         self.show()

#         # single‑shot timer → re‑arm inside update()
#         self.timer = QtCore.QTimer(singleShot=True)
#         self.timer.timeout.connect(self.update_curve)
#         self.timer.start(0)               # fire ASAP

#         self._last_print = start_time

#     # ----------------------------------------------------------------------
#     def update_curve(self):
#         global frames_drawn

#         # move arrays from queue → deque (extend)
#         while not self.q.empty():
#             vals = self.q.get_nowait()    # vals.shape = (10,)
#             self.buffer.extend(vals)

#         frames_drawn = len(self.buffer)

#         if self.buffer:
#             y = np.fromiter(self.buffer, dtype=np.float32, count=len(self.buffer))
#             x = np.linspace(-1, 0, len(y), endpoint=False)
#             self.curve.setData(x, y, clear=True)

#         # throttled console stats
#         now = time.perf_counter()
#         if now - self._last_print >= PRINT_EVERY:
#             with stat_lock:
#                 r = frames_read
#             print(f"t = {now - start_time:6.2f}s | "
#                   f"read {r:6d} | shown {frames_drawn:6d}", flush=True)
#             self._last_print = now

#         # re‑arm timer immediately
#         self.timer.start(0)


# # ----------------------------- main ----------------------------------------
# if __name__ == "__main__":
#     data_q = queue.Queue(maxsize=HISTORY)

#     threading.Thread(target=fast_reader,
#                      args=(data_q,),
#                      daemon=True).start()

#     pg.setConfigOptions(antialias=True)
#     app = QtWidgets.QApplication([])
#     win = LivePlot(data_q)
#     app.exec_()


# benchmark_raw_read.py
import time
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    FRAME_W, FRAME_H     = 1024, 608
    FRAMES_PER_FILE      = 10
    EXPECTED_FRAMES      = 50000
    NEEDED_BYTES         = FRAME_W * FRAME_H * FRAMES_PER_FILE

    RUN_DIR = Path("output/EVT_Py_convert")
    print(f"Benchmarking memmap read speed from: {RUN_DIR.resolve()}")

    # Pre-list all valid files once
    files = sorted(f for f in RUN_DIR.glob("batch_*.raw")
                   if f.stat().st_size >= NEEDED_BYTES)

    if not files:
        raise FileNotFoundError("No batch_*.raw files found with correct size.")

    t0 = time.perf_counter()
    frames_read = 0
    sum = 0

    for i, f in enumerate(files):
        # Use np.memmap for zero-copy fast access

        mm = np.memmap(f, dtype=np.uint8, mode='r', shape=(FRAMES_PER_FILE, FRAME_H, FRAME_W))
        arr = np.array(mm[3])  # Force load into RAM
        sum = sum + arr
        #_ = arr.mean(axis=(1, 2), dtype=np.uint8)


        frames_read += FRAMES_PER_FILE

        if frames_read >= EXPECTED_FRAMES:
            break
    print(sum)
    t_final = time.perf_counter()
    total_elapsed = t_final - t0
    print(f"\n✅ Read {frames_read} frames in {total_elapsed:.2f} seconds "
          f"→ {frames_read / total_elapsed:.0f} fps")


    FRAME_H = 608
    FRAME_W = 1024
    FRAMES = 100  # adjust as needed
    NEEDED_BYTES = FRAME_H * FRAME_W * FRAMES

    DATA_SHAPE = (FRAMES, FRAME_H, FRAME_W)
    RAW_FILE = Path("test_batch.raw")
    NPY_FILE = Path("test_batch.npy")

    # Step 1: Generate and save test data
    print(f"Generating {FRAMES} frames of shape {FRAME_H}×{FRAME_W}...")
    data = np.random.randint(0, 256, size=DATA_SHAPE, dtype=np.uint8)

    print("Saving test_batch.raw...")
    t0 = time.perf_counter()
    data.tofile(RAW_FILE)
    t1 = time.perf_counter()
    print(f"RAW write time: {t1 - t0:.4f} sec")

    print("Saving test_batch.npy...")
    t0 = time.perf_counter()
    np.save(NPY_FILE, data)
    t1 = time.perf_counter()
    print(f"NPY write time: {t1 - t0:.4f} sec")

    # Step 2: Benchmark reading .raw
    print("\nReading RAW...")
    t0 = time.perf_counter()
    raw_bytes = np.fromfile(RAW_FILE, dtype=np.uint8, count=NEEDED_BYTES)
    raw_data = raw_bytes.reshape(DATA_SHAPE)
    _ = raw_data.mean(axis=(1, 2))  # simulate binning
    t1 = time.perf_counter()
    print(f"RAW read + reshape + mean: {t1 - t0:.4f} sec")

    # Step 3: Benchmark reading .npy
    print("\nReading NPY...")
    t0 = time.perf_counter()
    npy_data = np.load(NPY_FILE)
    _ = npy_data.mean(axis=(1, 2))  # simulate binning
    t1 = time.perf_counter()
    print(f"NPY load + mean: {t1 - t0:.4f} sec")



    # -------- parameters ----------
    FRAME_H, FRAME_W = 608, 1024
    FRAMES           = 5000          # 1‑second worth of data
    RAW_FILE         = Path("test_5k.raw")
    NPY_FILE         = Path("test_5k.npy")
    SHAPE            = (FRAMES, FRAME_H, FRAME_W)
    N_BYTES          = np.prod(SHAPE)               # total bytes
    # --------------------------------

    def write_test_files():
        if RAW_FILE.exists() and NPY_FILE.exists():
            return
        print(f"Creating test files ~{N_BYTES/1e9:.2f} GB …")
        data = np.random.randint(0, 256, size=SHAPE, dtype=np.uint8)

        t = time.perf_counter()
        data.tofile(RAW_FILE)
        print(f"raw  write: {time.perf_counter()-t:.2f} s")

        t = time.perf_counter()
        np.save(NPY_FILE, data)
        print(f"npy  write: {time.perf_counter()-t:.2f} s\n")
        del data

    def bench(label, func):
        t0 = time.perf_counter()
        func()
        dt = time.perf_counter() - t0
        gbps = (N_BYTES/1e9)/dt
        print(f"{label:<20} {dt:6.2f} s  | {gbps:5.2f} GB/s")

    # ---------- read methods --------------
    def read_fromfile():
        arr = np.fromfile(NPY_FILE, dtype=np.uint8, count=N_BYTES)
        arr.reshape(SHAPE)

    def read_readinto():
        buf = np.empty(N_BYTES, dtype=np.uint8)
        with open(RAW_FILE, "rb") as f:
            f.readinto(buf)
        buf.reshape(SHAPE)

    def read_memmap():
        arr = np.memmap(NPY_FILE, dtype=np.uint8, mode="r", shape=SHAPE)
        _ = arr[0, 0, 0]  # touch once to force map
        del arr

    def read_npy():
        np.load(NPY_FILE)

    # -------------- main ------------------
    if __name__ == "__main__":
        write_test_files()
        print("Read speed benchmark (5 000 frames / 3 GB):")
        bench("np.fromfile",  read_fromfile)
        bench("readinto",     read_readinto)
        bench("np.memmap",    read_memmap)
        bench("np.load(.npy)",read_npy)
