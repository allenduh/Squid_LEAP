import os
import numpy as np
import time
from pathlib import Path

# ------------- Config -------------
FRAME_H, FRAME_W = 608, 1024
FRAMES_PER_FILE = 10
NUM_FILES = 500
SHAPE_PER_FILE = (FRAMES_PER_FILE, FRAME_H, FRAME_W)
N_BYTES_PER_FILE = np.prod(SHAPE_PER_FILE)
DATA_DIR = Path("benchmark_batches")
RAW_DIR = DATA_DIR / "raw"
NPY_DIR = DATA_DIR / "npy"

RAW_DIR.mkdir(parents=True, exist_ok=True)
NPY_DIR.mkdir(parents=True, exist_ok=True)

# ------------- Write Test Files -------------
def write_batch_files():
    if list(RAW_DIR.glob("*.raw")) and list(NPY_DIR.glob("*.npy")):
        print("Test files already exist. Skipping write.")
        return

    print(f"Creating {NUM_FILES} test files × {FRAMES_PER_FILE} frames …")
    for i in range(NUM_FILES):
        data = np.random.randint(0, 256, size=SHAPE_PER_FILE, dtype=np.uint8)
        data.tofile(RAW_DIR / f"batch_{i:04d}.raw")
        np.save(NPY_DIR / f"batch_{i:04d}.npy", data)
    print("✅ Done writing test files\n")

# ------------- Benchmark Utility -------------
def bench(label, read_func):
    t0 = time.perf_counter()
    read_func()
    dt = time.perf_counter() - t0
    total_bytes = N_BYTES_PER_FILE * NUM_FILES
    print(f"{label:<18} {dt:.2f} s  | {(total_bytes/1e9)/dt:.2f} GB/s")

# ---------- Read Methods ----------

def read_fromfile():
    for f in sorted(RAW_DIR.glob("*.raw")):
        data = np.fromfile(f, dtype=np.uint8, count=N_BYTES_PER_FILE)
        _ = data.reshape(SHAPE_PER_FILE)

def read_readinto():
    for f in sorted(RAW_DIR.glob("*.raw")):
        buf = np.empty(N_BYTES_PER_FILE, dtype=np.uint8)
        with open(f, "rb") as fp:
            fp.readinto(buf)
        _ = buf.reshape(SHAPE_PER_FILE)

def read_memmap():
    for f in sorted(RAW_DIR.glob("*.raw")):
        mm = np.memmap(f, dtype=np.uint8, mode="r", shape=SHAPE_PER_FILE)
        _ = mm[0, 0, 0]  # touch
        del mm

def read_npy():
    for f in sorted(NPY_DIR.glob("*.npy")):
        _ = np.load(f)

# ------------- Run All -------------
if __name__ == "__main__":
    write_batch_files()
    bench("fromfile", read_fromfile)
    bench("readinto", read_readinto)
    bench("memmap", read_memmap)
    bench("npy load", read_npy)
