import ctypes, os, glob, time
from typing import Optional

# ---- Load DLL ----
DLL_PATH = os.path.abspath("cpp/raw_bin.dll")
lib = ctypes.CDLL(DLL_PATH)

lib.tail_bin_to_tiff.argtypes = [
    ctypes.c_char_p,  # raw_path (bytes)
    ctypes.c_char_p,  # out_path (bytes)
    ctypes.c_int,     # width
    ctypes.c_int,     # height
    ctypes.c_int,     # bin
    ctypes.c_uint64,  # max_frames (0 = unlimited)
    ctypes.c_int,     # tail_timeout_ms
]
lib.tail_bin_to_tiff.restype = ctypes.c_int

def newest_raw_in(folder: str) -> Optional[str]:
    files = sorted(glob.glob(os.path.join(folder, "*.raw")), key=os.path.getmtime)
    return files[-1] if files else None

def run_one(raw_path: str,
            out_path: str,
            width: int = 1024,
            height: int = 608,
            bin_factor: int = 16,
            tail_timeout_ms: int = 3000,
            max_frames: int = 0) -> int:
    """Tail a growing RAW, bin, and stream to TIFF via DLL. Returns frames written."""
    raw_b = os.fsencode(raw_path)
    out_b = os.fsencode(out_path)
    t0 = time.perf_counter()
    frames = lib.tail_bin_to_tiff(raw_b, out_b, width, height, bin_factor,
                                  ctypes.c_uint64(max_frames), tail_timeout_ms)
    dt = time.perf_counter() - t0
    if frames < 0:
        raise RuntimeError(f"DLL error {frames} (raw='{raw_path}', out='{out_path}')")
    print(f"[tail_bin] wrote {frames} pages to '{out_path}' in {dt:.2f}s")
    return frames

def run_from_folders(input_folder: str,
                     output_folder: str,
                     width: int = 1024,
                     height: int = 608,
                     bin_factor: int = 16,
                     tail_timeout_ms: int = 3000,
                     max_frames: int = 0,
                     raw_name: Optional[str] = None) -> str:
    """
    Choose a RAW from input_folder, write TIFF to output_folder with sensible name.
    - If raw_name is None: pick the newest *.raw.
    - Returns the output TIFF path.
    """
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    if raw_name:
        raw_path = os.path.join(input_folder, raw_name)
    else:
        raw_path = newest_raw_in(input_folder)
        if not raw_path:
            raise FileNotFoundError(f"No .raw found in '{input_folder}'")

    base = os.path.splitext(os.path.basename(raw_path))[0]
    out_path = os.path.join(output_folder, f"{base}_binned_{bin_factor}x{bin_factor}.tif")

    run_one(raw_path, out_path, width, height, bin_factor, tail_timeout_ms, max_frames)
    return out_path

if __name__ == "__main__":
    # EXAMPLE: set these once; rerun as needed (no typing long CLI)
    INPUT_FOLDER  = r"C:\data"
    OUTPUT_FOLDER = r"C:\data\binned"
    WIDTH, HEIGHT = 1024, 608
    BIN          = 16
    TAIL_MS      = 3000
    MAX_FRAMES   = 0  # 0 = unlimited (stop when stream idles for TAIL_MS)

    out_tiff = run_from_folders(INPUT_FOLDER, OUTPUT_FOLDER,
                                width=WIDTH, height=HEIGHT,
                                bin_factor=BIN,
                                tail_timeout_ms=TAIL_MS,
                                max_frames=MAX_FRAMES,
                                raw_name=None)  # or "frames.raw"
    print("Output:", out_tiff)
