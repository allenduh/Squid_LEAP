#!/usr/bin/env python3
# convert_raw_to_bmp_folder.py

import os
import re
import sys
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog
import imageio.v2 as imageio

# -------------------- CONFIG --------------------
WIDTH  = 1024
HEIGHT = 608
FRAMES_PER_FILE = 10
DTYPE = np.uint8  # change to np.uint16 if your camera writes 16-bit
RAW_PATTERN = re.compile(r'(?i)^batch[_-]?(\d+)\.raw$')  # captures batch index
BMP_FOLDER_NAME = "bmp_export"
# ------------------------------------------------

def natural_key(name: str):
    # batch_2 before batch_10
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]

def choose_folder() -> Path:
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select folder containing batch_*.raw")
    root.update()
    root.destroy()
    if not folder:
        print("No folder selected. Exiting.")
        sys.exit(0)
    return Path(folder)

def find_raws(folder: Path):
    # Only top-level files in the chosen folder; change to rglob if yours are nested
    raws = []
    for p in folder.iterdir():
        if p.is_file():
            m = RAW_PATTERN.match(p.name)
            if m:
                batch = int(m.group(1))
                raws.append((batch, p))
    raws.sort(key=lambda t: t[0])  # sort by batch number
    return raws

def convert_one_raw(raw_path: Path, batch_idx: int, out_dir: Path):
    bytes_per_pixel = 1 if DTYPE == np.uint8 else 2
    expected_count = WIDTH * HEIGHT * FRAMES_PER_FILE
    expected_bytes = expected_count * bytes_per_pixel

    with open(raw_path, "rb") as f:
        buf = np.fromfile(f, dtype=DTYPE, count=expected_count)

    if buf.size * bytes_per_pixel != expected_bytes:
        print(f"[WARN] {raw_path.name}: size mismatch "
              f"(got {buf.size*bytes_per_pixel} bytes, expected {expected_bytes}). Skipping.")
        return 0

    # reshape and save
    frames = buf.reshape(FRAMES_PER_FILE, HEIGHT, WIDTH)

    saved = 0
    for i in range(FRAMES_PER_FILE):
        # file name must match your desired BMP_RE = r"^(\d+)_(\d+)\.bmp$"
        # so it's "<batch>_<frame>.bmp" (digits only). We'll zero pad frame to 3 for readability,
        # but it's still digits-only and matches your regex.
        bmp_name = f"{batch_idx}_{i:03d}.bmp"
        imageio.imwrite(out_dir / bmp_name, frames[i].astype(np.uint8))
        saved += 1
    return saved

def main():
    folder = choose_folder()
    raws = find_raws(folder)
    if not raws:
        print(f"No RAW files named like 'batch_###.raw' in: {folder}")
        sys.exit(1)

    out_dir = folder / BMP_FOLDER_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(raws)} RAW files from:\n  {folder}\ninto BMPs under:\n  {out_dir}\n")
    total_saved = 0
    for batch_idx, raw_path in raws:
        saved = convert_one_raw(raw_path, batch_idx, out_dir)
        print(f"  {raw_path.name} -> {saved} BMP(s)")
        total_saved += saved

    print(f"\nDone. Wrote {total_saved} BMP files to: {out_dir}")
    print("Example name pattern:", f"{raws[0][0]}_000.bmp  (matches  ^(\\d+)_(\\d+)\\.bmp$ )")

if __name__ == "__main__":
    main()
