#!/usr/bin/env python3
# GUI: parallel 16x16 binning & unbinned slice (NumPy) + DLL tail->bin->TIFF button.

import time
import re
import os
import glob
import ctypes
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tifffile as tiff

# --------------------------------
# Constants (same as your original)
# --------------------------------
H, W = 608, 1024
BIN_DEFAULT = 16
N_FRAMES_PER_RAW = 10
DTYPE_RAW = np.uint8

# --------------------------------
# Helpers (same as your original)
# --------------------------------
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def read_and_bin_one_raw(raw_path: str) -> np.ndarray:
    rp = Path(raw_path)
    buf = np.fromfile(rp, dtype=DTYPE_RAW, count=N_FRAMES_PER_RAW * H * W)
    frames = buf.reshape(N_FRAMES_PER_RAW, H, W)
    hh, ww = H // BIN_DEFAULT, W // BIN_DEFAULT
    frames = frames[:, :hh*BIN_DEFAULT, :ww*BIN_DEFAULT]
    sums = frames.reshape(N_FRAMES_PER_RAW, hh, BIN_DEFAULT, ww, BIN_DEFAULT).sum(axis=(2,4), dtype=np.uint32)
    return sums.astype(np.uint16, copy=False)

def write_tiff_imagej_uint16(path: Path, stack_uint16: np.ndarray):
    tiff.imwrite(path, stack_uint16, dtype=np.uint16, imagej=True)

def write_tiff_imagej_uint8(path: Path, stack_uint8: np.ndarray):
    tiff.imwrite(path, stack_uint8, dtype=np.uint8, imagej=True)

def extract_unbinned_slice(raws, start_frame, end_frame_excl):
    if end_frame_excl <= start_frame:
        raise ValueError("End frame must be greater than start frame.")
    total_frames = len(raws) * N_FRAMES_PER_RAW
    if start_frame < 0 or end_frame_excl > total_frames:
        raise ValueError(f"Requested range [{start_frame}, {end_frame_excl}) outside [0, {total_frames})")
    T = end_frame_excl - start_frame
    out = np.empty((T, H, W), dtype=np.uint8)
    file_start = start_frame // N_FRAMES_PER_RAW
    off_start  = start_frame %  N_FRAMES_PER_RAW
    file_last  = (end_frame_excl - 1) // N_FRAMES_PER_RAW
    off_last   = (end_frame_excl - 1) %  N_FRAMES_PER_RAW
    dst = 0
    for fi in range(file_start, file_last + 1):
        rp = raws[fi]
        buf = np.fromfile(rp, dtype=DTYPE_RAW, count=N_FRAMES_PER_RAW * H * W)
        frames = buf.reshape(N_FRAMES_PER_RAW, H, W)
        s = off_start if fi == file_start else 0
        e = (off_last + 1) if fi == file_last else N_FRAMES_PER_RAW
        n = e - s
        out[dst:dst+n] = frames[s:e]
        dst += n
    return out

# ================================
#            GUI
# ================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAW tools: 16×16 bin (parallel) / Unbinned slice / DLL tail→TIFF")
        self.geometry("900x560")
        self.in_dir = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.workers = tk.IntVar(value=8)
        self.slice_start = tk.IntVar(value=24000)
        self.slice_end   = tk.IntVar(value=24200)

        # --- DLL additions ---
        self.bin_factor       = tk.IntVar(value=BIN_DEFAULT)
        self.tail_timeout_ms  = tk.IntVar(value=3000)
        self.max_frames       = tk.IntVar(value=0)
        self.selected_raw     = tk.StringVar(value="")  # optional explicit .raw
        self.dll_loaded       = False
        self._load_dll_once()
        # ---

        self._build()

    # --- DLL additions ---
    def _load_dll_once(self):
        """Load raw_tail_bin_tiff.dll (if present) and configure ctypes prototype."""
        dll_name = "raw_tail_bin_tiff.dll"
        candidates = [
            Path(os.getcwd()) / dll_name,
            Path(__file__).with_name(dll_name),
        ]
        dll_path = None
        for p in candidates:
            if p.exists():
                dll_path = str(p)
                break
        if not dll_path:
            self.dll_loaded = False
            return

        try:
            self._dll = ctypes.CDLL(dll_path)
            # int tail_bin_to_tiff(const char* raw_path, const char* out_path,
            #   int W, int H, int bin, unsigned long long max_frames, int tail_timeout_ms)
            self._dll.tail_bin_to_tiff.argtypes = [
                ctypes.c_char_p, ctypes.c_char_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_uint64, ctypes.c_int
            ]
            self._dll.tail_bin_to_tiff.restype = ctypes.c_int
            self.dll_loaded = True
        except Exception as e:
            messagebox.showwarning("DLL load failed", f"Could not load DLL:\n{e}")
            self.dll_loaded = False

    def _tail_bin_to_tiff(self, raw_path: Path, out_path: Path, B: int, tail_ms: int, max_frames: int) -> int:
        """Call into the DLL tailer."""
        if not self.dll_loaded:
            raise RuntimeError("DLL not loaded. Put raw_tail_bin_tiff.dll next to this script.")
        return self._dll.tail_bin_to_tiff(
            os.fsencode(str(raw_path)),
            os.fsencode(str(out_path)),
            ctypes.c_int(W), ctypes.c_int(H), ctypes.c_int(B),
            ctypes.c_uint64(max_frames),
            ctypes.c_int(tail_ms)
        )
    # ---

    def _build(self):
        pad = {"padx":10, "pady":6}
        root = ttk.Frame(self); root.pack(fill="both", expand=True)

        ttk.Label(root, text="RAW input").grid(row=0, column=0, sticky="e")
        ttk.Entry(root, textvariable=self.in_dir, width=64).grid(row=0, column=1, sticky="we")
        ttk.Button(root, text="Browse…", command=self._pick_in).grid(row=0, column=2, **pad)

        ttk.Label(root, text="Output").grid(row=1, column=0, sticky="e")
        ttk.Entry(root, textvariable=self.out_dir, width=64).grid(row=1, column=1, sticky="we")
        ttk.Button(root, text="Browse…", command=self._pick_out).grid(row=1, column=2, **pad)

        s = ttk.LabelFrame(root, text="Settings")
        s.grid(row=2, column=0, columnspan=3, sticky="we", **pad)
        ttk.Label(s, text=f"Frame: {H}×{W}   Frames/RAW: {N_FRAMES_PER_RAW}").grid(row=0, column=0, sticky="w")
        ttk.Label(s, text="Workers").grid(row=0, column=1, sticky="e", padx=(16,4))
        ttk.Entry(s, textvariable=self.workers, width=6).grid(row=0, column=2, sticky="w")

        # --- DLL additions: tail settings ---
        d = ttk.LabelFrame(root, text="DLL tail→bin→TIFF")
        d.grid(row=3, column=0, columnspan=3, sticky="we", **pad)
        ttk.Label(d, text="Bin").grid(row=0, column=0, sticky="e")
        ttk.Entry(d, textvariable=self.bin_factor, width=6).grid(row=0, column=1, sticky="w")

        ttk.Label(d, text="Tail timeout (ms)").grid(row=0, column=2, sticky="e")
        ttk.Entry(d, textvariable=self.tail_timeout_ms, width=8).grid(row=0, column=3, sticky="w")

        ttk.Label(d, text="Max frames (0=∞)").grid(row=0, column=4, sticky="e")
        ttk.Entry(d, textvariable=self.max_frames, width=8).grid(row=0, column=5, sticky="w")

        ttk.Label(d, text="(Optional) pick a single .raw").grid(row=1, column=0, sticky="e")
        ttk.Entry(d, textvariable=self.selected_raw, width=64).grid(row=1, column=1, columnspan=4, sticky="we")
        ttk.Button(d, text="Pick .raw…", command=self._pick_one_raw).grid(row=1, column=5, sticky="w", padx=6)
        # ---

        slice_box = ttk.LabelFrame(root, text="Unbinned slice (uint8)")
        slice_box.grid(row=4, column=0, columnspan=3, sticky="we", **pad)
        self.slice_start = tk.IntVar(value=24000)
        self.slice_end   = tk.IntVar(value=24200)
        ttk.Label(slice_box, text="Start frame (inclusive)").grid(row=0, column=0, sticky="e")
        ttk.Entry(slice_box, textvariable=self.slice_start, width=12).grid(row=0, column=1, sticky="w")
        ttk.Label(slice_box, text="End frame (exclusive)").grid(row=0, column=2, sticky="e")
        ttk.Entry(slice_box, textvariable=self.slice_end, width=12).grid(row=0, column=3, sticky="w")

        p = ttk.LabelFrame(root, text="Progress")
        p.grid(row=5, column=0, columnspan=3, sticky="we", **pad)
        self.pb = ttk.Progressbar(p, mode="determinate")
        self.pb.grid(row=0, column=0, sticky="we", padx=8, pady=8)
        self.plabel = ttk.Label(p, text="Idle")
        self.plabel.grid(row=0, column=1, sticky="w", padx=8)
        p.columnconfigure(0, weight=1)

        btns = ttk.Frame(root)
        btns.grid(row=6, column=0, columnspan=3, sticky="we", **pad)
        ttk.Button(btns, text="Bin 16×16 (parallel)", command=self.run_binned).pack(side="left")
        ttk.Button(btns, text="Export unbinned slice", command=self.run_slice).pack(side="left", padx=12)

        # --- DLL additions: new button ---
        ttk.Button(btns, text="Tail .raw → TIFF (DLL)", command=self.run_tail_dll).pack(side="left", padx=12)
        # ---

        ttk.Button(btns, text="Quit", command=self.destroy).pack(side="right")

        self.log = tk.Text(root, height=12, wrap="none")
        self.log.grid(row=7, column=0, columnspan=3, sticky="nsew", **pad)

        root.columnconfigure(1, weight=1)
        root.rowconfigure(7, weight=1)

    def _pick_in(self):
        d = filedialog.askdirectory(title="Select RAW input folder")
        if d: self.in_dir.set(d)

    def _pick_out(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d: self.out_dir.set(d)

    # --- DLL additions ---
    def _pick_one_raw(self):
        f = filedialog.askopenfilename(
            title="Pick a .raw file",
            filetypes=[("RAW files","*.raw"),("All files","*.*")]
        )
        if f:
            self.selected_raw.set(f)

    def _newest_raw(self, folder: Path):
        files = sorted(folder.glob("*.raw"), key=lambda p: p.stat().st_mtime)
        return files[-1] if files else None
    # ---

    def _ui(self, fn, *a, **k): self.after(0, lambda: fn(*a, **k))
    def _log(self, s): self.log.insert("end", s + "\n"); self.log.see("end")
    def _progress(self, k, n): self.pb["maximum"]=max(1,n); self.pb["value"]=k; self.plabel.configure(text=f"{k}/{n}"); self.update_idletasks()

    def _common_setup(self):
        in_dir = Path(self.in_dir.get())
        if not in_dir.exists():
            messagebox.showerror("Error", "Pick a valid input folder")
            return None, None, None
        out_dir = Path(self.out_dir.get()) if self.out_dir.get() else in_dir / "out_raw_tools"
        out_dir.mkdir(parents=True, exist_ok=True)
        raws = sorted(in_dir.glob("*.raw"), key=lambda p: natural_key(p.name))
        if not raws:
            messagebox.showerror("Error", "No .raw files found")
            return None, None, None
        return in_dir, out_dir, raws

    # ------------------------
    # Existing: parallel bin
    # ------------------------
    def run_binned(self):
        setup = self._common_setup()
        if setup is None or setup[0] is None: return
        in_dir, out_dir, raws = setup

        hh, ww = H // BIN_DEFAULT, W // BIN_DEFAULT
        total_files = len(raws)
        total_frames = total_files * N_FRAMES_PER_RAW
        stack16 = np.empty((total_frames, hh, ww), dtype=np.uint16)

        self._log(f"[Bin 16x16] Files: {total_files}, Frames total: {total_frames}, Grid: {hh}x{ww}")
        self._progress(0, total_files)

        def bg():
            t0 = time.perf_counter()
            processed = 0; errors = 0
            print_every = 100

            with ProcessPoolExecutor(max_workers=max(1, int(self.workers.get()))) as ex:
                futs = { ex.submit(read_and_bin_one_raw, str(p)): (i, p) for i, p in enumerate(raws) }
                for fu in as_completed(futs):
                    i, rp = futs[fu]
                    try:
                        payload = fu.result()
                        off = i * N_FRAMES_PER_RAW
                        stack16[off:off+N_FRAMES_PER_RAW] = payload
                        if (i+1) % print_every == 0:
                            self._ui(self._log, f"  processed {i+1}/{total_files} raws")
                    except Exception as e:
                        errors += 1
                        self._ui(self._log, f"[ERROR] {rp.name}: {e}")
                    finally:
                        processed += 1
                        self._ui(self._progress, processed, total_files)

            t1 = time.perf_counter()
            npy16 = out_dir / "stack_16x16_uint16.npy"
            tif16 = out_dir / "stack_16x16_uint16.tif"
            save_t0 = time.perf_counter()
            np.save(npy16, stack16)
            write_tiff_imagej_uint16(tif16, stack16)
            save_t1 = time.perf_counter()

            self._ui(self._log, f"[SAVED] {npy16}")
            self._ui(self._log, f"[SAVED] {tif16}")
            self._ui(self._log, f"[Timing] process(read+bin): {t1 - t0:.3f}s | save: {save_t1 - save_t0:.3f}s | total: {save_t1 - t0:.3f}s")
            self._ui(messagebox.showinfo, "Done (16x16)", f"NPY: {npy16}\nTIFF: {tif16}")

        import threading; threading.Thread(target=bg, daemon=True).start()

    # ------------------------
    # Existing: unbinned slice
    # ------------------------
    def run_slice(self):
        setup = self._common_setup()
        if setup is None or setup[0] is None: return
        in_dir, out_dir, raws = setup

        s = int(self.slice_start.get())
        e = int(self.slice_end.get())

        self._log(f"[Unbinned slice] Range: [{s}, {e})")

        def bg():
            t0 = time.perf_counter()
            raw_slice = extract_unbinned_slice(raws, s, e)
            t1 = time.perf_counter()
            npy8 = out_dir / f"raw_slice_{s}_{e}_uint8.npy"
            tif8 = out_dir / f"raw_slice_{s}_{e}_uint8.tif"
            save_t0 = time.perf_counter()
            np.save(npy8, raw_slice)
            write_tiff_imagej_uint8(tif8, raw_slice)
            save_t1 = time.perf_counter()

            self._ui(self._log, f"[SAVED] {npy8}")
            self._ui(self._log, f"[SAVED] {tif8}")
            self._ui(self._log, f"[Timing] read: {t1 - t0:.3f}s | save: {save_t1 - save_t0:.3f}s | total: {save_t1 - t0:.3f}s")
            self._ui(messagebox.showinfo, "Done (slice)", f"NPY: {npy8}\nTIFF: {tif8}")

        import threading; threading.Thread(target=bg, daemon=True).start()

    # ------------------------
    # NEW: DLL tail→bin→TIFF
    # ------------------------
    def run_tail_dll(self):
        if not self.dll_loaded:
            messagebox.showwarning("DLL not found", "Place raw_tail_bin_tiff.dll next to this script.")
            return

        setup = self._common_setup()
        if setup is None or setup[0] is None: return
        in_dir, out_dir, raws = setup

        # choose raw: explicit file or newest in folder
        raw_sel = self.selected_raw.get().strip()
        if raw_sel:
            raw_path = Path(raw_sel)
        else:
            newest = self._newest_raw(Path(self.in_dir.get()))
            if not newest:
                messagebox.showerror("Error", "No .raw file found to tail.")
                return
            raw_path = newest

        # output TIFF name
        base = raw_path.stem
        B = int(self.bin_factor.get())
        tail_ms = int(self.tail_timeout_ms.get())
        maxf = int(self.max_frames.get())
        out_path = Path(out_dir) / f"{base}_binned_{B}x{B}_DLL.tif"

        self._log(f"[DLL tail] raw: {raw_path.name}  →  {out_path.name}  (bin={B}, tail={tail_ms} ms, max={maxf})")

        def bg():
            t0 = time.perf_counter()
            try:
                pages = self._tail_bin_to_tiff(raw_path, out_path, B, tail_ms, maxf)
            except Exception as e:
                self._ui(self._log, f"[DLL ERROR] {e}")
                return
            dt = time.perf_counter() - t0
            self._ui(self._log, f"[DLL] wrote {pages} pages in {dt:.2f}s → {out_path}")
            self._ui(messagebox.showinfo, "DLL tail done",
                     f"Pages: {pages}\nTIFF: {out_path}")

        import threading; threading.Thread(target=bg, daemon=True).start()

def main(): App().mainloop()

if __name__ == "__main__":
    main()
