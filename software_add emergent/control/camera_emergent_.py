
import os
import sys
import time
import json
import ctypes
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable

# ---- Optional third‑party / SDK imports (graceful fallback) ----
try:
    from EVT_Py import EVT_Py, EVT_Util
    from EVT_Py.EVT_Py import EvtPixelFormat, EvtBitConvert, EvtColorConvert
    EVT_OK = True
except Exception:
    EVT_OK = False
    EVT_Py = None
    EVT_Util = None
    EvtPixelFormat = EvtBitConvert = EvtColorConvert = None

try:
    import numpy as np
except Exception as e:
    raise RuntimeError("camera_emergent needs NumPy installed") from e

try:
    import cv2  # only for optional quick saves when DLL isn't present
except Exception:
    cv2 = None

# From your codebase
try:
    from control._def import Acquisition
except Exception:
    class Acquisition:
        CROP_WIDTH = 1024
        CROP_HEIGHT = 608
        IMAGE_FORMAT = "BMP"

# ---------------- Constants & defaults ----------------
NUM_ALLOCATED_FRAMES = 10
FRAME_PRINTOUT_NUM = 1000

WIDTH_HZ = 1024
HEIGHT_HZ = 608

DEFAULT_OUTPUT_ROOT = Path("output") / "EVT_Py_convert"
DEFAULT_EXTENSION = "bmp"  # for optional per-frame debug saves

# DirectIO DLL (optional, for fast RAW batch writes)
def _load_directio():
    try:
        dll = ctypes.WinDLL(r"drivers and libraries/emergent/DirectIO.dll")
        dll.save_direct_io.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
        dll.save_direct_io.restype = ctypes.c_int
        return dll
    except Exception:
        return None

_DIRECTIO_DLL = _load_directio()

# ---------------- Recording options ----------------
@dataclass
class RecordingOptions:
    # acquisition timing
    exposure_ms: Optional[float] = None   # If None, leave current exposure (e.g., your 0.16 ms default)
    fps: Optional[float] = None           # If None, leave current FPS (e.g., 5000)
    # destination
    choose_destination: bool = True       # prompt for parent folder + run name (no overwrite by default)
    parent_folder: Optional[Path] = None  # if provided, used as default parent path
    run_name: Optional[str] = None        # if provided, used directly; otherwise prompt with a time-stamped default
    # post pipeline
    convert_raw_to_bmp: bool = False
    run_roi_selector: bool = False
    delete_raw_after: bool = False

# ---------------- Small helpers ----------------
def _now_stamp():
    import datetime as _dt
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _tk_choose_parent(default: Optional[Path]) -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        start = str(default or Path.cwd())
        d = filedialog.askdirectory(title="Choose destination parent folder", initialdir=start)
        root.destroy()
        if d:
            return Path(d)
    except Exception:
        pass
    return None

def _tk_ask_string(title: str, prompt: str, initial: str) -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk(); root.withdraw()
        s = simpledialog.askstring(title, prompt, initialvalue=initial)
        root.destroy()
        return s
    except Exception:
        return None

def _windows_path(p: Path) -> str:
    # Ensure backslashes for external tools on Windows
    return str(p.resolve())

# ---------------- Frame saver (RAW batches) ----------------
class _RawBatchWriter:
    """
    Accumulate frames into batches and write as 'batch_<N>.raw' using DirectIO if available,
    else fall back to NumPy .tofile(). Each frame is expected as uint8 HxW.
    """
    def __init__(self, out_dir: Path, width: int, height: int, frames_per_file: int = 10):
        self.out_dir = Path(out_dir)
        self.W, self.H = int(width), int(height)
        self.frames_per_file = int(frames_per_file)
        self._buf = []
        self._batch_idx = 0
        _safe_mkdir(self.out_dir)

    def add(self, frame_u8: np.ndarray):
        if frame_u8.dtype != np.uint8:
            frame_u8 = frame_u8.astype(np.uint8, copy=False)
        self._buf.append(frame_u8)
        if len(self._buf) >= self.frames_per_file:
            self._flush()

    def _flush(self):
        if not self._buf:
            return
        arr = np.stack(self._buf, axis=0)  # (B,H,W)
        fname = self.out_dir / f"batch_{self._batch_idx:05d}.raw"
        data = arr.ravel()  # contiguous uint8
        if _DIRECTIO_DLL is not None:
            ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            nbytes = data.nbytes
            rc = _DIRECTIO_DLL.save_direct_io(str(fname).encode("utf-8"), ptr, nbytes)
            if rc == -1:
                # fallback
                data.tofile(fname)
        else:
            data.tofile(fname)
        self._buf.clear()
        self._batch_idx += 1

    def close(self):
        self._flush()

# ---------------- Camera Simulation (very simple) ----------------
class Camera_Simulation:
    def __init__(self, rotate_image_angle=None, flip_image=None):
        self.rotate_image_angle = rotate_image_angle
        self.flip_image = flip_image
        self.Width = Acquisition.CROP_WIDTH
        self.Height = Acquisition.CROP_HEIGHT
        self._callback: Optional[Callable] = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._is_streaming = False

    def open(self, *_, **__): pass
    def close(self): self._stop.set()

    def set_callback(self, fn: Callable): self._callback = fn
    def enable_callback(self): self.start_software_acquisition()

    def set_line3_to_exposure_active(self): pass

    def start_software_acquisition(self, fps: float = 100.0):
        if self._is_streaming: return
        self._stop.clear()
        self._is_streaming = True
        def loop():
            period = 1.0/max(1e-3, fps)
            frame_id = 0
            while not self._stop.is_set():
                img = (np.random.rand(self.Height, self.Width)*255).astype(np.uint8)
                if self._callback:
                    self._callback(img, frame_id)  # (image, soft_id)
                frame_id += 1
                time.sleep(period)
        self._stream_thread = threading.Thread(target=loop, daemon=True)
        self._stream_thread.start()

    # Stubs to satisfy GUI
    def start_cont_acquisition_and_save(self, *args, **kwargs):
        print("[Sim] start_cont_acquisition_and_save: not writing RAW (simulation).")
    def stop_acquisition(self):
        self._stop.set(); self._is_streaming = False

# ---------------- Main Camera class (Emergent EVT) ----------------
class Camera:
    def __init__(self, sn=None, is_global_shutter=False, rotate_image_angle=None, flip_image=None):
        self.sn = sn
        self.is_global_shutter = is_global_shutter
        self.rotate_image_angle = rotate_image_angle
        self.flip_image = flip_image

        self.Width = Acquisition.CROP_WIDTH or WIDTH_HZ
        self.Height = Acquisition.CROP_HEIGHT or HEIGHT_HZ

        self._evt_ctx = None
        self._cam = None
        self._callback: Optional[Callable] = None

        self._stream_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._raw_writer: Optional[_RawBatchWriter] = None
        self._is_streaming = False

        if not EVT_OK:
            # degrade to simulation if SDK missing
            self.__class__ = Camera_Simulation  # type: ignore
            Camera_Simulation.__init__(self, rotate_image_angle, flip_image)

    # ---- EVT helpers ----
    def _configure_defaults(self):
        # conservative defaults; caller may override via set_exposure_time / set_frame_rate
        try:
            EVT_Util.set_param_str(self._cam, "Bin", "2x2")
            EVT_Util.set_param_max(self._cam, "Width")
            EVT_Util.set_param_max(self._cam, "Height")
            EVT_Util.set_param(self._cam, "FrameRate", 5000)
            EVT_Util.set_param(self._cam, "Exposure", 160)  # 0.16 ms default
            EVT_Util.set_param_max(self._cam, "LineTime")
            EVT_Util.set_param(self._cam, "LineTime", 105)
            EVT_Util.set_param(self._cam, "Gain", 256)
        except Exception:
            pass

    def open(self, index: int = 0):
        device_list = self._evt_ctx().list_devices()
        if not device_list.dev_infos:
            raise RuntimeError("No EVT cameras detected")
        dev = device_list.dev_infos[index]
        params = self._evt_ctx().create_open_camera_params()
        self._cam = self._evt_ctx().open_camera(dev, params)
        self._configure_defaults()

    def _evt_ctx(self):
        if self._evt_ctx is None:
            self._evt_ctx = EVT_Py.EvtContext()
        return self._evt_ctx

    # ---- GUI contract ----
    def set_callback(self, fn: Callable):
        """Register a function(image_u8, frame_id_soft) that receives MONO8 frames."""
        self._callback = fn

    def enable_callback(self):
        self.start_software_acquisition()

    def set_line3_to_exposure_active(self):
        try:
            EVT_Util.set_param_str(self._cam, "Line3Selector", "ExposureActive")
        except Exception:
            pass

    # ---- Basic timing controls ----
    def set_exposure_time(self, exposure_ms: float):
        """Set exposure in milliseconds (works before or during acquisition)."""
        if not EVT_OK or self._cam is None:
            return
        try:
            EVT_Util.set_param(self._cam, "Exposure", float(exposure_ms))
        except Exception:
            pass

    def set_frame_rate(self, fps: float):
        if not EVT_OK or self._cam is None:
            return
        try:
            EVT_Util.set_param(self._cam, "FrameRate", float(fps))
        except Exception:
            pass

    # ---- Streaming (software) ----
    def start_software_acquisition(self, fps: Optional[float] = None):
        if self._is_streaming:
            return
        self._stop.clear()
        self._is_streaming = True

        # Prepare stream
        self._cam.open_stream()
        for _ in range(NUM_ALLOCATED_FRAMES):
            frame = self._cam.allocate_frame()
            self._cam.queue_frame(frame)
        self._cam.execute_command("AcquisitionStart")

        # Optional: update FPS
        if fps is not None:
            self.set_frame_rate(fps)

        def loop():
            frame_soft = 0
            while not self._stop.is_set():
                frame = self._cam.get_frame()
                # convert to MONO8 view (zero-copy)
                conv = self._cam.allocate_convert_frame(
                    frame.width, frame.height, EvtPixelFormat.GVSP_PIX_MONO8,
                    EvtBitConvert.EVT_CONVERT_NONE, EvtColorConvert.EVT_CONVERT_NONE
                )
                frame.convert(conv, EvtBitConvert.EVT_CONVERT_NONE, EvtColorConvert.EVT_CONVERT_NONE, self._cam.get_lines_reorder_handle())
                buf = (ctypes.c_char * conv.buffer_size).from_address(conv.img_ptr)
                img = np.frombuffer(memoryview(buf), dtype=np.uint8).reshape(frame.height, frame.width)
                if self._callback:
                    self._callback(img, frame_soft)
                frame_soft += 1
                self._cam.queue_frame(frame)
                self._cam.release_frame(conv)
            # stop stream
            try:
                self._cam.execute_command("AcquisitionStop")
            except Exception:
                pass

        self._stream_thread = threading.Thread(target=loop, daemon=True)
        self._stream_thread.start()

    def stop_acquisition(self):
        self._stop.set()
        self._is_streaming = False

    # ---- Continuous acquisition + save ----
    def _prepare_destination(self, opt: RecordingOptions) -> Path:
        parent = opt.parent_folder
        if opt.choose_destination or parent is None:
            parent = _tk_choose_parent(parent) or Path.cwd()
        ts = _now_stamp()
        default_name = opt.run_name or f"EVT_Py_convert_{ts}"
        name = _tk_ask_string("Run name", "Enter a run folder name:", default_name) or default_name
        out_dir = Path(parent) / name
        if out_dir.exists():
            # never overwrite silently -> append counter
            k = 1
            base = out_dir
            while (Path(f"{base}_{k}")).exists():
                k += 1
            out_dir = Path(f"{base}_{k}")
        _safe_mkdir(out_dir)
        return out_dir

    def start_cont_acquisition_and_save(self, options: Optional[RecordingOptions] = None):
        """
        Starts a background thread that:
          - streams frames
          - delivers them to GUI callback
          - writes RAW batches as batch_00000.raw, batch_00001.raw, ...
          - (Optionally) after stop, runs RAW->BMP conversion, ROI selector, and deletes RAW.
        """
        if self._is_streaming:
            return
        opt = options or RecordingOptions()
        out_dir = self._prepare_destination(opt) if opt.choose_destination else (opt.parent_folder or DEFAULT_OUTPUT_ROOT)
        _safe_mkdir(out_dir)

        # timing updates
        if opt.exposure_ms is not None:
            self.set_exposure_time(opt.exposure_ms)
        if opt.fps is not None:
            self.set_frame_rate(opt.fps)

        # Start stream
        self._stop.clear()
        self._is_streaming = True
        self._raw_writer = _RawBatchWriter(out_dir, self.Width, self.Height, frames_per_file=10)

        self._cam.open_stream()
        for _ in range(NUM_ALLOCATED_FRAMES):
            frame = self._cam.allocate_frame()
            self._cam.queue_frame(frame)
        self._cam.execute_command("AcquisitionStart")

        def loop():
            frame_soft = 0
            try:
                while not self._stop.is_set():
                    frame = self._cam.get_frame()
                    conv = self._cam.allocate_convert_frame(
                        frame.width, frame.height, EvtPixelFormat.GVSP_PIX_MONO8,
                        EvtBitConvert.EVT_CONVERT_NONE, EvtColorConvert.EVT_CONVERT_NONE
                    )
                    frame.convert(conv, EvtBitConvert.EVT_CONVERT_NONE, EvtColorConvert.EVT_CONVERT_NONE, self._cam.get_lines_reorder_handle())
                    buf = (ctypes.c_char * conv.buffer_size).from_address(conv.img_ptr)
                    img = np.frombuffer(memoryview(buf), dtype=np.uint8).reshape(frame.height, frame.width)

                    # 1) deliver to GUI
                    if self._callback:
                        self._callback(img, frame_soft)

                    # 2) save into RAW batches
                    if self._raw_writer is not None:
                        self._raw_writer.add(img)

                    frame_soft += 1
                    self._cam.queue_frame(frame)
                    self._cam.release_frame(conv)
            finally:
                try: self._cam.execute_command("AcquisitionStop")
                except Exception: pass
                if self._raw_writer is not None:
                    self._raw_writer.close()
                self._post_pipeline(out_dir, opt)
                self._is_streaming = False

        self._stream_thread = threading.Thread(target=loop, daemon=True)
        self._stream_thread.start()
        return out_dir  # so callers can know where the data is saved

    def _post_pipeline(self, out_dir: Path, opt: RecordingOptions):
        """Optionally run: convert RAW->BMP → ROI selection → delete RAW."""
        try:
            if opt.convert_raw_to_bmp:
                self._run_convert_raw_to_bmp(out_dir)
            if opt.run_roi_selector:
                self._run_roi_selector(out_dir)
            if opt.delete_raw_after:
                self._delete_raw_files(out_dir)
        except Exception as e:
            print(f"[PostPipeline] Warning: {e}")

    # ----- hooks into your existing scripts (paths must exist on your machine) -----
    def _run_convert_raw_to_bmp(self, folder: Path):
        # expects convert_raw_to_bmp.py in current working directory or control root
        cand = [
            Path("convert_raw_to_bmp.py"),
            Path(__file__).with_name("convert_raw_to_bmp.py"),
        ]
        for p in cand:
            if p.exists():
                os.system(f'"{sys.executable}" "{_windows_path(p)}" "{_windows_path(folder)}"')
                return
        print("[convert_raw_to_bmp] Script not found. Place convert_raw_to_bmp.py next to camera_emergent.py.")

    def _run_roi_selector(self, folder: Path):
        # Use your analyze_end_to_end_v6 OR a dedicated ROI tool; we call the v6 script with folder
        cand = [
            Path("analyze_experiment_end_to_end_v6.py"),
            Path(__file__).with_name("analyze_experiment_end_to_end_v6.py"),
        ]
        for p in cand:
            if p.exists():
                os.system(f'"{sys.executable}" "{_windows_path(p)}" --folder "{_windows_path(folder)}" --roi_only')
                return
        print("[ROI] analyze_experiment_end_to_end_v6.py not found. Put it next to camera_emergent.py, or disable ROI step.")

    def _delete_raw_files(self, folder: Path):
        n = 0
        for f in Path(folder).glob("batch_*.raw"):
            try:
                f.unlink()
                n += 1
            except Exception:
                pass
        print(f"[Cleanup] Deleted {n} RAW file(s) in {folder}")
