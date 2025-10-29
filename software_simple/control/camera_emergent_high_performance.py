"""Camera acquisition and high-speed saving for Emergent EVT_Py cameras.
Conservative style pass: import de-duplication and whitespace cleanup only.
Functionality is intentionally unchanged.
"""

import argparse
import cv2
import time
import numpy as np

from control._def import *

from PIL import Image
import os
import ctypes
import time
import threading
import matplotlib.pyplot as plt
import queue
import imageio
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor


# Import the EVT_Py API
from EVT_Py import EVT_Py, EVT_Util
from EVT_Py.EVT_Py import EvtPixelFormat
from EVT_Py.EVT_Py import EvtBitConvert
from EVT_Py.EVT_Py import EvtColorConvert


# Number of frame buffers to be allocated and used for acquisition
NUM_ALLOCATED_FRAMES = 20

# Number of frames total to grab before closing
NUM_FRAMES_TO_GRAB = 2000

# Print status every X frames
FRAME_PRINTOUT_NUM = 1000

FRAMES_PER_FILE   = 10
HEIGHT_HZ         = 608
WIDTH_HZ          = 1024

# --- Acquisition defaults (override per run) ---
DEFAULT_FPS = 5000              # frames per second
DEFAULT_EXPOSURE_US = 160       # microseconds, e.g. 160 us = 0.16 ms

# The path to save the output
OUTPUT_PATH = "output/EVT_Py_convert"

# Load the C++ shared library (DLL)
save_lib = ctypes.WinDLL("drivers and libraries/emergent/DirectIO.dll")  # Replace with actual path
save_lib.save_direct_io.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
save_lib.save_direct_io.restype = ctypes.c_int  # Returns bytes written


def save_batch_direct_io(filename, np_data):
    """Save batch of frames using direct I/O"""
    data_ptr = np_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    bytes_written = save_lib.save_direct_io(filename.encode('utf-8'), data_ptr, np_data.nbytes)

def save_batch_np_save(filename, frames):
    np_data = np.concatenate(frames, axis=0).astype(np.uint8)
    start_time = time.time()
    np.save(filename, np_data)
    end_time = time.time()
    print(f"np.save time: {end_time - start_time:.6f} sec")


def extract_frame_to_numpy(cam: EVT_Py.EvtCamera, frame: EVT_Py.EvtFrame, save_path=None):
    """Extracts frame data and converts it to a NumPy array using zero-copy memoryview."""
    start_time = time.time()
    
    # Allocate conversion frame
    conversion_frame = cam.allocate_convert_frame(
        frame.width, frame.height, 
        EvtPixelFormat.GVSP_PIX_MONO8,
        EvtBitConvert.EVT_CONVERT_NONE, 
        EvtColorConvert.EVT_CONVERT_NONE
    )

    # Convert frame in-place
    frame.convert(conversion_frame, EvtBitConvert.EVT_CONVERT_NONE, EvtColorConvert.EVT_CONVERT_NONE, cam.get_lines_reorder_handle())

    # Use memoryview for zero-copy access
    buffer_ptr = (ctypes.c_char * conversion_frame.buffer_size).from_address(conversion_frame.img_ptr)
    view = np.frombuffer(memoryview(buffer_ptr), dtype=np.uint8).reshape(frame.height, frame.width)
    np_image = view.copy()  
    # Release frame memory
    cam.release_frame(conversion_frame)

    # Save asynchronously
    if save_path:
        frame_saver.save_frame(np_image, save_path)

    end_time = time.time()
    #print(f"extract_frame_to_numpyy np Conversion time: {end_time - start_time:.6f} sec")

    return np_image



def extract_frame_pointer(cam: EVT_Py.EvtCamera, frame: EVT_Py.EvtFrame):
    """Extracts raw image data from EVT_Py.EvtFrame and converts it to a NumPy array as fast as possible."""
    # Convert the frame directly without allocating extra memory
    conversion_frame = cam.allocate_convert_frame(
        frame.width, frame.height, 
        EvtPixelFormat.GVSP_PIX_MONO8,  # Assume 8-bit mono format
        EvtBitConvert.EVT_CONVERT_NONE, 
        EvtColorConvert.EVT_CONVERT_NONE
    )

    # Convert frame in-place
    frame.convert(conversion_frame, EvtBitConvert.EVT_CONVERT_NONE, EvtColorConvert.EVT_CONVERT_NONE, cam.get_lines_reorder_handle())

    # Use memoryview for zero-copy access instead of converting to bytes
    buffer_ptr = (ctypes.c_char * conversion_frame.buffer_size).from_address(conversion_frame.img_ptr)
    #np_image = np.frombuffer(memoryview(buffer_ptr), dtype=np.uint8).reshape(frame.height, frame.width)

    # Release the conversion frame memory
    cam.release_frame(conversion_frame)

    return buffer_ptr        



# Save a frame as an 8 bit single channel image
def convert_and_save_frame(cam: EVT_Py.EvtCamera, frame: EVT_Py.EvtFrame, path: str)-> None:
    # The frame that we use to convert the input frame to a different format before saving
    conversion_frame = None

    # The image mode (for the PIL library) representing the image format for saving to disk
    image_mode = None

    # Determine if we need to convert the bit depth
    convert_bit_flag = EvtBitConvert.EVT_CONVERT_NONE
    convert_format = EvtPixelFormat.GVSP_PIX_MONO8
    image_mode = "L"
    if(not EVT_Util.is_8bit(frame.pixel_type)):
        convert_bit_flag = EvtBitConvert.EVT_CONVERT_16BIT
        convert_format = EvtPixelFormat.GVSP_PIX_MONO10 # This pixel format is large enough for 16bit mono
        image_mode = "I;16"

    # Allocate an output conversion frame
    conversion_frame = cam.allocate_convert_frame(frame.width, frame.height, 
        convert_format, convert_bit_flag, EvtColorConvert.EVT_CONVERT_NONE)
    
    # Convert the frame
    frame.convert(conversion_frame, convert_bit_flag, EvtColorConvert.EVT_CONVERT_NONE, cam.get_lines_reorder_handle())


    # Copy the image data to a python managed buffer
    img_bytes = bytes((ctypes.c_char * conversion_frame.buffer_size).from_address(conversion_frame.img_ptr))

    # Free the newly allocated conversion frame now that we're done with it
    cam.release_frame(conversion_frame)

    im = Image.frombytes(image_mode, (frame.width, frame.height), img_bytes, 'raw')
    im.save(path)

def configure_camera(cam: EVT_Py.EvtCamera):

    EVT_Util.set_param_str(cam, "Bin", "2x2")
    EVT_Util.set_param_max(cam, "Width")
    EVT_Util.set_param_max(cam, "Height")
    EVT_Util.set_param(cam, "FrameRate", 100)
    EVT_Util.set_param(cam, "Exposure", 160)
    EVT_Util.set_param(cam, "FrameRate", 2000)

    EVT_Util.set_param_max(cam, "LineTime")
    EVT_Util.set_param(cam, "LineTime", 105)
    EVT_Util.set_param(cam, "Gain", 256)


    pixel_type = EVT_Py.EvtPixelFormat(cam.get_enum_int("PixelFormat"))
    print(f"\tPixelFormat: {pixel_type.name}")

def configure_camera(cam: EVT_Py.EvtCamera, fps: int = DEFAULT_FPS, exposure_us: int = DEFAULT_EXPOSURE_US):
    """
    Minimal, predictable config. We only set what we must, and we pass FPS/exposure in.
    exposure_us is in microseconds (160 -> 0.16 ms).
    """
    # Sensor binning and max ROI
    EVT_Util.set_param_str(cam, "Bin", "2x2")
    EVT_Util.set_param_max(cam, "Width")
    EVT_Util.set_param_max(cam, "Height")

    # Exposure / frame rate
    EVT_Util.set_param(cam, "Exposure", int(exposure_us))
    EVT_Util.set_param(cam, "FrameRate", int(fps))

    # Keep your LineTime/Gain, but don't hardcode FPS/exposure again
    EVT_Util.set_param_max(cam, "LineTime")
    #EVT_Util.set_param(cam, "LineTime", 105)
    EVT_Util.set_param(cam, "Gain", 256)

    pixel_type = EVT_Py.EvtPixelFormat(cam.get_enum_int("PixelFormat"))
    print(f"\tPixelFormat: {pixel_type.name} | Exposure(us)={exposure_us} | FPS={fps}")




class Camera(object):

    def __init__(self,sn=None,is_global_shutter=False,rotate_image_angle=None,flip_image=None):

        # Initialize the EVT_Py context
        self.evt_context = EVT_Py.EvtContext()

        # many to be purged
        self.sn = sn
        self.is_global_shutter = is_global_shutter
        self.device_info_list = None
        self.device_index = 0
        self.camera = None
        self.is_color = None
        self.gamma_lut = None
        self.contrast_lut = None
        self.color_correction_param = None

        self.rotate_image_angle = rotate_image_angle
        self.flip_image = flip_image

        self.exposure_time = 0
        self.analog_gain = 0
        self.frame_ID = 0
        self.frame_ID_software = -1
        self.frame_ID_offset_hardware_trigger = 0
        self.timestamp = 0

        self.image_locked = False
        self.current_frame = None 

        self.callback_is_enabled = False
        self.is_streaming = False
        self.high_performance_recording = False

        self.GAIN_MAX = 24
        self.GAIN_MIN = 0
        self.GAIN_STEP = 1
        self.EXPOSURE_TIME_MS_MIN = 0.01
        self.EXPOSURE_TIME_MS_MAX = 4000

        #self.trigger_mode = None
        self.pixel_size_byte = 1

        # below are values for IMX226 (MER2-1220-32U3M) - to make configurable 
        self.row_period_us = 10
        self.row_numbers = 3036
        self.exposure_delay_us_8bit = 650
        self.exposure_delay_us = self.exposure_delay_us_8bit*self.pixel_size_byte
        self.strobe_delay_us = self.exposure_delay_us + self.row_period_us*self.pixel_size_byte*(self.row_numbers-1)

        self.pixel_format = 'MONO8'

        self.is_live = False
        self.Width = Acquisition.CROP_WIDTH
        self.Height = Acquisition.CROP_HEIGHT
        self.resolution = (self.Width, self.Height)

        self.WidthMax = 4000
        self.HeightMax = 3000
        self.OffsetX = 0
        self.OffsetY = 0
        self.max_frames_save = 5*5000  # Important for fast read mode 

        self.new_image_callback_external = None
        self.frame_queue = queue.Queue(maxsize=50000)  # Buffer frames safely
        #self.frame_saver = FrameSaver()
        self.output_path = "output/EVT_Py_convert"
        self.batch_size = 10
        self.bin = 16
        self.frame_batch = np.zeros((self.batch_size, HEIGHT_HZ, WIDTH_HZ), dtype=np.uint8)  # Pre-allocated 3D array
        self.frame_batch_bin = np.zeros((self.max_frames_save, HEIGHT_HZ//self.bin, WIDTH_HZ//self.bin), dtype=np.uint16)  # Pre-allocated 3D array
        self.hh = HEIGHT_HZ // self.bin
        self.ww = WIDTH_HZ // self.bin
        self.want_binning = False
        self.bytes_per_frame = HEIGHT_HZ * WIDTH_HZ


    
    # ---- allocate & manage ping-pong batches ----
    # ---------- Minimal helpers (no ping-pong, no single-frame fallback) ----------
    def _make_work_batch(self):
        total_bytes     = self.bytes_per_frame * self.batch_size
        buf  = (ctypes.c_uint8 * total_bytes)()
        view = np.frombuffer(buf, dtype=np.uint8).reshape(self.batch_size, HEIGHT_HZ, WIDTH_HZ)
        return buf, view

    def _memcpy_frame_into_slot(self, dst_buf, slot, src_ptr):
        # src_ptr can be int address or ctypes.c_void_p
        # if hasattr(src_ptr, "value"):
        #     src_ptr = src_ptr.value
        dst_addr = ctypes.addressof(dst_buf) + slot * self.bytes_per_frame
        ctypes.memmove(dst_addr, src_ptr, self.bytes_per_frame)

    def _bin_batch_view(self, batch_view, bin_size):
        Hbin = (HEIGHT_HZ // bin_size) * bin_size
        Wbin = (WIDTH_HZ  // bin_size) * bin_size
        Hg   = Hbin // bin_size
        Wg   = Wbin // bin_size
        return (batch_view[:, :Hbin, :Wbin]
                .reshape(batch_view.shape[0], Hg, bin_size, Wg, bin_size)
                .mean(axis=(2,4), dtype=np.float32))   # shape (B, Hg, Wg)
        
    def bin_batch_frames(self, arr) -> np.ndarray:
       
        sums = arr.reshape(self.batch_size, self.hh, self.bin, self.ww, self.bin).sum(axis=(2,4), dtype=np.uint32)
        return sums.astype(np.uint16, copy=False)

    def bin_single_frame(self, arr) -> np.ndarray:
        
        sums = arr.reshape(self.hh, self.bin, self.ww, self.bin).sum(axis=(1,3), dtype=np.uint32)
        return sums.astype(np.uint16, copy=False)
            
    def set_output_path(self, path: str):
        """Set per-run output folder and ensure it exists."""
        self.output_path = os.path.abspath(path)
        os.makedirs(self.output_path, exist_ok=True)
        print(f"[Camera] output_path = {self.output_path}")

    def set_max_frame(self, max_frames):
        self.max_frames_save = max_frames

    def open(self,index=0):

        # Make sure the output path exists
        os.makedirs(self.output_path, exist_ok=True)
        
        # Enumerate connected cameras
        device_list = self.evt_context.list_devices()

        num_cameras = len(device_list.dev_infos)

        
        print(f"Cameras detected: {num_cameras}")
        for dev in device_list.dev_infos:
            print(f"\t{dev.camera_id}")

        # Open the first camera
        first_dev_info = device_list.dev_infos[0]
        print(f"Setting up cam: {first_dev_info.camera_id}")

        #(device_num, self.device_info_list) = self.device_manager.update_device_list()
        if num_cameras == 0:
            raise RuntimeError('Could not find any USB camera devices!')

        # Create a new set of parameters to use when opening the camera.
        # We can use these parameters to configue how the camera is opened.
        open_camera_params = self.evt_context.create_open_camera_params()

        # Connect to the camera
        self.camera = self.evt_context.open_camera(first_dev_info, open_camera_params)

        # Configure the camera
        configure_camera(self.camera, fps=getattr(self, "fps", DEFAULT_FPS),
                 exposure_us=getattr(self, "exposure_us", DEFAULT_EXPOSURE_US))



    def set_callback(self,function):
        self.new_image_callback_external = function

    def enable_callback(self):  #To Do understand callback 
        self.start_software_acquisition()
        self.callback_is_enabled = True


    def disable_callback(self):
        self.callback_is_enabled = False



    def close(self):
        pass

    def set_exposure_time(self,exposure_time):
        pass

    def update_camera_exposure_time(self):
        pass

    def set_analog_gain(self,analog_gain):
        pass

    def get_awb_ratios(self):
        pass

    def set_wb_ratios(self, wb_r=None, wb_g=None, wb_b=None):
        pass

    def set_balance_white_auto(self, value):
        pass

    def get_balance_white_auto(self):
        return 0

    def start_streaming(self):
        #self.trigger_mode = "Software"
        self.frame_ID_software = 0
        self.camera.open_stream()
        
        # queue up all our frames
        for _ in range(NUM_ALLOCATED_FRAMES):
            frame = self.camera.allocate_frame()
            self.camera.queue_frame(frame)
        self.camera.execute_command("AcquisitionStart")  
        self.is_streaming = True

    def start_software_acquisition(self):
        self.streaming_thread = threading.Thread(target=self.cont_acquisition, name="streaming_acq", daemon=False)
        self.streaming_thread.start()   

    def start_cont_acquisition_thread(self):  # Used un core live for software acqusision
        self.streaming_thread = threading.Thread(target=self.cont_acquisition, daemon=True)
        self.streaming_thread.start()    

    # KEY function wiidget call toggle_recording_5000fps
    # ---------- Producer (batch-at-source, no ping-pong) ----------
    def start_high_performance_recording(self):
        """
        Batches at source:
        - Copies each frame into a single working contiguous batch
        - On full batch, enqueues a COPY of the batch view to the saver thread
        """
        self.camera.open_stream()
        os.makedirs(self.output_path, exist_ok=True)

        # Prequeue camera buffers
        for _ in range(NUM_ALLOCATED_FRAMES):
            f = self.camera.allocate_frame()
            self.camera.queue_frame(f)

        # Ensure queues/flags
        self.batch_queue = queue.Queue(maxsize=1024)
        self.is_streaming = True
        self.high_performance_recording = True

        # Working batch (single buffer)
        self.work_buf, self.work_view = self._make_work_batch()
        fill_idx   = 0
        batch_id   = 0
        frame_idx  = 0
        dropped    = 0

        # Start saver
        self.saving_thread = threading.Thread(target=self.saving_worker, daemon=True)
        self.saving_thread.start()

        self.camera.execute_command("AcquisitionStart")
        t0 = time.time()

        try:
            while frame_idx < self.max_frames_save and self.high_performance_recording:
                try:
                    frame = self.camera.get_frame()
                    self.camera.queue_frame(frame)  # return to driver ASAP
                    frame_ptr = extract_frame_pointer(self.camera, frame)

                    # Copy into working batch
                    self._memcpy_frame_into_slot(self.work_buf, fill_idx, frame_ptr)
                    fill_idx  += 1
                    frame_idx += 1

                    # If batch full: enqueue a COPY (to avoid overwrite while saver runs)
                    if fill_idx >= self.batch_size:
                        fill_idx = 0
                        batch_id += 1

                    if frame_idx and (frame_idx % FRAME_PRINTOUT_NUM == 0):
                        print(f"[Producer] enqueued frames: {frame_idx}")

                except Exception:
                    dropped += 1
                    # keep going; log if you want

        finally:
            # Flush partial batch (if any)
            if fill_idx > 0:
                self.batch_queue.put((work_view[:fill_idx].copy(), batch_id))

            self.high_performance_recording = False
            t1 = time.time()
            print(f"[Producer] dropped={dropped} | duration={t1 - t0:.3f}s")

    # ---------- Saver (only whole batches) ----------
    def saving_worker(self):
        """
        Consumes (batch_view, batch_id) from self.batch_queue.
        Writes RAW with DirectIO, then optional vectorized binning.
        """


        while self.high_performance_recording or not self.work_buf.empty():
            try:
                batch_view, bid = self.work_buf.get(timeout=0.05)  # (B,H,W) uint8
            except queue.Empty:
                continue

            # 1) save RAW
            out = f"{self.output_path}/batch_{bid}.raw"
            save_batch_direct_io(out, batch_view)

            # 2) optional bin
            if self.want_binning:
                binned = self._bin_batch_view(batch_view, self.bin)  # (B, Hg, Wg) float32
                if hasattr(self, "frame_batch_bin"):
                    start = bid * self.batch_size
                    end   = start + self.batch_size

                    self.frame_batch_bin[start:end] = binned
                    self.work_buf.task_done()

        print("[Saver] finished.")
        self.after_high_performance_recording()

    def after_high_performance_recording(self):
        self.stop_streaming()

        print("save binned frames...")
        if self.want_binning:
            np.save(f"{self.output_path}/binned_16x16", self.frame_batch_bin)
        print("plot binned frames")

    def cont_acquisition(self):  ## Used in software acqusition
        """Continuously acquire frames and send them for live display."""
        start = time.time()
        self.frame_ID = 0
        while self.is_streaming:  # Use a flag instead of a fixed loop count

            frame = self.camera.get_frame()
            # Requeue the frame
            self.camera.queue_frame(frame)
            if self.frame_ID > 0 and self.frame_ID % FRAME_PRINTOUT_NUM == 0:
                print(f"Background acquire: {self.frame_ID} frames")

            # Convert frame to NumPy
            np_frame = extract_frame_to_numpy(self.camera, frame)
            self.current_frame = np_frame

            self.timestamp = time.perf_counter()

            # Display / Save frame in streamhandler -- on new frame
            try:
                self.new_image_callback_external(self)  # or push np_frame into your handler
                self.frame_ID += 1
            except Exception:
                pass
        
            

        print("Acquisition stopped.")
        end = time.time()
        print(f"Total duration: {end - start} seconds")



    def stop_streaming(self):
        self.is_streaming = False
        self.camera.execute_command("AcquisitionStop")
        self.camera.close_stream()
        print("close stream!")
        try:
            if self.streaming_thread.is_alive():  
                self.streaming_thread.join()
        except:
            print("No streaming thread")
        try:
            if self.saving_thread.is_alive():  
                self.saving_thread.join()  # Wait for the thread to exit
        except:
            print("No saving thread")
    
        print("Stream stopped safely!")

    def set_pixel_format(self,pixel_format):
        self.pixel_format = pixel_format
        print(pixel_format)
        self.frame_ID = 0

    def set_continuous_acquisition(self):
        pass

    def set_software_triggered_acquisition(self):
        pass

    def set_hardware_triggered_acquisition(self):
        pass

    def send_trigger(self):
        self.frame_ID = self.frame_ID + 1
        self.timestamp = time.time()

        if self.new_image_callback_external is not None and self.callback_is_enabled:
            self.new_image_callback_external(self)


    def read_frame(self):
        return self.current_frame


    def _on_frame_callback(self, user_param, raw_image):
        print("in_on_frame_callback ")
        self.current_frame = raw_image

    def set_ROI(self,offset_x=None,offset_y=None,width=None,height=None):
        pass

    def reset_camera_acquisition_counter(self):
        pass

    def set_line3_to_strobe(self):
        pass

    def set_line3_to_exposure_active(self):
        pass


