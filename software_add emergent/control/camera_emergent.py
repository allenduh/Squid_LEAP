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


# Import the EVT_Py API
from EVT_Py import EVT_Py, EVT_Util

# Import EVT_Py enums directly to remove namespace qualifier
from EVT_Py.EVT_Py import EvtPixelFormat
from EVT_Py.EVT_Py import EvtBitConvert
from EVT_Py.EVT_Py import EvtColorConvert
import matplotlib.pyplot as plt

# Number of frame buffers to be allocated and used for acquisition
NUM_ALLOCATED_FRAMES = 10

# Number of frames total to grab before closing
NUM_FRAMES_TO_GRAB = 2000

# Print status every X frames
FRAME_PRINTOUT_NUM = 1000

WIDTH_HZ = 1024
HEIGHT_HZ = 608

# The path to save the output
OUTPUT_PATH = "output/EVT_Py_convert"

# The output image extension
OUTPUT_EXTENSION = "tiff"

print("loading dll")
# Load the C++ shared library (DLL)
save_lib = ctypes.WinDLL("drivers and libraries/emergent/DirectIO.dll")  # Replace with actual path

# Define the function signature
save_lib.save_direct_io.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
save_lib.save_direct_io.restype = ctypes.c_int  # Returns bytes written
print("loadedd dll")



import threading
import queue
import numpy as np
import time
import cv2
import imageio
from PIL import Image
from multiprocessing import Process, Queue


class FrameSaver_old:
    def __init__(self, save_method="cv2", batch_size=10):
        """Handles saving frames asynchronously in a separate thread."""
        self.save_queue = queue.Queue()
        self.save_method = save_method
        self.batch_size = batch_size  # Number of frames before flushing queue
        self.saving_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.saving_thread.start()

    def save_frame(self, np_image, save_path):
        """Adds a NumPy frame to the queue for asynchronous saving."""
        self.save_queue.put((np_image, save_path))

    def _save_worker(self):
        """Worker thread that processes the saving queue."""
        batch = []  # Store frames in batches to optimize writes
        while True:
            try:
                np_image, save_path = self.save_queue.get(timeout=1)
                if np_image is None:  # Stop signal
                    break
                batch.append((np_image, save_path))

                if len(batch) >= self.batch_size:
                    self._process_batch(batch)
                    batch = []

                self.save_queue.task_done()

            except queue.Empty:
                if batch:
                    self._process_batch(batch)
                    batch = []

    def _process_batch(self, batch):
        """Processes a batch of frames for efficient saving."""
        for np_image, save_path in batch:
            if self.save_method == "numpy":
                np.save(save_path, np_image)

            elif self.save_method == "raw":
                with open(save_path, "wb") as f:
                    f.write(np_image.tobytes())

            elif self.save_method == "cv2":
                cv2.imwrite(save_path, np_image)

            elif self.save_method == "imageio":
                imageio.imwrite(save_path, np_image)

            elif self.save_method == "pillow":
                im = Image.fromarray(np_image, mode="L")  # Assume grayscale
                im.save(save_path, format="BMP")

    def stop(self):
        """Stops the worker thread gracefully."""
        self.save_queue.put((None, None))  # Send stop signal
        self.save_queue.join()  # Ensure all queued frames are processed before exiting
        self.saving_thread.join()
#frame_saver_old = FrameSaver(save_method="numpy")  # Change to "numpy", "raw", "imageio", "pillow" as needed


class FrameSaver:
    def __init__(self):
        self.save_queue = Queue(maxsize=1000)  # Large buffer for saving
        self.saving_process = Process(target=self._save_worker, daemon=True)
        self.saving_process.start()

    def save_frame(self, np_image, save_path):
        """Adds frame to the multiprocessing queue for faster saving."""
        self.save_queue.put((np_image, save_path))

    def _save_worker(self):
        """Worker process for saving frames asynchronously."""
        while True:
            np_image, save_path = self.save_queue.get()
            np.save(save_path, np_image)
    
    def stop(self):
        """Ensure all frames are saved before exiting."""
        self.saving_process.join()



def save_batch_direct_io(filename, np_data):
    """Save batch of frames using direct I/O"""

    # Time the conversion to ctypes pointer
    start_ptr_time = time.time()
    data_ptr = np_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    end_ptr_time = time.time()
    ptr_time = end_ptr_time - start_ptr_time  # Time for pointer conversion

    # Time the direct I/O saving process
    start_io_time = time.time()
    bytes_written = save_lib.save_direct_io(filename.encode('utf-8'), data_ptr, np_data.nbytes)
    end_io_time = time.time()
    io_time = end_io_time - start_io_time  # Time for Direct I/O write

    print(f" data_ptr conversion time: {ptr_time:.6f} sec | Direct I/O time: {io_time:.6f} sec")

    if bytes_written == -1:
        raise IOError("Direct I/O write failed")

def save_batch_np_save(filename, frames):
    np_data = np.concatenate(frames, axis=0).astype(np.uint8)
    start_time = time.time()
    np.save(filename, np_data)
    end_time = time.time()
    print(f"np.save time: {end_time - start_time:.6f} sec")

# Example Usage

frames = [np.random.randint(0, 256, (WIDTH_HZ, HEIGHT_HZ), dtype=np.uint8) for _ in range(100)]
# Make sure the output path exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Test direct I/O method
np_data = np.array(frames)
save_batch_direct_io(f"{OUTPUT_PATH}/testdirect", np_data)


# Test np.save method

save_batch_np_save(f"{OUTPUT_PATH}/test_np_save", frames)


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
    np_image = np.frombuffer(memoryview(buffer_ptr), dtype=np.uint8).reshape(frame.height, frame.width)

    # Release frame memory
    cam.release_frame(conversion_frame)

    # Save asynchronously
    if save_path:
        frame_saver.save_frame(np_image, save_path)

    end_time = time.time()
    print(f"extract_frame_to_numpyy np Conversion time: {end_time - start_time:.6f} sec")

    return np_image


def extract_frame_to_numpy_(cam: EVT_Py.EvtCamera, frame: EVT_Py.EvtFrame):
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
    np_image = np.frombuffer(memoryview(buffer_ptr), dtype=np.uint8).reshape(frame.height, frame.width)

    # Release the conversion frame memory
    cam.release_frame(conversion_frame)

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


def get_sn_by_model(model_name):
    try:
        device_manager = gx.DeviceManager()
        device_num, device_info_list = device_manager.update_device_list()
    except:
        device_num = 0
    if device_num > 0:
        for i in range(device_num):
            if device_info_list[i]['model_name'] == model_name:
                return device_info_list[i]['sn']
    return None # return None if no device with the specified model_name is connected

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

    # EVT_Util.set_param_str(cam, "Bin", "2x2")
    # EVT_Util.set_param_max(cam, "Width")
    # EVT_Util.set_param_max(cam, "Height")
    # EVT_Util.set_param(cam, "FrameRate", 100)
    # EVT_Util.set_param(cam, "Exposure", 260)
    # EVT_Util.set_param(cam, "FrameRate", 3000)

    # EVT_Util.set_param_max(cam, "LineTime")
    # EVT_Util.set_param(cam, "LineTime", 175)
    # EVT_Util.set_param(cam, "Gain", 256)
##
    EVT_Util.set_param_str(cam, "Bin", "2x2")
    EVT_Util.set_param_max(cam, "Width")
    EVT_Util.set_param_max(cam, "Height")
    EVT_Util.set_param(cam, "FrameRate", 100)
    EVT_Util.set_param(cam, "Exposure", 160)
    EVT_Util.set_param(cam, "FrameRate", 5000)

    EVT_Util.set_param_max(cam, "LineTime")
    EVT_Util.set_param(cam, "LineTime", 105)
    EVT_Util.set_param(cam, "Gain", 256)


    pixel_type = EVT_Py.EvtPixelFormat(cam.get_enum_int("PixelFormat"))
    print(f"\tPixelFormat: {pixel_type.name}")

def run_grab_loop(cam: EVT_Py.EvtCamera) -> None:
    # start streaming
    cam.execute_command("AcquisitionStart")

    frame_id_prev = 0
    num_dropped_frames = 0

    # Grab a bunch of frames
    print(f"{cam.id}: Grabbing frames")
    for frame_idx in range(NUM_FRAMES_TO_GRAB):
        frame = cam.get_frame()

        if frame_idx > 0 and frame_idx % FRAME_PRINTOUT_NUM == 0:
            print(f"{cam.id}: {frame_idx} frames")

        # Calculate the number of dropped frames
        expected_frame_id = frame_id_prev + 1
        if frame.frame_id != expected_frame_id and frame_idx != 0: # Ignore very first frame as id is unknown.
            frame_id_diff = frame.frame_id - expected_frame_id
            num_dropped_frames = num_dropped_frames + frame_id_diff

        # In GVSP there is no id 0 so when 16 bit id counter in camera is max then the next id is 1 so set prev id to 0 for math above.
        if frame.frame_id == 65535:
            frame_id_prev = 0
        else:
            frame_id_prev = frame.frame_id

        convert_and_save_frame(cam, frame, f"{OUTPUT_PATH}/test-{cam.id}-{frame_idx}.{OUTPUT_EXTENSION}")
        # Save the last frame to disk.
        # if frame_idx == NUM_FRAMES_TO_GRAB - 1:
        #     print(f"{cam.id}: Saving last frame")
        #     convert_and_save_frame(cam, frame, f"{OUTPUT_PATH}/test-{cam.id}-{frame_idx}.{OUTPUT_EXTENSION}")

        # Requeue the frame
        cam.queue_frame(frame)

    # Stop streaming
    cam.execute_command("AcquisitionStop")
    print(f"{cam.id}: Stopped streaming")
    print(f"{cam.id}: Dropped frames = {num_dropped_frames}")



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

        self.GAIN_MAX = 24
        self.GAIN_MIN = 0
        self.GAIN_STEP = 1
        self.EXPOSURE_TIME_MS_MIN = 0.01
        self.EXPOSURE_TIME_MS_MAX = 4000

        self.trigger_mode = None
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

        self.WidthMax = 4000
        self.HeightMax = 3000
        self.OffsetX = 0
        self.OffsetY = 0

        self.new_image_callback_external = None
        self.frame_queue = queue.Queue(maxsize=10000)  # Buffer frames safely
        self.frame_saver = FrameSaver()
        self.output_path = "output/EVT_Py_convert"
        self.batch_size = 100
        self.max_files_saved = 3000 * 10 / self.batch_size

        

        


    def open(self,index=0):

        # Make sure the output path exists
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
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
        configure_camera(self.camera)



    def set_callback(self,function):
        self.new_image_callback_external = function

    def enable_callback(self):
        self.start_software_acquisition()
        self.callback_is_enabled = True



    def disable_callback(self):
        self.callback_is_enabled = False


    def open_by_sn(self,sn):
        pass

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

    def get_is_color(self):
        return False

    def start_streaming(self):
        self.trigger_mode = "Software"
        self.frame_ID_software = 0
        self.camera.open_stream()
        
        # queue up all our frames
        for _ in range(NUM_ALLOCATED_FRAMES):
            frame = self.camera.allocate_frame()
            self.camera.queue_frame(frame)
        self.camera.execute_command("AcquisitionStart")  
        self.is_streaming = True


        
    def start_software_acquisition(self):
        self.streaming_thread = threading.Thread(target=self.cont_acquisition, daemon=True)
        self.streaming_thread2 = threading.Thread(target=self.cont_acquisition, daemon=True)
        self.streaming_thread.start()    


    def start_cont_acquisition(self):
        """Runs the acquisition loop in a separate thread to allow continuous display."""
        self.trigger_mode = "Contineous"
        # Make sure the output path exists
        os.makedirs(OUTPUT_PATH, exist_ok=True)

 
        # queue up all our frames
        for _ in range(NUM_ALLOCATED_FRAMES):
            frame = self.camera.allocate_frame()
            self.camera.queue_frame(frame)

        self.saving_thread = threading.Thread(target=self.saving_worker, daemon=True)
        self.saving_thread.start()

        
        
        start = time.time()
        frame_idx = 0
        while self.is_streaming:  # Use a flag instead of a fixed loop count

            frame = self.camera.get_frame()
            # Requeue the frame
            self.camera.queue_frame(frame)
            if frame_idx > 0 and frame_idx % FRAME_PRINTOUT_NUM == 0:
                print(f"{self.camera.id}: {frame_idx} frames")

            # Convert frame to NumPy
            frame_pointer = extract_frame_pointer(self.camera, frame)
 

            # Add frame to queue (non-blocking)

            try:
                self.frame_queue.put_nowait(frame_pointer)
                print(f"qaize = {self.frame_queue.qsize()} sending frame {frame_idx}" )

            except queue.Full:
                print(f"total frame = {frame_idx}")
                print(f"Time before queue full : {time.time() - start} seconds")
                print("Warning: Frame queue is full, dropping frame")
                stop_here_hahahah

            # No Display 

            #self.new_image_callback_external(self)

            frame_idx += 1

        print("Acquisition stopped.")
        end = time.time()
        print(f"Total duration: {end - start} seconds")






    def cont_acquisition(self):
        """Continuously acquire frames and send them for live display."""
        start = time.time()
        frame_idx = 0
        while self.is_streaming:  # Use a flag instead of a fixed loop count

            frame = self.camera.get_frame()
            # Requeue the frame
            self.camera.queue_frame(frame)
            if frame_idx > 0 and frame_idx % FRAME_PRINTOUT_NUM == 0:
                print(f"{self.camera.id}: {frame_idx} frames")

            # Convert frame to NumPy
            np_frame = extract_frame_to_numpy(self.camera, frame)
            self.current_frame = np_frame

            # Add frame to queue (non-blocking)
            if self.trigger_mode == "Contineous":
                try:
                    self.frame_queue.put_nowait(np_frame)
                except queue.Full:
                    print("Warning: Frame queue is full, dropping frame")
        

            # Display frame
            #self.new_image_callback_external(np_frame)
            self.new_image_callback_external(self)

            frame_idx += 1

        print("Acquisition stopped.")
        end = time.time()
        print(f"Total duration: {end - start} seconds")





    def saving_worker_one_by_one(self):
        """Continuously save frames from the queue."""
        frame_idx = 0
        while self.is_streaming or not self.frame_queue.empty():
            try:
                buffer_ptr = self.frame_queue.get(timeout=1)  # Wait for a frame pointer
                start_time = time.time()
                #self.frame_saver.save_frame(frame_to_save, f"{OUTPUT_PATH}/testnp-{self.camera.id}-{frame_idx}")
                np_image = np.frombuffer(memoryview(buffer_ptr), dtype=np.uint8).reshape(HEIGHT_HZ, WIDTH_HZ)
                np.save(f"{OUTPUT_PATH}/testnp-{self.camera.id}-{frame_idx}", np_image)
                print(f"File saved in {time.time() - start_time:.6f} sec")

                frame_idx += 1
            except queue.Empty:
                pass  # Avoid blocking if no frames are available
        
        print("Finished saving worker")


    def saving_worker_(self):
        """Save frames from the queue in batches using np.savez_compressed."""
        save_idx = 0
        frame_idx = 0
        allocate_idx = 0
        self.frame_batch = np.zeros((self.batch_size, HEIGHT_HZ, WIDTH_HZ), dtype=np.uint8)  # Pre-allocated 3D array

        while self.is_streaming and save_idx < self.max_files_saved:
            if self.frame_queue:
                print(f"frame idx {frame_idx}")
                # Get the frame from the queue
                start_time = time.time()
                buffer_ptr = self.frame_queue.get(timeout=1)  # Wait for a frame
                np_image = np.frombuffer(memoryview(buffer_ptr), dtype=np.uint8).reshape(HEIGHT_HZ, WIDTH_HZ)
                np_image_time = time.time() - start_time  # Time to convert the frame
                self.frame_queue.task_done()

                # Store the frame in the pre-allocated array
                self.frame_batch[frame_idx] = np_image
                frame_idx += 1
                allocate_idx += 1

                 # If batch is full, save and reset
                if allocate_idx >= self.batch_size:
                    save_start_time = time.time()
                    save_batch_direct_io(f"{self.output_path}/batch_{save_idx}.raw", self.frame_batch)
                    #np.savez(f"{self.output_path}/batch_{frame_idx}", *frame_batch)
                    
                    save_time = time.time() - save_start_time  # Time to save the batch
                    save_idx += 1
                    allocate_idx = 0

                    # Output the timing for the save step
                    print(f"Saved batch {save_idx} | np_image time: {np_image_time:.6f}s | save time: {save_time:.6f}s")
    

                

            # except queue.Empty:
            #     pass  # If the queue is empty, just wait for new frames

        # Save any remaining frames when streaming ends
        print("already saved = " )
        print(frame_idx)
        if frame_batch:
            save_batch_direct_io(f"{self.output_path}/batch_{save_idx}.raw", self.frame_batch)
            print(f"Saved final batch {frame_idx}")
        
        print("Finished saving worker")

    def saving_worker(self):
        """Save frames from the queue in batches using np.savez_compressed."""
        save_idx = 0
        frame_idx = 0
        allocate_idx = 0
        self.frame_batch = np.zeros((self.batch_size, HEIGHT_HZ, WIDTH_HZ), dtype=np.uint8)  # Pre-allocated 3D array

        while self.is_streaming and save_idx < self.max_files_saved:
            try:
                # Get the frame from the queue
                start_time = time.time()
                buffer_ptr = self.frame_queue.get(timeout=1)  # Wait for a frame
                np_image = np.frombuffer(memoryview(buffer_ptr), dtype=np.uint8).reshape(HEIGHT_HZ, WIDTH_HZ)
                np_image_time = time.time() - start_time  # Time to convert the frame

                # Store the frame in the pre-allocated array
                self.frame_batch[frame_idx] = np_image
                frame_idx += 1
                allocate_idx += 1

                # Mark the frame as processed
                self.frame_queue.task_done()  

                # If batch is full, save and reset
                if allocate_idx >= self.batch_size:
                    save_start_time = time.time()
                    print(f"{self.output_path}/batch_{save_idx} HHHHHHHH")
                    save_batch_direct_io(f"{self.output_path}/batch_{save_idx}.raw", self.frame_batch)
                    
                    save_time = time.time() - save_start_time  # Time to save the batch
                    save_idx += 1
                    allocate_idx = 0
                    frame_idx = 0  # Reset frame index

                    print(f"Saved batch {save_idx} | np_image time: {np_image_time:.6f}s | save time: {save_time:.6f}s")

            except queue.Empty:
                pass  # If queue is empty, continue waiting for frames

        print("Finished saving worker.")


    

    def cont_acquisition_and_saving(self):
        """Continuously acquire frames and send them for live display."""

        frame_idx = 0
        while self.is_streaming:  # Use a flag instead of a fixed loop count
            frame = self.camera.get_frame()
            start = time.time()
            np_image = extract_frame_to_numpy(self.camera, frame)
            end = time.time()
            print(f"extract_frame_to_numpy Conversion time: {end - start:.6f} sec")

            if frame_idx > 0 and frame_idx % FRAME_PRINTOUT_NUM == 0:
                print(f"{self.camera.id}: {frame_idx} frames")

                # Convert frame to NumPy
                

                # Queue frame for asynchronous saving
            #frame_saver.save_frame(np_image, f"{OUTPUT_PATH}/testnp-{self.camera.id}-{frame_idx}")

                # Pass the frame to the display callback (ensure it accepts np_frame)
                #self.new_image_callback_external(self)
            self.current_frame = frame_saver.save_frame(np_image, f"{OUTPUT_PATH}/testnp-{self.camera.id}-{frame_idx}")
            # Requeue the frame
            self.camera.queue_frame(frame)

            frame_idx += 1



# high performance saving neglect display
   





    def stop_streaming(self):
        self.is_streaming = False
        self.camera.execute_command("AcquisitionStop")
        self.camera.close_stream()
        print("close stream!")
        if self.streaming_thread.is_alive():  
            self.streaming_thread.join()
            
        if self.trigger_mode == "Contineous":
            if self.saving_thread.is_alive():  
                self.saving_thread.join()  # Wait for the thread to exit
    
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
        print("send trigger at")
        print(self.timestamp)


        # if self.is_streaming:
        #     frame = self.camera.get_frame()

        #     start = time.time()
        #     self.current_frame = extract_frame_to_numpy(self.camera, frame)
        #     end = time.time()
        #     print(f"Conversion time: {end - start} seconds")
        #     self.camera.queue_frame(frame)


        if self.new_image_callback_external is not None and self.callback_is_enabled:
            self.new_image_callback_external(self)
            print("send callback external")
    # def send_trigger(self):
    #     self.frame_ID = self.frame_ID + 1
    #     self.timestamp = time.time()
    #     if self.frame_ID == 1:
    #         if self.pixel_format == 'MONO8':
    #             self.current_frame = np.random.randint(255,size=(self.Height,self.Width),dtype=np.uint8)
    #             self.current_frame[self.Height//2-99:self.Height//2+100,self.Width//2-99:self.Width//2+100] = 200
    #         elif self.pixel_format == 'MONO12':
    #             self.current_frame = np.random.randint(4095,size=(self.Height,self.Width),dtype=np.uint16)
    #             self.current_frame[self.Height//2-99:self.Height//2+100,self.Width//2-99:self.Width//2+100] = 200*16
    #             self.current_frame = self.current_frame << 4
    #         elif self.pixel_format == 'MONO16':
    #             self.current_frame = np.random.randint(65535,size=(self.Height,self.Width),dtype=np.uint16)
    #             self.current_frame[self.Height//2-99:self.Height//2+100,self.Width//2-99:self.Width//2+100] = 200*256
    #     else:
    #         self.current_frame = np.roll(self.current_frame,10,axis=0)
    #         pass 
    #         # self.current_frame = np.random.randint(255,size=(768,1024),dtype=np.uint8)
    #     if self.new_image_callback_external is not None and self.callback_is_enabled:
    #         self.new_image_callback_external(self)            

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



class Camera_Simulation(object):

    def __init__(self,sn=None,is_global_shutter=False,rotate_image_angle=None,flip_image=None):
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

        self.GAIN_MAX = 24
        self.GAIN_MIN = 0
        self.GAIN_STEP = 1
        self.EXPOSURE_TIME_MS_MIN = 0.01
        self.EXPOSURE_TIME_MS_MAX = 4000

        self.trigger_mode = None
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
        # self.resolution=(self.Width,self.Height)
        # self.res_list = [(1000,1000), (2000,2000), (3000,3000), (4000,3000)]
        self.WidthMax = 4000
        self.HeightMax = 3000
        self.OffsetX = 0
        self.OffsetY = 0

        self.new_image_callback_external = None
        

    def open(self,index=0):
        pass


    def set_callback(self,function):
        self.new_image_callback_external = function

    def enable_callback(self):
        self.callback_is_enabled = True

    def disable_callback(self):
        self.callback_is_enabled = False

    def open_by_sn(self,sn):
        pass

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

    def get_is_color(self):
        return False

    def start_streaming(self):
        self.frame_ID_software = 0

    def stop_streaming(self):
        pass

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
        if self.frame_ID == 1:
            if self.pixel_format == 'MONO8':
                self.current_frame = np.random.randint(255,size=(self.Height,self.Width),dtype=np.uint8)
                self.current_frame[self.Height//2-99:self.Height//2+100,self.Width//2-99:self.Width//2+100] = 200
            elif self.pixel_format == 'MONO12':
                self.current_frame = np.random.randint(4095,size=(self.Height,self.Width),dtype=np.uint16)
                self.current_frame[self.Height//2-99:self.Height//2+100,self.Width//2-99:self.Width//2+100] = 200*16
                self.current_frame = self.current_frame << 4
            elif self.pixel_format == 'MONO16':
                self.current_frame = np.random.randint(65535,size=(self.Height,self.Width),dtype=np.uint16)
                self.current_frame[self.Height//2-99:self.Height//2+100,self.Width//2-99:self.Width//2+100] = 200*256
        else:
            self.current_frame = np.roll(self.current_frame,10,axis=0)
            pass 
            # self.current_frame = np.random.randint(255,size=(768,1024),dtype=np.uint8)
        if self.new_image_callback_external is not None and self.callback_is_enabled:
            self.new_image_callback_external(self)

    def read_frame(self):
        return self.current_frame

    def _on_frame_callback(self, user_param, raw_image):
        pass

    def set_ROI(self,offset_x=None,offset_y=None,width=None,height=None):
        pass

    def reset_camera_acquisition_counter(self):
        pass

    def set_line3_to_strobe(self):
        pass

    def set_line3_to_exposure_active(self):
        pass
