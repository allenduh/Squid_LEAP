 #############################################################################
 #
 # Project:      EVT_Py
 # Description:  Python Wrapper for eSDK
 #
 # (c) 2024      by Emergent Vision Technologies Inc
 #               www.emergentvisiontec.com
 #############################################################################

# This is a sample application showing how to use GPUDirect, and access the acquired frame data using numba.
# Assumes that the camera is set to an 8-bit pixel format. 

from EVT_Py import EVT_Py, EVT_Util

from PIL import Image
import os
import ctypes

import numba
import numba.cuda

# Number of frame buffers to be allocated and used for acquisition
NUM_ALLOCATED_FRAMES = 10

# Number of frames total to grab before closing
NUM_FRAMES_TO_GRAB = 100

# Print status every X frames
FRAME_PRINTOUT_NUM = 25

# The index of the GPU device for using GPU Direct.
GPU_DEVICE_ID = 0

# The path to save the output
OUTPUT_PATH = "output/EVT_Py_numba/"

# The output image extension
OUTPUT_EXTENSION = "tiff"

# A basic wrapper around an EvtFrame which implements the CUDA Array Interface from numba 
class FrameWrapperNumba:
    def __init__(self, frame: EVT_Py.EvtFrame):
        self.__cuda_array_interface__ = {"shape": (frame.width, frame.height), "typestr": "<u1", "data": (frame.img_ptr, True), "version": 3, "strides": None, "descr": None, "mask": None, "stream": None}
        pass

def save_frame_gpu(frame: EVT_Py.EvtFrame, path: str, gpu_id: int)-> None:
    # Only support single channel 8bit images
    supported_pixel_type = EVT_Util.is_8bit(frame.pixel_type) and (EVT_Util.is_bayer(frame.pixel_type) or EVT_Util.is_mono(frame.pixel_type))
    if not supported_pixel_type:
        print(f"Could not save frame. Unsupported pixel format {frame.pixel_type}")
        return

    # Select the device you wish to use
    numba.cuda.select_device(gpu_id)

    # Assuming 8bit single channel images
    image_mode = 'L'

    # Create a wrapper around the frame and convert that frame to a numba cuda array.
    # No memory should be copied during this conversion.
    frame_wrapper_numba = FrameWrapperNumba(frame)
    frame_numba_device = numba.cuda.as_cuda_array(frame_wrapper_numba)
    
    # At this point you can use any numba operation to operate on the GPU memory

    # Copy the GPU memory to main system memory and convert to a `bytes` object
    frame_numba_host = frame_numba_device.copy_to_host()
    img_bytes = frame_numba_host.data.tobytes()

    im = Image.frombytes(image_mode, (frame.width, frame.height), img_bytes, 'raw')
    im.save(path)

def configure_camera(cam: EVT_Py.EvtCamera):
    EVT_Util.set_param_max(cam, "Width")
    EVT_Util.set_param_max(cam, "Height")
    EVT_Util.set_param_max(cam, "FrameRate")
    EVT_Util.set_param_max(cam, "Exposure")

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

        # Save the last frame to disk.
        # Note that we cannot directly save frames using GPU direct because the memory exists on the GPU.
        # To handle frames using GPU Direct, the data must either be moved to host (CPU) memory or processed using GPU based functions.  
        if frame_idx == NUM_FRAMES_TO_GRAB - 1:
            print(f"{cam.id}: Saving last frame")
            save_frame_gpu(frame, f"{OUTPUT_PATH}/test-{cam.id}.{OUTPUT_EXTENSION}", GPU_DEVICE_ID)

        # Requeue the frame
        cam.queue_frame(frame)

    # Stop streaming
    cam.execute_command("AcquisitionStop")
    print(f"{cam.id}: Stopped streaming")
    print(f"{cam.id}: Dropped frames = {num_dropped_frames}")

def main():
    # Initialize the EVT_Py context
    evt_context = EVT_Py.EvtContext()

    # Make sure the output path exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Enumerate connected cameras
    device_list = evt_context.list_devices()

    num_cameras = len(device_list.dev_infos)
    
    print(f"Cameras detected: {num_cameras}")
    for dev in device_list.dev_infos:
        print(f"\t{dev.camera_id}")

    if num_cameras == 0:
        return

    # Open the first camera
    first_dev_info = device_list.dev_infos[0]
    print(f"Setting up cam: {first_dev_info.camera_id}")

    # Create a new set of parameters to use when opening the camera.
    # We can use these parameters to configue how the camera is opened.
    # Eg. Set GPUDirect device ID
    open_camera_params = evt_context.create_open_camera_params()
    open_camera_params.gpu_direct_device_id = GPU_DEVICE_ID

    # Connect to the camera
    cam = evt_context.open_camera(first_dev_info, open_camera_params)

    # Configure the camera
    configure_camera(cam)

    # Open the camera for streaming
    cam.open_stream()

    # queue up all our frames
    for _ in range(NUM_ALLOCATED_FRAMES):
        frame = cam.allocate_frame()
        cam.queue_frame(frame)

    # Run our grab loop
    run_grab_loop(cam)

    # Close the stream and the camera
    cam.close_stream()
    evt_context.close_camera(cam)

    print("Complete!")
    
if __name__ == "__main__":
    main()