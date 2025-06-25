 #############################################################################
 #
 # Project:      EVT_Py
 # Description:  Python Wrapper for eSDK
 #
 # (c) 2024      by Emergent Vision Technologies Inc
 #               www.emergentvisiontec.com
 #############################################################################

# This is a basic sample application showing how to connect a camera, configure it, and stream images.
import EVT_Py.EVT_Py as EVT_Py
import EVT_Py.EVT_Util as EVT_Uti
import sys
import os
sys.path.append(r'C:\Program Files\EVT\eSDK\bin')  # Use raw string

import os
from pathlib import Path
import ctypes

class _yEvtContext:
    def __init__(self):
        os_name = os.name

        # Get the path for the DLL based on the EMERGENT_DIR environment variable
        evt_c_lib_path = Path(os.environ['EMERGENT_DIR']) / ("eSDK/bin/EmergentCameraC.dll" if os_name == "nt" else "eSDK/lib/libEmergentCameraC.lib")
        evt_c_lib_path = "C:\Program Files\EVT\eSDK\bin\EmergentCameraC.dll"
        print(f"Checking if the eSDK library exists at '{evt_c_lib_path}'")
   
        try:
            # Load the DLL using ctypes
            print("jijijiijijij")
            #self._g_lib = ctypes.windll.LoadLibrary(r"C:\Program Files\EVT\eSDin\EmergentCameraC.dll")
            #self._g_lib = ctypes.windll.LoadLibrary("drivers and libraries/toupcam/windows/toupcam.dll")
            self._g_lib = ctypes.windll.LoadLibrary("drivers and libraries/emergent/EmergentCameraC__.dll")
            
            print("eSDK library loaded successfully.")
        except OSError as e:
            print(f"Error loading the eSDK library: {e}")
            raise

        self._open_cameras = set()



# Import the EVT_Py API
#from EVT_Py import EVT_Py, EVT_Util

# Number of frame buffers to be allocated and used for acquisition
NUM_ALLOCATED_FRAMES = 10

# Number of frames total to grab before closing
NUM_FRAMES_TO_GRAB = 100

# Print status every X frames
FRAME_PRINTOUT_NUM = 25

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

        # Requeue the frame
        cam.queue_frame(frame)

    # Stop streaming
    cam.execute_command("AcquisitionStop")
    print(f"{cam.id}: Stopped streaming")
    print(f"{cam.id}: Dropped frames = {num_dropped_frames}")

def main():
    # Initialize the EVT_Py context
    evt_context = _yEvtContext()
    evt_context = EVT_Py.EvtContext()

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
    open_camera_params = evt_context.create_open_camera_params()

    # Connect to the camera
    cam = evt_context.open_camera(first_dev_info, open_camera_params)

    # Configure the camera. Comment out this, some cameras don't work if parameter set max.
    #configure_camera(cam)

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