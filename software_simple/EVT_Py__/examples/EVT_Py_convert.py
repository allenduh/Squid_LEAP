 #############################################################################
 #
 # Project:      EVT_Py
 # Description:  Python Wrapper for eSDK
 #
 # (c) 2024      by Emergent Vision Technologies Inc
 #               www.emergentvisiontec.com
 #############################################################################

# This is a basic sample application showing how to connect to multiple cameras, configure them, and access the 
#   acquired frame data by converting and saving an image to disk.
# Assumes that the camera is set to a bayer or mono pixel format. 

from PIL import Image
import os
import ctypes

# Import the EVT_Py API
from EVT_Py import EVT_Py, EVT_Util

# Import EVT_Py enums directly to remove namespace qualifier
from EVT_Py.EVT_Py import EvtPixelFormat
from EVT_Py.EVT_Py import EvtBitConvert
from EVT_Py.EVT_Py import EvtColorConvert

# Number of frame buffers to be allocated and used for acquisition
NUM_ALLOCATED_FRAMES = 10

# Number of frames total to grab before closing
NUM_FRAMES_TO_GRAB = 100

# Print status every X frames
FRAME_PRINTOUT_NUM = 25

# The path to save the output
OUTPUT_PATH = "output/EVT_Py_convert"

# The output image extension
OUTPUT_EXTENSION = "tiff"

# Save a frame as an 8 bit single channel image
def convert_and_save_frame(cam: EVT_Py.EvtCamera, frame: EVT_Py.EvtFrame, path: str)-> None:
    # The frame that we use to convert the input frame to a different format before saving
    conversion_frame = None

    # The image mode (for the PIL library) representing the image format for saving to disk
    image_mode = None

    if EVT_Util.is_bayer(frame.pixel_type):
        # Determine if we need to convert the bit depth
        convert_bit_flag = EvtBitConvert.EVT_CONVERT_NONE
        convert_color_flag = EvtColorConvert.EVT_COLOR_CONVERT_BILINEAR_RGB
        convert_format = EvtPixelFormat.GVSP_PIX_RGB8
        image_mode = "RGB"
        if(not EVT_Util.is_8bit(frame.pixel_type)):
            # Greater than 8 bits, convert to 16bit
            # PIL does not have support for 16bit color, so we convert to 8bit RGB 
            convert_bit_flag = EvtBitConvert.EVT_CONVERT_8BIT
            convert_format = EvtPixelFormat.GVSP_PIX_RGB10 # This pixel format is large enough for 16bit RGB
            convert_color_flag = EvtColorConvert.EVT_COLOR_CONVERT_BILINEAR_RGB

        # Allocate an output conversion frame
        conversion_frame = cam.allocate_convert_frame(frame.width, frame.height, 
            convert_format, convert_bit_flag, convert_color_flag)
        
        # Convert the frame
        frame.convert(conversion_frame, convert_bit_flag, convert_color_flag, cam.get_lines_reorder_handle())
    elif EVT_Util.is_mono(frame.pixel_type):
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
    else:
        print(f"Could not save frame. Unsupported pixel format {frame.pixel_type.name}")

    # Copy the image data to a python managed buffer
    img_bytes = bytes((ctypes.c_char * conversion_frame.buffer_size).from_address(conversion_frame.img_ptr))

    # Free the newly allocated conversion frame now that we're done with it
    cam.release_frame(conversion_frame)

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
        if frame_idx == NUM_FRAMES_TO_GRAB - 1:
            print(f"{cam.id}: Saving last frame")
            convert_and_save_frame(cam, frame, f"{OUTPUT_PATH}/test-{cam.id}.{OUTPUT_EXTENSION}")

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
    open_camera_params = evt_context.create_open_camera_params()

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