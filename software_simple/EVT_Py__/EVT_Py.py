 #############################################################################
 #
 # Project:      EVT_Py
 # Description:  Python Wrapper for eSDK
 #
 # (c) 2024      by Emergent Vision Technologies Inc
 #               www.emergentvisiontec.com
 #############################################################################

from ._eSDK_cffi._eSDK_cffi_wrapper import _EvtContext, _EvtCameraOpenParams, _EvtDeviceList, _EvtDeviceInfo, _EvtCamera, _EvtFrame, EvtException

from enum import IntEnum

# The GVSP Pixel format
# Taken from emergentgigevisiondef.h
class EvtPixelFormat(IntEnum):
    GVSP_PIX_UNKOWN = 0
    GVSP_PIX_MONO8 = 17301505
    GVSP_PIX_MONO10 = 17825795
    GVSP_PIX_MONO10_PACKED = 17563652
    GVSP_PIX_MONO12 = 17825797
    GVSP_PIX_MONO12_PACKED = 17563654
    GVSP_PIX_BAYGB8 = 17301514
    GVSP_PIX_BAYGB10 = 17825806
    GVSP_PIX_BAYGB10_PACKED = 17563688
    GVSP_PIX_BAYGB12 = 17825810
    GVSP_PIX_BAYGB12_PACKED = 17563692
    GVSP_PIX_BAYGR8 = 17301512
    GVSP_PIX_BAYGR10 = 17825804
    GVSP_PIX_BAYGR10_PACKED = 17563686
    GVSP_PIX_BAYGR12 = 17825808
    GVSP_PIX_BAYGR12_PACKED = 17563690
    GVSP_PIX_BAYRG8 = 17301513
    GVSP_PIX_BAYRG10 = 17825805
    GVSP_PIX_BAYRG10_PACKED = 17563687
    GVSP_PIX_BAYRG12 = 17825809
    GVSP_PIX_BAYRG12_PACKED = 17563691
    GVSP_PIX_BAYBG8 = 17301515
    GVSP_PIX_BAYBG10 = 17825807
    GVSP_PIX_BAYBG10_PACKED = 17563689
    GVSP_PIX_BAYBG12 = 17825811
    GVSP_PIX_BAYBG12_PACKED = 17563693
    GVSP_PIX_RGB8 = 35127316
    GVSP_PIX_BGR8 = 35127317
    GVSP_PIX_RGB10 = 36700184
    GVSP_PIX_RGB12 = 36700186
    GVSP_PIX_BGR10 = 36700185
    GVSP_PIX_BGR12 = 36700187
    GVSP_PIX_YUV411_PACKED = 34340894
    GVSP_PIX_YUV422_PACKED = 34603039
    GVSP_PIX_YUV444_PACKED = 35127328

    UNPACK_PIX_MONO8 = 285753345
    UNPACK_PIX_MONO16 = 286277634
    UNPACK_PIX_BAYGB8 = 285753347
    UNPACK_PIX_BAYGB16 = 286277636
    UNPACK_PIX_BAYGR8 = 285753354
    UNPACK_PIX_BAYGR16 = 286277643
    UNPACK_PIX_BAYRG8 = 285753356
    UNPACK_PIX_BAYRG16 = 286277645
    UNPACK_PIX_BAYBG8 = 285753358
    UNPACK_PIX_BAYBG16 = 286277647
    UNPACK_PIX_RGB8 = 319324165
    UNPACK_PIX_RGB16 = 319848454
    UNPACK_PIX_BGR8 = 319324167
    UNPACK_PIX_BGR16 = 319848456
    UNPACK_PIX_YUV8 = 319324169
    CONVERT_PIX_MONO8 = 554188801
    CONVERT_PIX_MONO16 = 554713090
    CONVERT_PIX_BAYGB8 = 554188803
    CONVERT_PIX_BAYGB16 = 554713092
    CONVERT_PIX_BAYGR8 = 554188810
    CONVERT_PIX_BAYGR16 = 554713099
    CONVERT_PIX_BAYRG8 = 554188812
    CONVERT_PIX_BAYRG16 = 554713101
    CONVERT_PIX_RGB8 = 587759621
    CONVERT_PIX_RGB16 = 588283910
    CONVERT_PIX_BGR8 = 587759623
    CONVERT_PIX_BGR16 = 588283912
    CONVERT_PIX_YUV8 = 587759625

# Image bit conversion constants
class EvtBitConvert(IntEnum):
    EVT_CONVERT_NONE = 0
    EVT_CONVERT_8BIT = 8
    EVT_CONVERT_16BIT = 16

# Image color conversion constants
class EvtColorConvert(IntEnum):
    EVT_CONVERT_NONE = 0
    EVT_COLOR_CONVERT_BILINEAR_RGB = 3
    EVT_COLOR_CONVERT_BILINEAR_BGR = 11

# A frame as received from an EvtCamera
class EvtFrame:
    def __init__(self, impl: _EvtFrame):
        self._impl = impl

        # The dimensions of the frame
        self.width = impl.width
        self.height = impl.height

        # The GVSP frame id
        self.frame_id = impl.frame_id

        # The raw image pointer as exposed through FFI.
        # This is the integer memory address of the image buffer. Care must be taken when accessing this pointer 
        # as the underlying memory is not managed by the python runtime. 
        # 
        # This buffer can be converted to a python managed `bytes` object through 
        # bytes((ctypes.c_char * frame.buffer_size).from_address(frame.img_ptr))
        self.img_ptr = impl.img_ptr

        # The size in bytes of img_ptr
        self.buffer_size = impl.buffer_size

        # The pixel format of the frame as defined in "emergentgigevisiondef.h"
        self.pixel_type = EvtPixelFormat(impl.pixel_type)

    # Updates member data
    def update(self, impl: _EvtFrame):
        self.width = impl.width
        self.height = impl.height
        self.frame_id = impl.frame_id
        self.img_ptr = impl.img_ptr
        self.buffer_size = impl.buffer_size
        self.pixel_type = EvtPixelFormat(impl.pixel_type)

    # Converts a frame according to the conversion parameters
    def convert(self, frame_dest, convert_bit_depth: EvtBitConvert, convert_color: EvtColorConvert, handle):
        self._impl.convert(frame_dest._impl, int(convert_bit_depth), int(convert_color), handle)

        # Update the frame data after conversion
        frame_dest.update(frame_dest._impl)

class EvtCamera:
    def __init__(self, impl: _EvtCamera):
        self._impl = impl
        self.id = impl.id
    
    # Opens a stream channel
    def open_stream(self) -> None:
        self._impl.open_stream()
    
    # Closes a stream channel
    def close_stream(self) -> None:
        self._impl.close_stream()

    # Allocates a frame matching the current settings of the camera. 
    # Returns the allocated frame.
    def allocate_frame(self) -> EvtFrame:
        return EvtFrame(self._impl.allocate_frame())

    # Allocates a frame to be used with the EvtFrame.convert(...) function
    def allocate_convert_frame(self, width: int, height: int, pixel_type: EvtPixelFormat, convert_bit_depth: EvtBitConvert, convert_color: EvtColorConvert) -> _EvtFrame:
        return EvtFrame(self._impl.allocate_convert_frame(width, height, int(pixel_type), int(convert_bit_depth), int(convert_color)))

    # Releases (deallocates) a frame
    def release_frame(self, frame: EvtFrame) -> None:
        self._impl.release_frame(frame._impl)

    # Queues a frame to be used in acquisition
    def queue_frame(self, frame: EvtFrame) -> None:
        self._impl.queue_frame(frame._impl)

    # Gets a frame from the camera.
    # Blocks until frame is recieved.
    # Returns the frame.
    def get_frame(self) -> EvtFrame:
       return EvtFrame(self._impl.get_frame())

    # Executes a command.
    def execute_command(self, command: str) -> None:
       self._impl.execute_command(command)

    # Gets a uint32 parameter.
    # Returns the parameter value.
    def get_param_uint32(self, param: str) -> int:
       return self._impl.get_param_uint32(param)

    # Sets a uint32 parameter.
    def set_param_uint32(self, param: str, val: int) -> None:
        self._impl.set_param_uint32(param, val)

    # Gets the max value of a uint32 parameter.
    def get_max_param_uint32(self, param: str) -> int:
        return self._impl.get_max_param_uint32(param)

    # Gets the min value of a uint32 parameter.
    def get_min_param_uint32(self, param: str) -> int:
        return self._impl.get_min_param_uint32(param)

    # Gets the string of the current entry of a enum parameter.
    def get_enum_str(self, param: str) -> int:
        return self._impl.get_enum_str(param)
    
    # Sets the current entry of a enum parameter by the string.
    def set_enum_str(self, param: str, enum_str: str) -> int:
        return self._impl.set_enum_str(param, enum_str)
    
    # Gets the int current entry of a enum parameter.
    def get_enum_int(self, param: str) -> int:
        return self._impl.get_enum_int(param)

    # Gets lines reorder handle
    def get_lines_reorder_handle(self):
        return self._impl.get_lines_reorder_handle()

class EvtDeviceInfo:
    def __init__(self, impl: _EvtDeviceInfo):
        self._impl = impl
        self.camera_id = impl.camera_id

class EvtDeviceList:
    def __init__(self, impl: _EvtDeviceList):
        self._impl = impl
        self.dev_infos = [EvtDeviceInfo(x) for x in impl.dev_infos]

class EvtCameraOpenParams:
    def __init__(self, impl: _EvtCameraOpenParams):
        self._impl = impl
        
        # Disables the heartbeat of the camera. Default false
        self.disable_heartbeat = impl.disable_heartbeat
        
        # The bit mask for reorder threads and cores. Default 0x0000000F: 4 threads pinned to core 0 - 3; 0: Line reordering is disabled.
        self.line_reorder_cpu_mask = impl.line_reorder_cpu_mask

        # Not use GPU-direct if < 0; specify GPU Id if >= 0, Default -1
        self.gpu_direct_device_id = impl.gpu_direct_device_id

    def _update_impl(self):
        self._impl.disable_heartbeat = self.disable_heartbeat
        self._impl.line_reorder_cpu_mask = self.line_reorder_cpu_mask
        self._impl.gpu_direct_device_id = self.gpu_direct_device_id

class EvtContext:
    def __init__(self):
        self._impl = _EvtContext()

    # Lists the connected devices
    def list_devices(self) -> EvtDeviceList:
        return EvtDeviceList(self._impl.list_devices())

    # Returns default initialized camera open parameters
    def create_open_camera_params(self) -> EvtCameraOpenParams:
        return EvtCameraOpenParams(self._impl.create_open_camera_params())

    # Connects to a camera.
    # Returns a handle to that camera
    def open_camera(self, dev_info: EvtDeviceInfo, open_params: EvtCameraOpenParams = None) -> EvtCamera:
        if open_params == None:
            open_params = self.create_open_camera_params()

        open_params._update_impl()
        return EvtCamera(self._impl.open_camera(dev_info._impl, open_params._impl))

    # Disconnects a camera
    def close_camera(self, cam: EvtCamera) -> None:
        self._impl.close_camera(cam._impl)
