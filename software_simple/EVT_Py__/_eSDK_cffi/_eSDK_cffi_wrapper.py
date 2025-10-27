 #############################################################################
 #
 # Project:      EVT_Py
 # Description:  Python Wrapper for eSDK
 #
 # (c) 2024      by Emergent Vision Technologies Inc
 #               www.emergentvisiontec.com
 #############################################################################

import os

from pathlib import Path
from typing import List

from . import _eSDK_cffi_loader

_ENUM_STR_MAX_SIZE = 256

# Constants from eSDK
_EVT_FRAME_BUFFER_DEFAULT = 0
_EVT_FRAME_BUFFER_ZERO_COPY = 1
_EVT_INFINITE = -1

_g_ffi = _eSDK_cffi_loader.load_cffi()
_g_lib = None

class EvtException(Exception):
    pass

def _check_result_evt_ffi(res: int, func_name: str) -> None:
    if res != 0:
        raise EvtException(f"[ERROR] {func_name} returned {res}")

# @TODO pass args as list
def _call_evt_ffi(func_name: str, *args):
    # Don't check for return value if there is none (ie, void)
    res = getattr(_g_lib, func_name)(*args)
    if res != None:
        _check_result_evt_ffi(res, func_name)

class _EvtFrame:
    def __init__(self, frame_ffi):
        self._frame_ffi = frame_ffi
        self.width = frame_ffi.size_x
        self.height = frame_ffi.size_y
        self.frame_id = frame_ffi.frame_id
        self.img_ptr = int(_g_ffi.cast("unsigned long long", frame_ffi.imagePtr))
        self.buffer_size = frame_ffi.bufferSize
        self.pixel_type = frame_ffi.pixel_type
    
    # Updates member data
    def update(self, frame_ffi):
        self.width = frame_ffi.size_x
        self.height = frame_ffi.size_y
        self.frame_id = frame_ffi.frame_id
        self.img_ptr = int(_g_ffi.cast("unsigned long long", frame_ffi.imagePtr))
        self.buffer_size = frame_ffi.bufferSize
        self.pixel_type = frame_ffi.pixel_type

    def convert(self, frame_dest, convert_bit_depth: int, convert_color: int, handle):
        _call_evt_ffi("EVT_FrameConvert", self._frame_ffi, frame_dest._frame_ffi, convert_bit_depth, convert_color, handle)

        # Update the frame data after conversion
        frame_dest.update(frame_dest._frame_ffi) 

class _EvtCamera:
    def __init__(self, handle, id: str):
        self._handle = handle
        self.id = id

    def open_stream(self) -> None:
        _call_evt_ffi("EVT_CameraOpenStream", self._handle)

    def close_stream(self) -> None:
        _call_evt_ffi("EVT_CameraCloseStream", self._handle)

    def allocate_frame(self) -> _EvtFrame:
        width = self.get_param_uint32("Width")
        height = self.get_param_uint32("Height")
        pixel_type = self.get_enum_int("PixelFormat")
        
        frame_ffi = _g_ffi.new("EmergentFrame*")
        frame_ffi.size_x = width
        frame_ffi.size_y = height
        frame_ffi.pixel_type = pixel_type

        _call_evt_ffi("EVT_AllocateFrameBuffer", self._handle, frame_ffi, _EVT_FRAME_BUFFER_ZERO_COPY)

        return _EvtFrame(frame_ffi)

    def allocate_convert_frame(self, width: int, height: int, pixel_type: int,  convert_bit_depth: int, convert_color: int) -> _EvtFrame:
        frame_dest_ffi = _g_ffi.new("EmergentFrame*")
        frame_dest_ffi.size_x = width
        frame_dest_ffi.size_y = height
        frame_dest_ffi.pixel_type = pixel_type
        frame_dest_ffi.convertColor = convert_color
        frame_dest_ffi.convertBitDepth = convert_bit_depth
        _call_evt_ffi("EVT_AllocateFrameBuffer", self._handle, frame_dest_ffi, _EVT_FRAME_BUFFER_DEFAULT)
        return _EvtFrame(frame_dest_ffi)

    def release_frame(self, frame: _EvtFrame) -> None:
        _call_evt_ffi("EVT_ReleaseFrameBuffer", self._handle, frame._frame_ffi)

    def queue_frame(self, frame: _EvtFrame) -> None:
        _call_evt_ffi("EVT_CameraQueueFrame", self._handle, frame._frame_ffi)

    def get_frame(self) -> _EvtFrame:
        frame_ffi = _g_ffi.new("EmergentFrame*")
        _call_evt_ffi("EVT_CameraGetFrame", self._handle, frame_ffi, _EVT_INFINITE)
        frame = _EvtFrame(frame_ffi)
        return frame

    def execute_command(self, command: str) -> None:
        name_ffi = _g_ffi.new("char[]", bytes(command, "ascii"))
        _call_evt_ffi("EVT_CameraExecuteCommand", self._handle, name_ffi)

    def get_param_uint32(self, param: str) -> int:
        name_ffi = _g_ffi.new("char[]", bytes(param, "ascii"))
        val_ffi = _g_ffi.new("unsigned int*")
        _call_evt_ffi("EVT_CameraGetUInt32Param", self._handle, name_ffi, val_ffi)
        return val_ffi[0]

    def set_param_uint32(self, param: str, val: int) -> None:
        name_ffi = _g_ffi.new("char[]", bytes(param, "ascii"))
        _call_evt_ffi("EVT_CameraSetUInt32Param", self._handle, name_ffi, val)

    def get_max_param_uint32(self, param: str) -> int:
        name_ffi = _g_ffi.new("char[]", bytes(param, "ascii"))
        val_ffi = _g_ffi.new("unsigned int*")
        _call_evt_ffi("EVT_CameraGetUInt32ParamMax", self._handle, name_ffi, val_ffi)
        return val_ffi[0]

    def get_min_param_uint32(self, param: str) -> int:
        name_ffi = _g_ffi.new("char[]", bytes(param, "ascii"))
        val_ffi = _g_ffi.new("unsigned int*")
        _call_evt_ffi("EVT_CameraGetUInt32ParamMin", self._handle, name_ffi, val_ffi)
        return val_ffi[0]

    def get_enum_str(self, param: str) -> str:
        name_ffi = _g_ffi.new("char[]", bytes(param, "ascii"))
        enum_str_ffi = _g_ffi.new("char[]", _ENUM_STR_MAX_SIZE)
        _call_evt_ffi("EVT_CameraGetStringEnumParam", self._handle, name_ffi, enum_str_ffi, _ENUM_STR_MAX_SIZE, _g_ffi.NULL)
        return _g_ffi.string(enum_str_ffi).decode("ascii")

    def set_enum_str(self, param: str, enum_str: str) -> None:
        name_ffi = _g_ffi.new("char[]", bytes(param, "ascii"))
        enum_str_ffi = _g_ffi.new("char[]", bytes(enum_str, "ascii"))
        _call_evt_ffi("EVT_CameraSetEnumParam", self._handle, name_ffi, enum_str_ffi)

    def get_enum_int(self, param: str) -> int:
        name_ffi = _g_ffi.new("char[]", bytes(param, "ascii"))
        val_ffi = _g_ffi.new("unsigned int*")
        _call_evt_ffi("EVT_CameraGetValueEnumParam", self._handle, name_ffi, val_ffi)
        return val_ffi[0]

    def get_lines_reorder_handle(self):
        return getattr(_g_lib, "EVT_GetLinesReorderHandle")(self._handle)

# Wrapper around the FFI device info set so that the memory remains valid during the lifetype of the _EvtDeviceInfo
class _EvtDeviceInfoSetWrapper:
    def __init__(self, dev_info_set_ffi):
        self.dev_info_set_ffi = dev_info_set_ffi

    def __del__(self):
        _call_evt_ffi("EVT_DeviceInfoSetCleanup", self.dev_info_set_ffi)

class _EvtDeviceInfo:
    def __init__(self, camera_id: str,  dev_idx: int, dev_info_set_wrapper: _EvtDeviceInfoSetWrapper):
        self.camera_id = camera_id
        self.dev_idx = dev_idx
        self.dev_info_set_wrapper = dev_info_set_wrapper

class _EvtDeviceList:
    def __init__(self, dev_infos: List[_EvtDeviceInfo]):
        self.dev_infos = dev_infos

class _EvtCameraOpenParams:
    def __init__(self, open_params_ffi):
        self.disable_heartbeat = open_params_ffi.disableHeartbeat
        self.line_reorder_cpu_mask = open_params_ffi.lineReorderCPUMask
        self.gpu_direct_device_id = open_params_ffi.gpuDirectDeviceId

        self.open_params_ffi = open_params_ffi
    
    def update_ffi(self):
        self.open_params_ffi.disableHeartbeat = self.disable_heartbeat
        self.open_params_ffi.lineReorderCPUMask = self.line_reorder_cpu_mask
        self.open_params_ffi.gpuDirectDeviceId = self.gpu_direct_device_id

class _EvtContext:
    def __init__(self):
        os_name = os.name
    
        # Detect the path for the eSDK C API library
        if os_name == "nt":
            evt_c_lib_path = Path(os.environ['EMERGENT_DIR']) / Path("eSDK/bin/EmergentCameraC.dll")
        else:
            evt_c_lib_path = Path(os.environ['EMERGENT_DIR']) / Path("eSDK/lib/libEmergentCameraC.so")

        print(f"Loading eSDK library from '{evt_c_lib_path}'")

        global _g_lib
        _g_lib = _g_ffi.dlopen(str(evt_c_lib_path))
        self._open_cameras = set()

    def __del__(self):
        for cam in self._open_cameras:
            _call_evt_ffi( "EVT_CameraClose", cam._handle)
        self._open_cameras = set()

    def list_devices(self) -> _EvtDeviceList:
        dev_info_set_ffi = _g_ffi.new("LabviewDeviceSet*")

        _call_evt_ffi("EVT_ListDevices", dev_info_set_ffi)

        # Wrap the device info set so that we can make sure the memory is valid during the lifetime of the device infos
        # Cleans up the memory when no longer referenced
        dev_info_set_wrapper = _EvtDeviceInfoSetWrapper(dev_info_set_ffi)

        dev_infos = []
        for dev_idx in range(dev_info_set_ffi.countDevice):
            dev_info_ffi = _g_ffi.new("LabviewDeviceInfo*")            
            
            _call_evt_ffi("EVT_GetDeviceInfo", dev_info_set_ffi, dev_info_ffi, dev_idx)
            
            camera_id = _g_ffi.string(dev_info_ffi.serialNumber).decode("ascii")

            dev_infos.append(_EvtDeviceInfo(camera_id, dev_idx, dev_info_set_wrapper))

        return _EvtDeviceList(dev_infos)

    def create_open_camera_params(self) -> _EvtCameraOpenParams:
        cam_open_params = _g_ffi.new("CameraOpenParams*")
        _call_evt_ffi("EVT_CameraOpenParams_Init", cam_open_params)

        return _EvtCameraOpenParams(cam_open_params)

    def open_camera(self, dev_info: _EvtDeviceInfo, params: _EvtCameraOpenParams) -> _EvtCamera:
        params.update_ffi()

        cam_handle = _g_ffi.new("Camera_Handle*")
        _call_evt_ffi( "EVT_CameraOpen_Ex", cam_handle, dev_info.dev_info_set_wrapper.dev_info_set_ffi, dev_info.dev_idx, params.open_params_ffi)

        cam = _EvtCamera(cam_handle[0], dev_info.camera_id)
        self._open_cameras.add(cam)

        return cam


    def close_camera(self, cam: _EvtCamera) -> None:
        _call_evt_ffi( "EVT_CameraClose", cam._handle)
        self._open_cameras.remove(cam)
