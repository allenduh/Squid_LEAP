 #############################################################################
 #
 # Project:      EVT_Py
 # Description:  Python Wrapper for eSDK
 #
 # (c) 2024      by Emergent Vision Technologies Inc
 #               www.emergentvisiontec.com
 #############################################################################

# Basic utility functions for the EVT_Py API

from EVT_Py import EVT_Py
from EVT_Py.EVT_Py import EvtPixelFormat

def set_param_max(cam: EVT_Py.EvtCamera, param: str) -> None:
    max = cam.get_max_param_uint32(param)
    cam.set_param_uint32(param, max)

    print(f"\t{param}: {cam.get_param_uint32(param)}")

def set_param(cam: EVT_Py.EvtCamera, param: str, value) -> None:
    cam.set_param_uint32(param, value)
    print(f"\t{param}: {cam.get_param_uint32(param)}")

def set_param_str(cam: EVT_Py.EvtCamera, param: str, value) -> None:
    cam.set_enum_str(param, value)
    print(f"\t{param}: {cam.get_enum_str(param)}")
    

def set_param_min(cam: EVT_Py.EvtCamera, param: str) -> None:
    max = cam.get_min_param_uint32(param)
    cam.set_param_uint32(param, max)

    print(f"\t{param}: {cam.get_param_uint32(param)}")

def set_param_enum(cam: EVT_Py.EvtCamera, param: str, enum_str: str) -> None:
    cam.set_enum_str(param, enum_str)

    print(f"\t{param}: {cam.get_enum_str(param)}")

def is_8bit(pixFormat: EvtPixelFormat) -> bool:
    if (pixFormat == EvtPixelFormat.GVSP_PIX_BAYBG8 or 
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYRG8 or
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYGB8 or
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYGR8 or
        pixFormat == EvtPixelFormat.GVSP_PIX_MONO8 or 
        pixFormat == EvtPixelFormat.GVSP_PIX_BGR8 or
        pixFormat == EvtPixelFormat.GVSP_PIX_RGB8):
        return True

    return False

def is_bayer(pixFormat: EvtPixelFormat) -> bool:
    if (pixFormat == EvtPixelFormat.GVSP_PIX_BAYBG8 or 
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYRG8 or
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYGB8 or
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYGR8 or 
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYBG10 or 
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYRG10 or
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYGB10 or
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYGR10 or
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYBG10_PACKED or 
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYRG10_PACKED or
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYGB10_PACKED or
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYGR10_PACKED or
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYBG12 or 
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYRG12 or
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYGB12 or
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYGR12 or
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYBG12_PACKED or 
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYRG12_PACKED or
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYGB12_PACKED or
        pixFormat == EvtPixelFormat.GVSP_PIX_BAYGR12_PACKED):
        return True

    return False

def is_mono(pixFormat: EvtPixelFormat) -> bool:
    if (pixFormat == EvtPixelFormat.GVSP_PIX_MONO8 or 
        pixFormat == EvtPixelFormat.GVSP_PIX_MONO10 or
        pixFormat == EvtPixelFormat.GVSP_PIX_MONO10_PACKED or
        pixFormat == EvtPixelFormat.GVSP_PIX_MONO12 or
        pixFormat == EvtPixelFormat.GVSP_PIX_MONO12_PACKED):
        return True

    return False