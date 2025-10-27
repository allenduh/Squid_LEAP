#include <stdio.h>

#include "common.h"

// triggerOpts[3] is 3 options of GPI_Start_Exp_Mode, GPI_Start_Exp_Event, GPI_End_Exp_Mode. empty string for default value.
// for exmple: GPI_4, Rising_Edge, Internal
EVT_ERROR SetupHWTrigger(CEmergentCamera* camera, const char triggerOpts[HW_TRIGGER_OPTS_COUNT][HW_TRIGGER_OPTS_LEN])
{
    EVT_ERROR ret = EVT_SUCCESS;
    // 1. set 4 common registers
    ret = EVT_CameraSetEnumParam(camera, "TriggerSource", "Hardware"); if(ret != EVT_SUCCESS) return ret;
    ret = EVT_CameraSetEnumParam(camera, "AcquisitionMode", "Continuous"); if(ret != EVT_SUCCESS) return ret;
    ret = EVT_CameraSetUInt32Param(camera, "AcquisitionFrameCount", 1); if(ret != EVT_SUCCESS) return ret;
	ret = EVT_CameraSetEnumParam(camera, "TriggerMode", "On"); if(ret != EVT_SUCCESS) return ret;

    // 2. customized trigger method
    if(strlen(triggerOpts[0])) { ret = EVT_CameraSetEnumParam(camera, "GPI_Start_Exp_Mode", triggerOpts[0]); if(ret != EVT_SUCCESS) return ret;}
    if(strlen(triggerOpts[1])) { ret = EVT_CameraSetEnumParam(camera, "GPI_Start_Exp_Event", triggerOpts[1]); if(ret != EVT_SUCCESS) return ret;}
    if(strlen(triggerOpts[2])) { ret = EVT_CameraSetEnumParam(camera, "GPI_End_Exp_Mode", triggerOpts[2]); if(ret != EVT_SUCCESS) return ret;}
    
    return ret;
}

