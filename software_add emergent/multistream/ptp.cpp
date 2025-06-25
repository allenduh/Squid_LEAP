#include <stdio.h>

#include "common.h"

// check if PTP is available
bool PTPAvailable(CEmergentCamera* camera)
{
    if(EVT_CameraSetEnumParam(camera, "PtpMode", "TwoStep") == EVT_SUCCESS) {
        int retry = 20;
        while(retry) {
            char ptpStatus[32];
            if(EVT_CameraGetEnumParam(camera, "PtpStatus", ptpStatus, sizeof(ptpStatus), NULL) == EVT_SUCCESS) {
                //printf("PTP Status: %s\n", ptpStatus);
                if(strcmp(ptpStatus, "Slave") == 0) return true;
            }
            CEvtUtil::SLEEP(100); // 2 seconds to confirm ptp is working
            retry--;
        }
    }
    printf("Fail to enter PTP slave mode.\n");
    return false;
}

bool SetupPTP(CEmergentCamera* camera)
{
    if(EVT_CameraSetEnumParam(camera, "TriggerSource", "Software") == EVT_SUCCESS) {
        if(EVT_CameraSetEnumParam(camera, "AcquisitionMode", "MultiFrame") == EVT_SUCCESS) {
            if(EVT_CameraSetUInt32Param(camera, "AcquisitionFrameCount", 1) == EVT_SUCCESS) {
	            if(EVT_CameraSetEnumParam(camera, "TriggerMode", "On") == EVT_SUCCESS) {
                    // we've set register "PtpMode" to "TwoStep" in PTPAvailable().
                    //EVT_CameraSetEnumParam(camera, "PtpMode", "TwoStep"); 
                    return true;
                }
            }
        }
    }
    printf("Fail to set PTP trigger mode.\n");
    return false;
}

bool DisablePTP(CEmergentCamera* camera)
{
    if(EVT_CameraSetEnumParam(camera, "AcquisitionMode", "Continuous") == EVT_SUCCESS)
    {
        if(EVT_CameraSetEnumParam(camera, "TriggerMode", "Off") == EVT_SUCCESS)
        {
            //EVT_CameraSetEnumParam(&camera, "PtpMode", "TwoStep"); 
            return true;
        }
    }
    return false;
}

#define SECOND_USECOND_FACTOR ((unsigned long long)1000000000)  // factor to convert second and micro-second
/* Start the streaming with PTP. 
   In record, we start the first camera with wait time in seconds and calculate the gatetime. So the other cameras use the same gatetime
   for PTP synchronization.
   1. Detect PTP availability by reading register "PtpStatus" and its value is "Slave".
   2. if PTP is available:
      The first camera: get the camera's current time plus ptpCount to get the gate time, set it to gatetime register. Pass out
                        the gatetime and adjusted ptpCount.
      The following cameras: Simply set gatetime register with the passed-in ptpGatetime. ptpGatetime was returned from the first camera.
   In normal streaming, all cameras act as the first camera with a short-period of ptpCount.

   ptpCount: seconds to count down. passed in from the first camera in order to calculate the gate time and passed it out with ptpGatetime.
             PTP_COUNT_MAX for the following cameras and ptpGatetime should be a valid value to be set to register.
   ptpGatetime: actual ptp start time. Passed out for the first camera if ptpCount != PTP_COUNT_MAX, passed in from the following cameras.
   ptpOffset: value read from register "PtpOffset".
 */
bool StartStreamingPtp(CEmergentCamera* camera, unsigned long long* ptpCount, unsigned long long* ptpGatetime, int* ptpOffset)
{
    if(EVT_CameraGetInt32Param(camera, "PtpOffset", ptpOffset) == EVT_SUCCESS) //get raw offsets.
    {
        unsigned long long ptp_time; // current time
        unsigned long long ptp_time_plus_delta_to_start = PTP_COUNT_MAX;  // time in the future to start streaming
        unsigned int ptp_time_low, ptp_time_high, ptp_time_plus_delta_to_start_low, ptp_time_plus_delta_to_start_high;
                
        if(*ptpCount != PTP_COUNT_MAX)  // The first camera: ptp_time_plus_delta_to_start = currenttime + ptpCount
        {
            // calculate ptp_time_plus_delta_to_start by ptpCount
            if(EVT_CameraExecuteCommand(camera, "GevTimestampControlLatch") == EVT_SUCCESS &&
                EVT_CameraGetUInt32Param(camera, "GevTimestampValueHigh", &ptp_time_high) == EVT_SUCCESS &&
                EVT_CameraGetUInt32Param(camera, "GevTimestampValueLow", &ptp_time_low) == EVT_SUCCESS)
            {
                ptp_time = (((unsigned long long)(ptp_time_high)) << 32) | ((unsigned long long)(ptp_time_low));
                printf("PTP Current Time(s): %llu s, %llu ns\n", ptp_time / SECOND_USECOND_FACTOR, ptp_time);
				if(*ptpCount <= 10) ptp_time_plus_delta_to_start =  ptp_time + (*ptpCount * SECOND_USECOND_FACTOR); // gate time = current time + ptpCount seconds
				else ptp_time_plus_delta_to_start =  ptp_time + *ptpCount; // gate time = current time + ptpCount nanoseconds
                printf("PTP    Gate Time(s): %llu s, %llu ns\n", ptp_time_plus_delta_to_start / SECOND_USECOND_FACTOR, ptp_time_plus_delta_to_start);
                *ptpGatetime = ptp_time_plus_delta_to_start;
            }
        }
        else  // The following cameras: ptp_time_plus_delta_to_start = ptpGatetime
        {
            if(EVT_CameraExecuteCommand(camera, "GevTimestampControlLatch") == EVT_SUCCESS &&
				EVT_CameraGetUInt32Param(camera, "GevTimestampValueHigh", &ptp_time_high) == EVT_SUCCESS &&
                EVT_CameraGetUInt32Param(camera, "GevTimestampValueLow", &ptp_time_low) == EVT_SUCCESS)
            {
                ptp_time = (((unsigned long long)(ptp_time_high)) << 32) | ((unsigned long long)(ptp_time_low));
                printf("PTP Current Time(s): %llu s, %llu ns\n", ptp_time / SECOND_USECOND_FACTOR, ptp_time);
            }

            // just set ptp_time_plus_delta_to_start to ptpGatetime
            ptp_time_plus_delta_to_start = *ptpGatetime;
			printf("Use passed gatetime: %llu s, %llu ns\n", ptp_time_plus_delta_to_start / SECOND_USECOND_FACTOR, ptp_time_plus_delta_to_start);
        }

        // Now we know the gatetime to start, set it to reg PtpAcquisitionGateTimeHigh/Low
        if(ptp_time_plus_delta_to_start != PTP_COUNT_MAX)
        {
            ptp_time_plus_delta_to_start_low  = (unsigned int)(ptp_time_plus_delta_to_start & 0xFFFFFFFF);
            ptp_time_plus_delta_to_start_high = (unsigned int)(ptp_time_plus_delta_to_start >> 32);
            if(EVT_CameraSetUInt32Param(camera, "PtpAcquisitionGateTimeHigh", ptp_time_plus_delta_to_start_high) == EVT_SUCCESS &&
                EVT_CameraSetUInt32Param(camera, "PtpAcquisitionGateTimeLow", ptp_time_plus_delta_to_start_low) == EVT_SUCCESS)
            {
                if(EVT_CameraExecuteCommand(camera, "AcquisitionStart") == EVT_SUCCESS)
                {
                    if(EVT_CameraExecuteCommand(camera, "GevTimestampControlLatch") == EVT_SUCCESS &&
                        EVT_CameraGetUInt32Param(camera, "GevTimestampValueHigh", &ptp_time_high) == EVT_SUCCESS &&
                        EVT_CameraGetUInt32Param(camera, "GevTimestampValueLow", &ptp_time_low) == EVT_SUCCESS)
                    {
                        ptp_time = (((unsigned long long)(ptp_time_high)) << 32) | ((unsigned long long)(ptp_time_low)); 
                        printf("AcquisitionStart tm: %llu s, %llu ns\n", ptp_time  / SECOND_USECOND_FACTOR, ptp_time);
                        *ptpCount = (ptp_time_plus_delta_to_start - ptp_time) / SECOND_USECOND_FACTOR; // convert to second
                        (*ptpCount)++;  // plus 1 second to make sure action start after saving start
                        //printf("++++++++++++++++++++++++ %s %d count down *ptpCountOut %llu S\n", __FUNCTION__, __LINE__, *ptpCount);
                        return true;
                    }
                }
            }
        }
    }
    return false;
}