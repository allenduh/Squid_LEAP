// CppSample.cpp : Defines the entry point for the console application.
//
/*
Compile: 
g++ multistream.cpp ptp.cpp getopt/getopt.c -o multistream -I$EMERGENT_DIR/eSDK/include/ -L$EMERGENT_DIR/eSDK/lib/  -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision  -pthread
NIC Set up
sudo ifconfig enp1s0f0 down
sudo ifconfig enp1s0f1 down
sleep 10 seconds
sudo ifconfig enp1s0f0 192.168.1.1 netmask 255.255.255.0 mtu 9000 up
sudo ifconfig enp1s0f1 192.168.2.1 netmask 255.255.255.0 mtu 9000 up

run:
clear; sudo ./multistream -w 0 -p 2 -c 16 -t 30 -b 16 -e 48; read -p "Press enter to continue"; 
sudo mlxlink -d /dev/mst/mt4121_pciconf0 -c; sudo mlxlink -d /dev/mst/mt4121_pciconf0.1 -c
C:\multistream\x64\Release\multistream.exe -p 5 -c 16 -t 10 -b 0 -e 12 -i 10  -j 15 -k 35

sudo MELLANOX_RINGBUFF_FACTOR=17 ./out/multistream -p 5 -c 12,4,0,0 -d 0,12,32,33 -f 60,145,60,60 -t 10 -i 1 -oq -w2 -s 1000
sudo MELLANOX_RINGBUFF_FACTOR=17 ./out/multistream -p 5 -c 12,4,0,0 -d 0,12,32,33 -f 60,145,60,60 -t 10 -i 1 -oq -w2 -s 60
sudo MELLANOX_RINGBUFF_FACTOR=17 ./out/multistream -p 0 -c 11,0,0,0 -d 0,12,32,33 -f 60,145,60,60 -t 30 -i 1 -oq -w2 -s 60
devel test:
sudo LD_LIBRARY_PATH=:$EMERGENT_DIR/eSDK/lib:$EMERGENT_DIR/eSDK/genicam//bin/Linux64_x64 ./out/multistream -p 0 -c 1,0,0,0 -d 8,16,32,48 -f 542,60,60,60 -t 30 -i 1 -oq -w2 -s 60
sudo LD_LIBRARY_PATH=:$EMERGENT_DIR/eSDK/lib:$EMERGENT_DIR/eSDK/genicam//bin/Linux64_x64 ./out/multistream -p 0 -c 9,9,9,9 -d 0,16,32,48 -f 60,60,60,60 -t 10 -i 1 -oq -w2 -s 60
On threadripper Pro, must set: MELLANOX_RINGBUFF_FACTOR=17
for HZ cameras: EMERGENT_PKT_SIZE=7972

For 36 cameras:
sudo MELLANOX_RINGBUFF_FACTOR=17 ./out/multistream -p 5 -c 9,9,9,9 -d 0,16,32,48 -f 60,60,60,60 -t 10 -i 1 -oq -w2 -s 60

sudo ./out/multistream -c 4,5,4,5 -t 10 -n 192.168.1@1-4,192.168.1@5-9,192.168.2@0-4,192.168.2@5 -e 6 -d 0,0,0,0
sudo MELLANOX_RINGBUFF_FACTOR=14 EVT_DEBUG_LOG=1 ./out/multistream -c 1 -t 10 -g 0

*/
#include "common.h"
#include "framesaver/framesaver.h"
#include <evtutil.h>

#define DEBUG_LOG_CONSOLE
//#include "debuglog.h"

//#define LOG_TO_FILE(STR,...) { FILE* fp = fopen("e:\\temp\\log.txt", "a"); fprintf(fp, STR,##__VA_ARGS__); fclose(fp); }
//#define LOG_TO_FILE_NAME(NAME, STR, ...) { FILE* fp = fopen(NAME, "a"); fprintf(fp, STR, ##__VA_ARGS__); fclose(fp); }

//Param values
#define AUTOGAIN_PARAM  "AutoGain"
#define EXPOSURE_PARAM "Exposure"
#define CUSTOM_6_PARAM "Custom_6"
#define ACQUISITIONSTART_PARAM "AcquisitionStart"
#define ACQUISITIONSTOP_PARAM  "AcquisitionStop"
#define ACQUISITIONMODE_PARAM "AcquisitionMode"
#define PIXELFORMAT_PARAM "PixelFormat"

// config data
unsigned int camTests[CAMERA_GROUP_MAX] = {0, 0, 0, 0};
char subnetPrefixes[CAMERA_GROUP_MAX][IP_MATCHING_BUFF_LEN]={"192.168.1", "192.168.2", "192.168.3","192.168.4"};
unsigned int subnetStart[CAMERA_GROUP_MAX] = {0, 0, 0, 0};
unsigned int subnetEnd[CAMERA_GROUP_MAX] = {254, 254, 254, 254};
int cpuStartStream[CAMERA_GROUP_MAX] = {-1, -1, -1, -1};
int cpuStartSave[CAMERA_GROUP_MAX] = {-1, -1, -1, -1};  // default no saving
int saveArrayCount = -1; // <=0: save to files, > 0 save to array of this size
char saveHomeDir[128] = "";
int saveCamsPerDir = 1;
int saveDirsPerCam = 0;
int saveDirsTotal = 100;  // total save dirs, wrap up for saveDirsPerCam, default 100: not wrap up
unsigned int coresGrpStream = 1;  // cores number group of streaming
unsigned int coresGrpSave = 1; // cores number group of saving
unsigned int frameRates[CAMERA_GROUP_MAX] = {0, 0, 0, 0};
char hwTriggerOpts[HW_TRIGGER_OPTS_COUNT][HW_TRIGGER_OPTS_LEN] = {"", "", ""};
int gpuDeviceIds[CAMERA_GROUP_MAX] = {-1, -1, -1, -1};
int testTime = 60;
int trigerModeValue = 0; // triger mode and PTP value
int getFrameTimeoutMs = 30;
int nonPtpSeconds = 0;
bool wrapAround = false;
char streamPortsDir[128] = "";
bool showNoFrame = false;
int testCount = 1;
bool quitOnError = false;
bool discoveryInBackground = false;  // perform discovery in background, the main thread
bool discoveryBroadcast = false;
bool copyFrame = false;
bool heartbeat = false;
bool pressEnterToStart = false;
char cameraIPDiscovery[IPV4_BYTES_SIZE] = "";
bool runtimeStatSlient = false; // don't print runtime statistics 
bool lazyMode = false; // don't get frame, just sleep.
bool viewSettingsOnly = false;
size_t userMemSize = 0; // user allocated memory size(MB) for ring buffer
char localXml[128] = "";

unsigned long long timeStampBaseDelta = 0xFFFFFFFFFFFFFFFF;
unsigned long long timeStampMaxDriftCheck = 0xFFFFFFFFFFFFFFFF;

UNSIGNED_INT_PARAM paramUI[PARAMETER_MAX];
int countParamUI = 0;
SIGNED_INT_PARAM paramI[PARAMETER_MAX];
int countParamI = 0;
ENUM_PARAM paramEnum[PARAMETER_MAX];
int countParamEnum = 0;

STRING_PARAMS paramStringsSet = {0};
STRING_PARAMS paramStringsGet = {0};

THREAD_FUNCTION Test_Streaming(void* ptr);

int Test_Streaming2Threads(STREAM_THREAD_INFO* streamThreadInfos, int bufSize);

void PrintTime()
{
    time_t t = time(NULL); 
    struct tm tm = *localtime(&t); 
    printf("%d-%02d-%02d %02d:%02d:%02d: ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
}

void GetTimeStr(char* str)
{
    time_t t = time(NULL); 
    struct tm tm = *localtime(&t); 
    sprintf(str, "%d-%02d-%02d_%02d_%02d_%02d", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
}

// get the group id by camera ip and subnetPrefixes[subnetStart - 254], return 0 - 4 if matches, -1 otherwise
int GroupId(struct GigEVisionDeviceInfo* deviceInfo)
{
    //int debug = 0;
    //if(strcmp(deviceInfo->currentIp, "192.168.1.5") == 0) debug = 1;
    for(int i = 0; i < CAMERA_GROUP_MAX; i++) {
        char* subnet = subnetPrefixes[i];
        for(unsigned int j = subnetStart[i]; j <= subnetEnd[i]; j++) {
            char ipMatching[IP_MATCHING_BUFF_LEN];
            sprintf(ipMatching, "%s.%u", subnet, j);
            //if(debug) printf("ipMatching: %s\n", ipMatching);
            size_t len = strlen(ipMatching);
            if(strncmp(deviceInfo->currentIp, ipMatching, len) == 0) return i;
        }
    }
    return -1;
}

bool PushFrameToSaver(STREAM_THREAD_INFO* streamThreadInfo, const CEmergentFrame* frame);

int main(int argc, char* argv[])
{
    ParseOpts(argc, argv);
    if(pressEnterToStart) {
        printf("Press Enter to Start\n");
        getchar();
    }
   
    unsigned int bufSize = 50;
    struct GigEVisionDeviceInfo* deviceInfos = new struct GigEVisionDeviceInfo[bufSize];

    PrintTime(); printf("start...\n");

    unsigned int actualCount = 0;
    struct ListDevicesSettings settings;
    if(discoveryBroadcast) settings.broadcast = 1;
#ifndef OLD_ESDK	
    if(cameraIPDiscovery[0]) settings.cameraIP = cameraIPDiscovery;
#endif	
    EVT_ListDevices(deviceInfos, &bufSize, &actualCount, &settings);
    if(! bufSize) {
        printf("no camera found.\n\n");
        delete[] deviceInfos;
        return 0;
    }
#if 0
    printf("%d camera(s) found.\n", bufSize);
    for(unsigned int i = 0; i < bufSize; i++)
    {
        printf("Camera %d: on %s: %s %s.\n", i, deviceInfos[i].nic.ip4Address, deviceInfos[i].manufacturerName, deviceInfos[i].currentIp);
    }
#endif

    int iCopied = 0;

    // In windows and Ubuntu, the camera can be discovered even it is not on the same subnet of the NIC port. 
    // Filter them out, otherwise Mellanox streaming won't work because it open stream with the wrong host address.
    printf("\n%d camera(s) discovered:\n", bufSize);
    for(unsigned int i = 0; i < bufSize; i++) {
        printf("Camera %02d: on %s: %s SN: %s.\n", i, deviceInfos[i].nic.ip4Address, deviceInfos[i].currentIp, deviceInfos[i].serialNumber);
        if(strncmp(deviceInfos[i].nic.ip4Address, deviceInfos[i].currentIp, 8) == 0) {
            deviceInfos[iCopied] = deviceInfos[i];
            iCopied++;
        }
    }
    bufSize = iCopied;

    STREAM_THREAD_INFO* streamThreadInfos = new STREAM_THREAD_INFO[bufSize];
    int cpuStreamCur[CAMERA_GROUP_MAX];  
    int cpuSaveCur[CAMERA_GROUP_MAX];
    memcpy(cpuStreamCur, cpuStartStream, sizeof(cpuStartStream));
    memcpy(cpuSaveCur, cpuStartSave, sizeof(cpuStartSave));

    // set up grouping and other configs.
    printf("\nGrouping:\n");
    iCopied = 0;
    for(unsigned int i = 0; i < bufSize; i++) { // set streamThreadInfos according to group
        int group = GroupId(&deviceInfos[i]);
        printf("Camera %02d: %s, group: %d\n", i, deviceInfos[i].currentIp, group);
        if(group >= 0) {
            //printf("%s in group %d\n", deviceInfos[i].currentIp, group);
            if(camTests[group]) { // group not end yet
                streamThreadInfos[iCopied].id = iCopied;
                streamThreadInfos[iCopied].group = group;
                streamThreadInfos[iCopied].deviceInfo = &deviceInfos[i];
                streamThreadInfos[iCopied].cpuStream = cpuStreamCur[group];
                streamThreadInfos[iCopied].cpuSave = cpuSaveCur[group];
                streamThreadInfos[iCopied].gpuDeviceId = gpuDeviceIds[group];

                // streaming core: go to next core or back to start core
                int cpuStart = cpuStartStream[group];
                if(cpuStart >= 0) {
                    int cpuEnd= cpuStart + coresGrpStream - 1;
                    cpuStreamCur[group]++;
                    if(cpuStreamCur[group] > cpuEnd ) cpuStreamCur[group] = cpuStart; // wrap around to the first core
                }
                    
                // saving core: go to next core or back to start core
                cpuStart = cpuStartSave[group];
                if(cpuStart >= 0) {
                    int cpuEnd= cpuStart + coresGrpSave - 1;
                    cpuSaveCur[group]++;
                    if(cpuSaveCur[group] > cpuEnd) cpuSaveCur[group] = cpuStart; // wrap around to the start core
                }
                iCopied++;
                camTests[group]--;
            }
        } else printf("%s not in any group\n", deviceInfos[i].currentIp);
    }

    printf("\n%d camera(s) will be tested:\n", iCopied);
    for(int i = 0; i < iCopied; i++) {
        STREAM_THREAD_INFO* stInfo = &streamThreadInfos[i];
        struct GigEVisionDeviceInfo* dInfo = stInfo->deviceInfo;
        printf("Camera %02d: %s, group: %d, stream cpu: %d, save cpu: %d, dir: %s, GPU: %d\n", i, dInfo->currentIp, stInfo->group, stInfo->cpuStream, stInfo->cpuSave, saveHomeDir, stInfo->gpuDeviceId);
    }

    if(!viewSettingsOnly) {
        for(int i = 0; i < testCount; i++) {
            printf("\nTest: %d starts\n", i);
            int ret = Test_Streaming2Threads(streamThreadInfos, iCopied);
            printf("Test : %d completed ============================================= %d\n", i, i);
            if(quitOnError && ret != 0) break;
            CEvtUtil::SLEEP(2000);
        }
    }

    delete[] streamThreadInfos;

    delete[] deviceInfos;
    return 0;
}


void ConnectionCallBackHeartbeat(void* callerData, void* arg)
{
    struct GigEVisionDeviceInfo* deviceInfo = (GigEVisionDeviceInfo*) callerData;
    ERROR_PRINT("ConnectionCallBackHeartbeat ip: %s\n", deviceInfo->currentIp);
}

//#define  CPU_TO_USE
THREAD_FUNCTION Test_Streaming(void* ptr)
{
    STREAM_THREAD_INFO* streamThreadInfo = (STREAM_THREAD_INFO*) ptr;
    if(streamThreadInfo->cpuStream >= 0) CEvtUtil::SetAffinity(streamThreadInfo->cpuStream);
    CEmergentFrame frame;
    CEmergentCamera* camera = &streamThreadInfo->camera;
    
    unsigned short fid = 0;
    int printStart = 0;
    unsigned long long timeStamp = 0;
    while(streamThreadInfo->threadOn) {
        if(lazyMode) {
            CEvtUtil::SLEEP(500);
            continue;
        }
        EVT_ERROR ret = EVT_CameraGetFrame(camera, &frame, getFrameTimeoutMs);
        if(ret == EVT_SUCCESS) {
            //printf("cam: %d fid: %d minArrTS: %llu maxArrTS: %llu\n", streamThreadInfo->id, frame.frame_id, frame.minArrTS, frame.maxArrTS);
            //printf("cam: %d fid: %d bufferSize: %d\n", streamThreadInfo->id, frame.frame_id, frame.bufferSize);
            //printf("%d offsetX: %d offsetY: %d\n", frame.frame_id, frame.offset_x, frame.offset_y);
            //printf("imagePtr: %p partialBuffers: %p %p\n", frame.imagePtr, frame.partialBuffers[0].imagePtr, frame.partialBuffers[1].imagePtr);
            if(streamThreadInfo->noFrame) {
                streamThreadInfo->noFrame = false;
                TIME_PRINT("frame back: %s last: %d\n", streamThreadInfo->deviceInfo->currentIp, streamThreadInfo->frameCount);
            }

            if(printStart < 1) {// check if the first frame is 1 because we noticed Mellanox driver could drop the first frame.
                if(frame.frame_id != 1) {
				    PrintTime();
				    printf("printStart: %d: %15s, first fid: %d timestamp: %llu\n", printStart, streamThreadInfo->deviceInfo->currentIp, frame.frame_id, frame.timestamp);
                }
                printStart++;
            }
			streamThreadInfo->frameCount++;

            // timestamp
            //printf("timeStampCheck: %lld group: %d  id %d\n", timeStampCheck, streamThreadInfo->group, streamThreadInfo->id);
            if(timeStampBaseDelta != 0xFFFFFFFFFFFFFFFF) {
                if(timeStamp) {
                    if(timeStampBaseDelta == 0) {  // just print delta for reference
                        if(streamThreadInfo->id == 0) printf("fid: %d timestamp: %llu, delta: %llu\n", frame.frame_id, frame.timestamp, frame.timestamp - timeStamp);
                    } else {  // check delta
                        unsigned long long delta = frame.timestamp - timeStamp;
                        unsigned long long drift = delta > timeStampBaseDelta ? delta - timeStampBaseDelta : timeStampBaseDelta - delta;
                        if(drift > streamThreadInfo->timeStampMaxDrift) streamThreadInfo->timeStampMaxDrift = drift;
                        if(timeStampMaxDriftCheck && (drift > timeStampMaxDriftCheck)) printf("Frame skipped: %15s, fid: %d timestamp: %llu - %llu, delta: %llu, drift: %llu\n", streamThreadInfo->deviceInfo->currentIp, frame.frame_id, timeStamp, frame.timestamp, delta, drift);
                    }
                }
                timeStamp = frame.timestamp;
            }

            // drop frame detect
            if(fid != 65535 && frame.frame_id != 1) {
                unsigned short frameIdShouldBe = fid + 1;
                if(frame.frame_id != frameIdShouldBe) {
                    PrintTime();
                    unsigned short dropCount = frame.frame_id - frameIdShouldBe;
                    printf("%s Drop fid: %d should be: %d\n", streamThreadInfo->deviceInfo->currentIp, frame.frame_id, frameIdShouldBe);
                    streamThreadInfo->dropCount += dropCount;
                }
            }
            fid = frame.frame_id;
#if 0
            unsigned int ptp_time_high = 0, ptp_time_low = 0;
            if(EVT_CameraExecuteCommand(camera, "GevTimestampControlLatch") == EVT_SUCCESS &&
                EVT_CameraGetUInt32Param(camera, "GevTimestampValueHigh", &ptp_time_high) == EVT_SUCCESS &&
                EVT_CameraGetUInt32Param(camera, "GevTimestampValueLow", &ptp_time_low) == EVT_SUCCESS) {
                unsigned long long ptp_time = (((unsigned long long)(ptp_time_high)) << 32) | ((unsigned long long)(ptp_time_low));
                if(fp) {
                    fprintf(fp, "%d: %lld\n", fid, ptp_time);
                    fflush(fp);
                }
            }
#endif
            // copy frame
            if(streamThreadInfo->frameBuffApp) {
                if(frame.partialBuffers[0].imagePtr) {
                    memcpy(streamThreadInfo->frameBuffApp, frame.partialBuffers[0].imagePtr, frame.partialBuffers[0].bufferSize);
                    if(frame.partialBuffers[1].imagePtr) memcpy(streamThreadInfo->frameBuffApp + frame.partialBuffers[0].bufferSize, frame.partialBuffers[1].imagePtr, frame.partialBuffers[1].bufferSize);
                } else {
                    memcpy(streamThreadInfo->frameBuffApp, frame.imagePtr, frame.bufferSize);
                }
            }

#if 0
            FILE* fp = fopen("c:\\temp\\aaa.raw", "wb");
            fwrite(frame.imagePtr, sizeof(char), frame.bufferSize, fp);
            fclose(fp);
#endif
            // save frame
            bool framePushedToSave = false;
            if(streamThreadInfo->cpuSave >= 0) {
                if(PushFrameToSaver(streamThreadInfo, &frame)) framePushedToSave = true;
            }

            if(!framePushedToSave) EVT_CameraQueueFrame(camera, &frame);
        } else { // error print: timeout only opted, always print others 
            if(ret == EVT_ERROR_AGAIN) {
                if(showNoFrame) {
                    TIME_PRINT("no frame: %d, %s last: %d\n", ret, streamThreadInfo->deviceInfo->currentIp, streamThreadInfo->frameCount);
                    streamThreadInfo->noFrame = true;
                }
            } else TIME_PRINT("GetFrame error: %d\n", ret);
        }
        // don't sleep for performance. But sleep is ok for some cases.
        //usleep(500);
    }
    // wait until all entried in saver have been completed and retune frame to driver.
    if(streamThreadInfo->cpuSave >= 0) {
        CleanSaver(streamThreadInfo);
    }
    return NULL;
}

// return 0 if succeeds
int Test_Streaming2Threads(STREAM_THREAD_INFO* streamThreadInfos, int bufSize)
{
    bool shouldOpenStream = true;
    if(*streamPortsDir) { // if streamPortsDir is set, set it to MELLANOX_STREAM_PORT_DIR
        CEvtUtil::SetEnv("MELLANOX_STREAM_PORT_DIR", streamPortsDir, 1);
        shouldOpenStream = false;
    }

    // get time str to store the frames
    char timeStr[64];
    GetTimeStr(timeStr);

	int result = 0;
    int iOpen = 0;
    for(iOpen = 0; iOpen < bufSize; iOpen++) {
        STREAM_THREAD_INFO* streamThreadInfo = &streamThreadInfos[iOpen];
        struct GigEVisionDeviceInfo* deviceInfo = streamThreadInfo->deviceInfo;
        streamThreadInfo->threadHandle = 0;
        streamThreadInfo->threadOn = false;
        streamThreadInfo->dropCount = 0;
		streamThreadInfo->frameCount = 0;

        // set affinity to allocate memory in the current MUNA
        if(streamThreadInfo->cpuStream >= 0) CEvtUtil::SetAffinity(streamThreadInfo->cpuStream);

        CEmergentCamera *camera = &streamThreadInfo->camera;
        if(heartbeat) {
            camera->disableHeartbeat = false;
            camera->SetConnectionCallBackFunc(ConnectionCallBackHeartbeat, deviceInfo);
        }
        int gpuDeviceId = streamThreadInfo->gpuDeviceId;
        if(gpuDeviceId >= 0) camera->gpuDirectDeviceId = gpuDeviceId;
		if(!wrapAround) camera->disableWrapAroundMemcpy = true;

        //printf("iOpen: %d opening: %s camera: %p\n", iOpen, deviceInfo->currentIp, camera);
        EVT_ERROR ret = EVT_CameraOpen(camera, deviceInfo, localXml[0] ? localXml : nullptr);
        if(!EVT_ERROR_SUCCESS(ret)) {
            printf("cam: %s open fail ret: %d\n", deviceInfo->currentIp, ret);
            break;
        }

        // set framerate
        int group = streamThreadInfo->group;
        unsigned int frameRate = frameRates[group];
        if(frameRate) ret = EVT_CameraSetUInt32Param(camera, "FrameRate", frameRate);

        // set/get parameters
        if(!ParamsGetSet(camera)) {
            EVT_CameraClose(camera);
            break;
        }

        unsigned int width = 0, height = 0, imageSize = 0;
        char pixelFomat[64];
        ret = EVT_CameraGetUInt32Param(camera, "Width", &width);
        ret = EVT_CameraGetUInt32Param(camera, "Height", &height);
        ret = EVT_CameraGetUInt32Param(camera, "FrameRate", &frameRate);
        ret = EVT_CameraGetEnumParam(camera, "PixelFormat", pixelFomat, 64, NULL);
        if(copyFrame || saveArrayCount > 0) { // get PayloadSize if we need to copy frames
            ret = EVT_CameraGetUInt32Param(camera, "PayloadSize", &imageSize);
            if(!imageSize) imageSize = width * height * 3;
        }
        
        unsigned long long bw = (unsigned long long)width * (unsigned long long)height * (unsigned long long)frameRate * 8;  //bits
        bw /= 1024 * 1024; //mb
        float bwf = (float)bw;
        bwf /= 1024; //gb

        printf("ip: %s, size: %d x %d, %d fps, %s, stream cpu: %d, save cpu: %d BW: %.2f Gb/s\n", streamThreadInfo->deviceInfo->currentIp, width, height, frameRate, pixelFomat, streamThreadInfo->cpuStream, streamThreadInfo->cpuSave, bwf);
        
        // user allocated ring buffer
        unsigned char* userMem = nullptr;
        if(userMemSize) {
            printf("User ring buffer size: %zu(MB)\n", userMemSize);
            userMemSize *= 1024 * 1024; // convert to MB
            userMem = (unsigned char*)CEvtUtil::AllocateAlignedLock(userMemSize, CEvtUtil::GetCacheLineSize());
        }
        streamThreadInfo->streamAttribute.ringBufferPtr = userMem;
        streamThreadInfo->streamAttribute.ringBufferSize = userMemSize;

        ret = EVT_CameraOpenStream(camera, &streamThreadInfo->streamAttribute);
        if(ret == EVT_SUCCESS) {
            if(shouldOpenStream) {
                // 1. start saving threads;
                int cpuSave = streamThreadInfo->cpuSave; 
                CFrameSaver* frameSaver = NULL;
                int countSaver = 0;
                if(cpuSave >= 0) {  // not -1, should initialize data and start saving threads
                    // 1. Setup Save entries
                    for(int i = 0; i < FRAMES_FREE_TOTAL; i++) {
                        struct FRAME_SAVE_ENTRY* entry = &streamThreadInfo->frameSaveEntries[i];
                        entry->frame.SetBufferHandle(NULL, 0); 
                        entry->frameIdTake = 0;
                        streamThreadInfo->frameSaveEntriesFreeQueue[i] = entry;
                    }
                    streamThreadInfo->frameSaveEntriesFreeQueueCount = FRAMES_FREE_TOTAL;

                    // 2. Setup counters
                    // mkdir dir/0/00, dir/0/01 or dir/0/00, dir/1/00
                    int camId = streamThreadInfo->id;
                    int subDir = -1;
                    if(saveArrayCount > 0) { // save to arrary, we just need 1 saver per camera
                        countSaver = 1;
                    } else {
                        if(saveCamsPerDir > 0) { // multi-cam to 1 SSD, we just need 1 saver per camera
                            countSaver = 1;
                            subDir = camId / saveCamsPerDir; // fixed dir: dir/0/
                        } else { // otherwise, 1 save per dir
                            countSaver = saveDirsPerCam; 
                            subDir = camId * saveDirsPerCam; // starting from: cam0: dir/0, cam1: dir/4
                        }
                    }

                    subDir = subDir % saveDirsTotal;
                    //if(subDir >= saveDirsTotal) subDir = 0; // dir wrap around
                    //printf("subDir %d camId %d saveDirsPerCam: %d\n", subDir , camId , saveDirsPerCam);

                    // 3. Setup Savers
                    frameSaver = new CFrameSaver[countSaver];
                    CFrameSaver* saver = frameSaver;
                    for(int i = 0; i < countSaver; i++) {
                        int cpuThread = cpuSave; // + i;
                        saver->Reset();
                        saver->SetCPU(cpuThread);
                        saver->SetInterval(4000); // 1/60 = 16 ms, fwrite 12 ms, ok to check frame every 4 ms to reduce CPU usage

                        char saveDir[256] = "";
                        char saveDirTime[256] = "";
                        if(saveArrayCount <= 0) { // save to saveHomeDir/subDir/camId
                            sprintf(saveDirTime, "%s" DIR_DELIMITER "%d" DIR_DELIMITER "%s", saveHomeDir, subDir, timeStr);
                            CEvtUtil::CheckMkDir(saveDirTime);

                            sprintf(saveDir, "%s" DIR_DELIMITER "%d" DIR_DELIMITER "%s" DIR_DELIMITER "%02d", saveHomeDir, subDir, timeStr, camId);
                            // saveHomeDir/subDir should have existed, just make saveHomeDir/subDir/Time/camId
                            if(!CEvtUtil::CheckMkDir(saveDir)) {
                                ERROR_PRINT("Fail to create dir: %s", saveDir);
                                break;
                            }
                        }

                        saver->SetSaveInfo(iOpen, saveArrayCount, imageSize, true, saveDir);
                        char tName[32];
                        sprintf(tName, "saver_cam%d_ssd%d", iOpen, subDir);
                        printf("Saving dir %s, thread: %s, cpu: %d\n", saveDir, tName, cpuThread);
                        saver->Start(tName);
                        saver++;
                        subDir++;  //cam 0: 0,1,2,3, cam 1: 4,5,6,7, in case of saveCamsPerDir i == 0.
                    }
                }
                streamThreadInfo->missedSave = 0;
                streamThreadInfo->countQueueIn = 0;
                streamThreadInfo->frameSavers = frameSaver;
                streamThreadInfo->countSaver = countSaver;

                // 2. allocate frames
                CEmergentFrame* frames = streamThreadInfo->frames;
                int allocatedCount = 0;
                for(allocatedCount = 0; allocatedCount < BUFFERS_MAX; allocatedCount++) {
                    frames[allocatedCount].size_x = width;
                    frames[allocatedCount].size_y = height;
                    frames[allocatedCount].pixel_type = GVSP_PIX_BAYGR12; //GVSP_PIX_BAYGR8; //GVSP_PIX_RGB8; //GVSP_PIX_BAYGR12_PACKED; 

                    if(EVT_AllocateFrameBuffer(camera, &frames[allocatedCount], EVT_FRAME_BUFFER_ZERO_COPY) == EVT_SUCCESS) {
                        if(EVT_CameraQueueFrame(camera, &frames[allocatedCount]) != EVT_SUCCESS) {
                            printf("EVT_CameraQueueFrame fail...\n");
                            break;
                        }
                    } else {
                        printf("EVT_AllocateFrameBuffer fail...\n");
                        break;
                    }
                }
                streamThreadInfo->allocatedCount = allocatedCount;


                // 3. start streaming threads
                streamThreadInfo->threadOn = true;
#ifdef _WIN32
                streamThreadInfo->threadHandle = CreateThread(NULL, 0, Test_Streaming, (void*)&streamThreadInfos[iOpen], 0, NULL);
                SetThreadPriority(streamThreadInfo->threadHandle, THREAD_PRIORITY_TIME_CRITICAL);
#else
                pthread_create(&streamThreadInfo->threadHandle, NULL, Test_Streaming, (void*)&streamThreadInfos[iOpen]);
#endif

                // 4. allocate frame buffer in app if need to copy frame
                streamThreadInfo->frameBuffApp = copyFrame? new unsigned char[imageSize] : NULL;
            }
        } else {
            printf("Fail to start streaming cam: %s ret: %d\n", deviceInfo->currentIp, ret);
            EVT_CameraClose(camera);
            break;
        }
        CEvtUtil::SLEEP(1);
        CEvtUtil::SetAffinity(0);
    }

    bool streamingReady = false;
    EVT_ERROR ret = EVT_SUCCESS;
    // Trigger all cameras to stream, and keep them to streaming for some time if all were opened successfully.
    if(iOpen == bufSize) {
        if(trigerModeValue == 0) {// no trigger, normal streaming
            for(int i = 0; i < bufSize; i++) DisablePTP(&streamThreadInfos[i].camera);
            if(pressEnterToStart) {
                printf("Press Enter to Start\n");
                getchar();
            }

            TIME_PRINT("sleep %d seconds before starting acquisition.\n",nonPtpSeconds);
            CEvtUtil::SLEEP(nonPtpSeconds * 1000); // wait for 2 seconds
            TIME_PRINT("Start streaming \n");
            int i = 0;
            for(; i < bufSize; i++) {
                //PrintTime(); printf("++++++++++++++++++++++++ %s %d EVT_CameraExecuteCommand start: %s\n", __FUNCTION__, __LINE__, streamThreadInfos[i].deviceInfo->currentIp);
                ret = EVT_CameraExecuteCommand(&streamThreadInfos[i].camera, ACQUISITIONSTART_PARAM);
                if(ret != EVT_SUCCESS) {
                    printf("++++++++++++++++++++++++ %s %d EVT_CameraExecuteCommand fail. ret: %d, %s\n", __FUNCTION__, __LINE__, ret, streamThreadInfos[i].deviceInfo->currentIp);
                    break;
                }
                CEvtUtil::SLEEP(1);
            }
            if(i == bufSize) streamingReady = true;
        } else if(trigerModeValue == -1) {
            /*HW trigger. Set trigger mode and options.
              To make an trigger source: connect GPO_0 and GPI_4, and set the following
	            EVT_CameraSetEnumParam(&camera, "GPO_0_Mode", "Test_Generator");
	            EVT_CameraSetUInt32Param(&camera, "TG_Frame_Time", 16666); //micro second, 60fps: 
	            EVT_CameraSetUInt32Param(&camera, "TG_High_Time", 1000);   //1000us

            */
            int i;
            for(i = 0; i < bufSize; i++) DisablePTP(&streamThreadInfos[i].camera);
            for(i = 0; i < bufSize; i++) {
                ret = SetupHWTrigger(&streamThreadInfos[i].camera, hwTriggerOpts);
                if(ret != EVT_SUCCESS) {
                    PrintTime(); printf("++++++++++++++++++++++++ %s %d SetupHWTrigger %d, %s\n", __FUNCTION__, __LINE__, ret, streamThreadInfos[i].deviceInfo->currentIp);
                    break;
                }
            }
            if(i == bufSize) { // HW trigger ok, ACQUISITIONSTART_PARAM
                for(i = 0; i < bufSize; i++) {
                    ret = EVT_CameraExecuteCommand(&streamThreadInfos[i].camera, ACQUISITIONSTART_PARAM);
                    if(ret != EVT_SUCCESS) {
                        PrintTime(); printf("++++++++++++++++++++++++ %s %d EVT_CameraExecuteCommand fail. ret: %d, %s\n", __FUNCTION__, __LINE__, ret, streamThreadInfos[i].deviceInfo->currentIp);
                        break;
                    }
                }
            }
            if(i == bufSize) streamingReady = true;
        } else { // PTP
            int ptpCamCount = 0;
            for(ptpCamCount = 0; ptpCamCount < bufSize; ptpCamCount++) {
                if(! PTPAvailable(&streamThreadInfos[ptpCamCount].camera)) break;
                if(! SetupPTP(&streamThreadInfos[ptpCamCount].camera)) break;
            }

            if(ptpCamCount == bufSize) {
                //SleepMilliSeconds(1000);  // delay 1 second to allow camera sync up with PTP server
                unsigned long long ptpCount = trigerModeValue;
                unsigned long long ptpGatetime;
                int ptpOffset;
                int i = 0;
                for(; i < bufSize; i++) {
                    struct GigEVisionDeviceInfo* deviceInfo = streamThreadInfos[i].deviceInfo;
				    TIME_PRINT("Starting ptp: %s\n",  deviceInfo->currentIp);
                    // first time, ptpCount == ptpSeconds, pass out ptpGatetime.
                    // rest: ptpCount == PTP_COUNT_MAX, pass in ptpGatetime.
                    if(! StartStreamingPtp(&streamThreadInfos[i].camera, &ptpCount, &ptpGatetime, &ptpOffset)) {
                       printf("Error: %s %s %d StartStreamingPtp fail: %s\n", __FILE__, __FUNCTION__, __LINE__, deviceInfo->currentIp);
                       break;
                    }
					ptpCount = PTP_COUNT_MAX;
                }
                if(i == bufSize) streamingReady = true;
            } else printf("Error: %s %s %d Ptp setup fail: %s\n", __FILE__, __FUNCTION__, __LINE__, streamThreadInfos[ptpCamCount].deviceInfo->currentIp);
        }

        // now, let it stream for a while
        if(streamingReady) {
            TIME_PRINT("Stream for %d seconds...\n", testTime);
            for(int i = 0; i < testTime; i++) {
                CEvtUtil::SLEEP(1000); // sleep 1 second
                if((i + 1) % 2) continue;
                //printf("\n");
                if(!runtimeStatSlient) TIME_PRINT("%d/%d seconds %d%%", i, testTime, i * 100 / testTime);
                int dropTotal = 0;
                int missedSaveTotal = 0;
                for(int j = 0; j < iOpen; j++) {
                    STREAM_THREAD_INFO* streamThreadInfo = &streamThreadInfos[j];
                    dropTotal += streamThreadInfo->dropCount;
                    missedSaveTotal += streamThreadInfo->missedSave;
                    if(!runtimeStatSlient) {
                        printf("[%02d: f:%d d:%d m: %d] ", j, streamThreadInfo->frameCount, streamThreadInfo->dropCount, streamThreadInfo->missedSave);
                        if (((j + 1) % 9) == 0) printf("\n");
                    }
                }
                if(!runtimeStatSlient) printf("Drop Total: %d\n", dropTotal);

                // do a background discovery
                if(discoveryInBackground) {
                    struct GigEVisionDeviceInfo deviceInfos[1];
                    unsigned int s = 1;
                    EVT_ListDevices(deviceInfos, &s, NULL, NULL);
                    if(!runtimeStatSlient) printf("%u camera found.\n", s);
                }

                if(quitOnError) { // check for any error
                    if(dropTotal || missedSaveTotal) {
                        result = -1;
                        break;
                    }
                }
            }
        }
    } else printf("Fail to open camera, or set up parameters, streaming etc..\n");

    // Streaming ends, stop the cameras' acquisition and terminate streaming threads
    for(int i = 0; i < iOpen; i++) {  //Tell camera to stop acquisition and quit streaming thread
        streamThreadInfos[i].threadOn = false;
        EVT_CameraExecuteCommand(&streamThreadInfos[i].camera, ACQUISITIONSTOP_PARAM);
    }

    if(shouldOpenStream) { // stop streaming thread, and save thread
        STREAM_THREAD_INFO* strInfo = streamThreadInfos;
        for(int i = 0; i < iOpen; i++) {
#ifdef _WIN32
            WaitForSingleObject(strInfo->threadHandle, INFINITE);
#else
            pthread_join(strInfo->threadHandle, NULL);
#endif
            // stop saver threads
            for(int j = 0; j < strInfo->countSaver; j++) {
                if(strInfo->frameSavers[j].IsMachineOn()) strInfo->frameSavers[j].Stop();
            }
            strInfo++;
        }
    }

    // close streams and cameras, print summaries
    printf("Summaries:\n");
    int dropTotal = 0;
    int frameTotal = 0;
    int missedSaveTotal = 0;
    for(int i = 0; i < iOpen; i++) {
        STREAM_THREAD_INFO* streamThreadInfo = &streamThreadInfos[i];
        if(shouldOpenStream) {
            // release allocated buffers
            for(int j = 0;  j < streamThreadInfo->allocatedCount; j++) {
                EVT_ReleaseFrameBuffer(&streamThreadInfo->camera, &streamThreadInfo->frames[j]);
            }
        }
        //Host side tear down for stream.
        EVT_CameraCloseStream(&streamThreadInfo->camera);
        if(streamThreadInfo->streamAttribute.ringBufferPtr) CEvtUtil::FreeAligned(streamThreadInfo->streamAttribute.ringBufferPtr); // delete user ring buffer

        //Close the camera
        EVT_CameraClose(&streamThreadInfo->camera);

        // print data
        printf("Cam %s frameCount: %d, dropCount: %d, missedSave: %d", streamThreadInfo->deviceInfo->currentIp, streamThreadInfo->frameCount, streamThreadInfo->dropCount, streamThreadInfo->missedSave);
        if(timeStampBaseDelta && timeStampBaseDelta != 0xFFFFFFFFFFFFFFFF) printf(" ts_max_drift: %llu", streamThreadInfo->timeStampMaxDrift);
        printf("\n");

        frameTotal += streamThreadInfo->frameCount;
        dropTotal += streamThreadInfo->dropCount;
        missedSaveTotal += streamThreadInfo->missedSave;

        if(streamThreadInfo->frameBuffApp) delete streamThreadInfo->frameBuffApp;
    }
    printf("All (%d) cameras Total:%d,  Drop: %d, missedSave: %d\n", iOpen, frameTotal, dropTotal, missedSaveTotal);

    return result;
}

//sudo RIVERMAX_LOG_LEVEL=6 EVT_DEBUG=0xff LD_LIBRARY_PATH=:$EMERGENT_DIR/eSDK/lib:$EMERGENT_DIR/eSDK/genicam//bin/Linux64_x64 ./tools/multistream/out/multistream -p 0 -c 0,1,9,9 -d 0,16,32,48 -f 0,10,60,60 -t 10 -i 1 -oq -w2 -s 60  -u 192.168.2.3
