#pragma once
#ifdef _WIN32
#include "windowsheaders.h"
#include "getopt\getopt.h"

#define THREAD_HANDLE HANDLE
#define THREAD_FUNCTION  DWORD WINAPI

#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
typedef int SOCKET;
#define INVALID_SOCKET -1
#define SOCKET_ERROR   -1
#define closesocket(s) close(s);
typedef void* HANDLE;

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define THREAD_HANDLE pthread_t
#define THREAD_FUNCTION void*

#define SleepMilliSeconds(ms) usleep(ms * 1000)  // sleep ms milli-seconds
#define SleepSeconds(s) sleep(s)   // sleep s seconds
#endif

#include <stdio.h>
#include <string.h>
#include <time.h>       /* time_t, struct tm, difftime, time, mktime */
#include <EmergentCameraAPIs.h>

#include "framesaver/framesaver.h"

using namespace std;
using namespace Emergent;

//#define OLD_ESDK 1

#define IP_MATCHING_BUFF_LEN 32
#define CAMERA_GROUP_MAX 4
#define BUFFERS_MAX 60

#define FRAMES_PER_SAVER 55
// FRAMES_PER_SAVER in queue to save, extra 1 could be held by the saving thread
#define FRAMES_QUEUE_TO_SAVER_MAX (FRAMES_PER_SAVER + 1)
// frames to get data from driver, FRAMES_QUEUE_TO_SAVER_MAX + 1. The extra 1 allows to keep streaming even all frames are in savers queue
#define FRAMES_FREE_TOTAL (FRAMES_QUEUE_TO_SAVER_MAX + 1) 

typedef struct {
    int group;
    int id;
    struct GigEVisionDeviceInfo* deviceInfo;
    CEmergentCamera camera;
    CEmergentFrame frames[BUFFERS_MAX];
    int allocatedCount;
    THREAD_HANDLE threadHandle;  // stream thread
    bool threadOn;
    int cpuStream;
	int frameCount;
    int dropCount;
    unsigned long long timeStampMaxDrift;
    bool noFrame; // true if last state is no frame, to avoid to many no frame print

    // GPU-direct
    int gpuDeviceId;

    // synchronized frame copy
    unsigned char* frameBuffApp;

    // user ring buffer size
    struct EVTStreamAttribute streamAttribute;

    // frame saving
    int cpuSave;
    CFrameSaver* frameSavers; // if cpuSave >= 0, allocate multiple of frameSaver according to saveCamsPerDir and saveDirsPerCam
    int countSaver;
    struct FRAME_SAVE_ENTRY frameSaveEntries[FRAMES_FREE_TOTAL]; // all frame entries, each of it could be used for grabbing, put to saver thread, returned from saver then returned to driver.
    struct FRAME_SAVE_ENTRY* frameSaveEntriesFreeQueue[FRAMES_FREE_TOTAL]; //  free queue FRAME_SAVE_ENTRY to keep track objects of frameSaveEntries which could be in savers or available.
    int frameSaveEntriesFreeQueueCount;  // count of framesFree left in grab thread, not in savers
    int missedSave;
    int countQueueIn;
} STREAM_THREAD_INFO;


#define PARAMETER_MAX 10
#define STRING_MAX 512

typedef enum {
    PARAM_READ = 0,
    PARAM_WRITE_MIN,
    PARAM_WRITE_VAL,
    PARAM_WRITE_MAX
} PARAM_OPER_TYPE;

typedef struct {
    char name[64];
    unsigned int val;
    PARAM_OPER_TYPE operType;
} UNSIGNED_INT_PARAM;

typedef struct {
    char name[64];
    int val;
    bool opWrite; // read or write
} SIGNED_INT_PARAM;

typedef struct {
    char name[64];
    char val[128];
} ENUM_PARAM;

typedef struct {
    char name[64];
    char val[STRING_MAX];
    unsigned int length;
} STRING_PARAM;

typedef struct {
    int count;
    STRING_PARAM params[PARAMETER_MAX];
} STRING_PARAMS;

extern void ParseOpts(int argc, char* argv[]);

// PTP
#define PTP_COUNT_MAX ((unsigned long long)0xFFFFFFFFFFFFFFFF) // seconds to delay/countdown for PTP streaming.
extern bool PTPAvailable(CEmergentCamera* camera);
extern bool SetupPTP(CEmergentCamera* camera);
extern bool DisablePTP(CEmergentCamera* camera);
extern bool StartStreamingPtp(CEmergentCamera* camera, unsigned long long* ptpCount, unsigned long long* ptpGatetime, int* ptpOffset);

//HW trigger
#define HW_TRIGGER_OPTS_COUNT 3
#define HW_TRIGGER_OPTS_LEN 64
extern EVT_ERROR SetupHWTrigger(CEmergentCamera* camera, const char triggerOpts[HW_TRIGGER_OPTS_COUNT][HW_TRIGGER_OPTS_LEN]);

// frame save
extern bool PushFrameToSaver(STREAM_THREAD_INFO* streamThreadInfo, const CEmergentFrame* frame);
extern void CleanSaver(STREAM_THREAD_INFO* streamThreadInfo);

// get/set parameters
extern bool ParamsGetSet(CEmergentCamera *camera);

extern void PrintTime();
#define TIME_PRINT(FORMAT,...) { PrintTime(); printf(FORMAT "\n", ##__VA_ARGS__); }
#define ERROR_PRINT(FORMAT,...) { PrintTime(); printf("Error:" FORMAT "\n", ##__VA_ARGS__); }
