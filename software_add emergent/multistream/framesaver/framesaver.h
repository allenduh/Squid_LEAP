#pragma once
#include "threadworker.h"
#include <emergentframe.h>

#if defined(__GNUC__)
#define DIR_DELIMITER "/"
#else
#define DIR_DELIMITER "\\"
#endif

using namespace Emergent;

#define SAVED_FILE_EXT "raw"

//#define FILE_WRITE_STATISTICS

#ifdef FILE_WRITE_STATISTICS
#define MAX_STATISTIC_FILE_WRITE 10
struct FILE_WRITE_STATISTICS_INFO
{
    void Reset() { // reset all member except mutex
        for(int i = 0; i < MAX_STATISTIC_FILE_WRITE; i++) timeCopy[i] = 0;
        for(int i = 0; i < MAX_STATISTIC_FILE_WRITE; i++) timeW[i] = 0;
        timeWWorst = countOT = currentState;
    }
    long long timeCopy[MAX_STATISTIC_FILE_WRITE];
    long long timeW[MAX_STATISTIC_FILE_WRITE];
    long long timeWWorst;
    int countOT;
    int currentState;
    CGenericMutex mutex;
};
#endif
#define SAVE_ARRAYS_MAX 64

struct FRAME_SAVE_ENTRY
{
    CEmergentFrame frame;
    unsigned int frameIdTake; // frame id counted in the entire take, 0 based
};

typedef bool (*FileWriteFunc)(const char* file, unsigned int sizeTotal, const void* buffers[], unsigned int sizes[], int countBuffs);

class CFrameSaver : public CThreadWorker
{
public:
    CFrameSaver(const char* name = "CFrameSaver");
    ~CFrameSaver();

    void SetSaveInfo(int cid, int sArrayCount, unsigned int saveArraySize, bool isDirectIO, const char* takeD);
    const char* SaveDir() const { return saveDir; }
    const char* SaveFileName() const { return saveFileName; }    

#ifdef FILE_WRITE_STATISTICS
    void GetFileWriteStatisticsInfo(struct FILE_WRITE_STATISTICS_INFO* f) {fileWriteStatisticsInfo.mutex.Lock(); *f = fileWriteStatisticsInfo; fileWriteStatisticsInfo.mutex.Unlock();}
#endif

private:  // overrides
    bool WorkerThreadStart();
	bool WorkerFunction(void* o); // the worker function sub-class
    void WorkerReset(); // worker's reset, called by Reset, 

private:
    void FreeArrays();

private:
    int camId; // just for debug filter
    char saveDir[256]; // dictory of the file save: saveHomeDir/subDir/camId
    char saveFileName[256];
    int mkDir1000; // id of the 1000 dir, initialzed as -1, so 0 means 0000000 has been made, 1 means 0001000 has been made.
    FileWriteFunc fileWriteFunc;
#ifdef FILE_WRITE_STATISTICS
    struct FILE_WRITE_STATISTICS_INFO fileWriteStatisticsInfo;
#endif
    int saveArrayCount;
    size_t saveArraySize;
    int saveArrayCurr;
    unsigned char* saveArrays[SAVE_ARRAYS_MAX];

public:
    // deciDir: /cap_storage/cap/raid0/E20190805115906953/C006/deci, frameId: 1001
    // form: /cap_storage/cap/raid0/E20190805115906953/C006/deci/0001000/F0001001.img
    static void FrameIdToDeciImageFile(char* fileName, const char* deciDir, unsigned int frameId)
    {
        unsigned int id1000 = frameId / 1000;
        sprintf(fileName, "%s" DIR_DELIMITER "%04d000" DIR_DELIMITER "F%07d.dec", deciDir, id1000, frameId);
    }

    // form image file: HOME/2020_08_18_10_53_40_345/0001000/00001200.
    static void FrameIdToTakeImageFile(char* fileName, const char* dirRecordHome, unsigned int frameId)
    {
        unsigned int id1000 = frameId / 1000;
        sprintf(fileName, "%s" DIR_DELIMITER "%04d000" DIR_DELIMITER "F%07d.%s", dirRecordHome, id1000, frameId, SAVED_FILE_EXT);
    }

}; 