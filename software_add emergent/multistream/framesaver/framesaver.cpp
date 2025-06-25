#include <stdio.h>
#include <EmergentCameraAPIs.h>
#include <emergentcameradef.h>
#include <emergentgigevisiondef.h>
#include <emergentframe.h>

#define DEBUG_LOG_CONSOLE
#include "../debuglog.h"
#include "framesaver.h"
#include <evtutil.h>

using namespace Emergent;

CFrameSaver::CFrameSaver(const char* name)
:CThreadWorker(name)
{
    saveArrayCount = 0;
    saveArrayCurr = 0;
    saveArraySize = 0;
    memset(saveArrays, 0, sizeof(saveArrays));
    WorkerReset();
}


CFrameSaver::~CFrameSaver()
{
    FreeArrays();
}

void CFrameSaver::FreeArrays()
{
    for(int i = 0; i < SAVE_ARRAYS_MAX; i++) {
        if(saveArrays[i]) {
            delete[] saveArrays[i];
            saveArrays[i] = nullptr;
        }
    }
    saveArrayCount = 0;
    saveArrayCurr = 0;
    saveArraySize = 0;
}

/*
  Reset CFrameSaver by reseting frame buffer. 
  This is to avoid the frame from requeued to NIC drvier when swapped out by SaveFrame() at the beginning of save frame.
*/
void CFrameSaver::WorkerReset()
{
    saveDir[0] = 0;
};

// called once when thread started
bool CFrameSaver::WorkerThreadStart()
{
    for(int i = 0; i < saveArrayCount; i++) {
        saveArrays[i] = new unsigned char[saveArraySize];
    }
    return true;
}

/* 
  Set some info for saving, if sArrayCount > 0, save to arrary, otherwise to files.
*/
void CFrameSaver::SetSaveInfo(int cid, int sArrayCount, unsigned int sArraySize, bool isDirectIO, const char* takeD)
{
    // arrays will be allocated in WorkerThreadStart()
    FreeArrays();
    saveArrayCount = sArrayCount;
    saveArraySize = sArraySize;

    camId = cid;
    if(saveArrayCount <= 0) {
        strncpy(saveDir, takeD, 256);
        fileWriteFunc = isDirectIO ? CEvtUtil::WriteBuffersToFileDirect : CEvtUtil::WriteBuffersToFile;
        mkDir1000 = -1;
     }
#ifdef FILE_WRITE_STATISTICS
    //must keep mutex
    fileWriteStatisticsInfo.Reset();
#endif
}

// make sub-dir by number in 1000 group
static void MakeDirBy1000(char* parentDir, int id1000)
{
    char dir[256];
    sprintf(dir, "%s" DIR_DELIMITER "%04d000", parentDir, id1000);
    CEvtUtil::MkDir(dir);
}

//#define SAVE_TIF
bool CFrameSaver::WorkerFunction(void* o)
{
    struct FRAME_SAVE_ENTRY* entry = (struct FRAME_SAVE_ENTRY*) o;
    const CEmergentFrame* frame = &entry->frame;

    if(saveArrayCount > 0) {  // save array
        unsigned char* buffer = (unsigned char*)saveArrays[saveArrayCurr];
        if(frame->partialBuffers[0].imagePtr) {
            memcpy(buffer, frame->partialBuffers[0].imagePtr, frame->partialBuffers[0].bufferSize);
            if(frame->partialBuffers[1].imagePtr) memcpy(buffer + frame->partialBuffers[0].bufferSize, frame->partialBuffers[1].imagePtr, frame->partialBuffers[1].bufferSize);
        } else {
            memcpy(buffer, frame->imagePtr, frame->bufferSize);
        }
        saveArrayCurr++;
        if(saveArrayCurr == saveArrayCount) saveArrayCurr = 0;
        //CEvtUtil::SLEEP(20);
        return true;
    }

    // make dir for 1000 frames and keep track
    int frameId = entry->frameIdTake;
    int id1000 = frameId / 1000; // 0, 1 for 0 - 999, 1000 - 1999
    if(id1000 > mkDir1000) {
        MakeDirBy1000(saveDir, id1000);
        mkDir1000++;
    }

    //FrameIdToTakeImageFile(fileName, dirTake, 1); // get the file name
    FrameIdToTakeImageFile(saveFileName, saveDir, frameId); // get the file name
    //LOG("++++++++++++++++++++++++ %s %d fileName: %s\n", __FUNCTION__, __LINE__, fileName);
    //unsigned long long t1 = CEvtUtil::GetMillisecond();
//printf("++++++++++++++++++++++++ %s %d \n", __FUNCTION__, __LINE__);            
    const void* buffers[2];
    unsigned int sizes[2];
    int countBuffs;
    if(frame->imagePtr) {  // with wraparound
        buffers[0] = frame->imagePtr;
        sizes[0] = frame->bufferSize;
        sizes[1] = 0;
        countBuffs = 1;
    } else {
        buffers[0] = frame->partialBuffers[0].imagePtr;
        buffers[1] = frame->partialBuffers[1].imagePtr;
        sizes[0] = frame->partialBuffers[0].bufferSize;
        sizes[1] = frame->partialBuffers[1].bufferSize;
        countBuffs = frame->partialBuffers[1].imagePtr ? 2 : 1;
    }
          
    if( (sizes[0] + sizes[1]) != frame->bufferSize) {
        LOG("++++++++++++++++++++++++ %s %d wrong bufferSize %d %d %d\n", __FUNCTION__, __LINE__, sizes[0], sizes[1], frame->bufferSize);
    }

    if( (sizes[0] + sizes[1]) % 512 ) printf("Size not 512 aligned.\n");

    int returnVal = true;
    int retry = 0;
    while(retry < 3) { // try 3 times
        //sizes[0] = 512;
        //printf("++++++++++++++++++++++++ %s %d buffer %p %d %d countBuffs: %d\n", __FUNCTION__, __LINE__, buffers[0], sizes[0], frame->bufferSize, countBuffs);
        returnVal = fileWriteFunc(saveFileName, frame->bufferSize, buffers, sizes, countBuffs);
        if(returnVal) break;
        else printf("fileWriteFunc Error! fileName: %s, retry: %d\n", saveFileName, retry);
        retry++;
    }
    if(retry) printf("IO error in file: %s, retry: %d, returnVal: %d\n", saveFileName, retry, returnVal);


    //unsigned long long timeW = CEvtUtil::GetMillisecond() - t1;
    //if(camId == 0) printf("++++++++++++++++++++++++ %s %d WT: %llu\n", __FUNCTION__, __LINE__, timeW);
    
//    LOG("%llu ", timeW);

#ifdef FILE_WRITE_STATISTICS            
    fileWriteStatisticsInfo.mutex.Lock(); /// update info

    fileWriteStatisticsInfo.timeCopy[fileWriteStatisticsInfo.currentState] = 0;
    fileWriteStatisticsInfo.timeW[fileWriteStatisticsInfo.currentState] = timeW;

    if(timeW >= 15)  {
        if(fileWriteStatisticsInfo.timeWWorst < (long long) timeW) fileWriteStatisticsInfo.timeWWorst = timeW;
        fileWriteStatisticsInfo.countOT++;
        #if 0               
            printf("timeW >= 15: %lld fileName: %s ", timeW, fileName);
            for(int ww = 0; ww < MAX_STATISTIC_FILE_WRITE; ww++) printf("%lld ", fileWriteStatisticsInfo.timeW[ww]);
            printf("\n");
        #endif                
    }

    fileWriteStatisticsInfo.mutex.Unlock();
    fileWriteStatisticsInfo.currentState++;
    if(fileWriteStatisticsInfo.currentState >= MAX_STATISTIC_FILE_WRITE) fileWriteStatisticsInfo.currentState = 0;
#endif
    return ! returnVal;
}
