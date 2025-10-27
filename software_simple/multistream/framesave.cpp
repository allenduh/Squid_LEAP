#include "common.h"
#include <evtutil.h>
// Get completed entris from saver to free queque, and return the frames to driver.
static void GetCompletedEntriesFromSaver(STREAM_THREAD_INFO* streamThreadInfo, CFrameSaver* frameSaver)
{
    struct FRAME_SAVE_ENTRY* entriesOut[FRAMES_PER_SAVER];  // entries got out from saver thread, their frames should be returned to driver queue.
    int entriesOutCount = FRAMES_PER_SAVER;
    frameSaver->GetObjectsFromQueueOut((void**)entriesOut, &entriesOutCount);
    if(entriesOutCount) { // return the frames to driver, and put entries back to frameSaveEntriesFreeQueue
        for(int j = 0; j < entriesOutCount; j++) {
            EVT_ERROR ret = EVT_CameraQueueFrame(&streamThreadInfo->camera, &entriesOut[j]->frame);
            if(!EVT_ERROR_SUCCESS(ret)) TIME_PRINT("Queue Frame error: fid %d, ret %d", entriesOut[j]->frame.frame_id, ret);
            streamThreadInfo->frameSaveEntriesFreeQueue[streamThreadInfo->frameSaveEntriesFreeQueueCount] = entriesOut[j];
            streamThreadInfo->frameSaveEntriesFreeQueueCount++;
        }
    }
}

// Get completed entris from saver to free queque, and return the frames to driver.
// Push the new frame to saver if there is free entries and return true, otherwise false.
bool PushFrameToSaver(STREAM_THREAD_INFO* streamThreadInfo, const CEmergentFrame* frame)
{
    for(int i = 0; i < streamThreadInfo->countSaver; i++) GetCompletedEntriesFromSaver(streamThreadInfo, &streamThreadInfo->frameSavers[i]);
    if(streamThreadInfo->frameSaveEntriesFreeQueueCount) {  // if there is free entry, push the save entry to it
        struct FRAME_SAVE_ENTRY* entryToSave = streamThreadInfo->frameSaveEntriesFreeQueue[streamThreadInfo->frameSaveEntriesFreeQueueCount - 1];
        streamThreadInfo->frameSaveEntriesFreeQueueCount--;

        // find the saver by frame id
        int saverId = 0;
        int frameIdSave = streamThreadInfo->frameCount - 1; // frame id, 0 based
        if(streamThreadInfo->countSaver > 1) {
            saverId = frameIdSave % streamThreadInfo->countSaver;
        }
        CFrameSaver* frameSaver = &streamThreadInfo->frameSavers[saverId];

        entryToSave->frameIdTake = frameIdSave;
        entryToSave->frame.UpdateFrame(frame);
        frameSaver->PutObjectToQueueIn(entryToSave);
        return true;
    } else {
        streamThreadInfo->missedSave++;
        //streamThreadInfo->countQueueIn = frameSaver->GetCountQueueIn();
        TIME_PRINT("Miss save: %s fid: %d, GetCountQueueIn:", streamThreadInfo->deviceInfo->currentIp, frame->frame_id);
        for(int i = 0; i < streamThreadInfo->countSaver; i++) printf("%d ", streamThreadInfo->frameSavers[i].GetCountQueueIn());
    }
    return false;
}

// complete all entries in all saver
void CleanSaver(STREAM_THREAD_INFO* streamThreadInfo)
{
    CFrameSaver* saver = streamThreadInfo->frameSavers;
    for(int i = 0; i < streamThreadInfo->countSaver; i++) {
        while(saver->GetCountInTotal() > saver->GetCountOutTotal()) { // just check the in and out are matching
            GetCompletedEntriesFromSaver(streamThreadInfo, saver);
            CEvtUtil::SLEEP(10);
        }
        saver++;
    }
}