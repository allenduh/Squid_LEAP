#pragma once
#include <queue> 
#include <evtutil.h>

using namespace std;
using namespace Emergent;

class CThreadWorker : public COffThreadMachine
{
public:
    CThreadWorker(const char* name); // name is the thread name
    virtual ~CThreadWorker();

    void Reset();
    int GetMyWork() { return myWork; }  // for statistics
   
    void PutObjectToQueueIn(void* f);
    void GetObjectsFromQueueOut(void** f, int* count);

    int GetCountQueueInSize();
    int GetCountQueueOutSize();
    
    // easy way to get counts without using mutext
    int GetCountQueueIn() const { return countQueueIn; }
    int GetCountQueueOut() const { return countQueueOut; }
    int GetCountInTotal() const { return countInTotal; }  // total count in
    int GetCountOutTotal() const {return countOutTotal; } // total count out
    int GetCountQueueInMax() const { return countQueueInMax; }

    // Microseconds in Linux, Milliseconds in Windows
    void SetInterval(unsigned int i) { 
        interval = i; 
#ifdef _WIN32
        intervalMilliSeconds = interval / 1000;
        if(interval % 1000) intervalMilliSeconds++;
#endif
    }


private: 
	virtual void ThreadRunning(); // overides of COffThreadMachine for worker thread

private:
    void ResetInner();
    void* GetObjectFromQueueIn();
    void PutObjectToQueueOut(void* f);
    
    virtual bool WorkerThreadStart() = 0;
    virtual bool WorkerFunction(void* f) = 0; // the worker function sub-class
    virtual void WorkerReset() = 0; // worker's reset, called by Reset, 

private:
    CGenericMutex mutexQueueIn; // protect mutexQueueIn between threads
    std::queue<void*> queueIn;	
    CGenericMutex mutexQueueOut; // protect queueOut between threads
    std::queue<void*> queueOut;	

    // tracing counts
    int countQueueIn; // size of queue in;
    int countQueueOut; // size of queue in;
    int countInTotal;  // count frames put into saver by grab thread calling PutFrameToQueueToSave()
    int countOutTotal;  // count frames get out of saver  by grab thread calling GetFramesFromQueueToDriver()
    int countQueueInMax; // maximum of queue in

//    int cpu;
    int myWork;
    unsigned int interval; // interval of checking queue in micro seconds.
#ifdef _WIN32
    unsigned int intervalMilliSeconds; // interval in Windows, minimum 1 millisecond sleep in Windows, converted from interval, minimum 1 millisecond.
#endif
};