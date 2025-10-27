#include "common.h"
#include <evtutil.h>

#define DEBUG_LOG_CONSOLE
#include "debuglog.h"

// config data
extern unsigned int camTests[CAMERA_GROUP_MAX];
extern char subnetPrefixes[CAMERA_GROUP_MAX][IP_MATCHING_BUFF_LEN];
extern unsigned int subnetStart[CAMERA_GROUP_MAX];
extern unsigned int subnetEnd[CAMERA_GROUP_MAX];
extern int cpuStartStream[CAMERA_GROUP_MAX];
extern int cpuStartSave[CAMERA_GROUP_MAX];
extern int saveArrayCount;
extern char saveHomeDir[];
extern int saveCamsPerDir;
extern int saveDirsPerCam;
extern int saveDirsTotal;
extern unsigned int coresGrpStream;
extern unsigned int coresGrpSave;
extern unsigned int frameRates[CAMERA_GROUP_MAX];
extern char hwTriggerOpts[HW_TRIGGER_OPTS_COUNT][HW_TRIGGER_OPTS_LEN];
extern int gpuDeviceIds[CAMERA_GROUP_MAX];
extern int testTime;
extern int trigerModeValue;
extern int getFrameTimeoutMs;
extern int nonPtpSeconds;
extern bool wrapAround;
extern char streamPortsDir[];
extern bool showNoFrame;
extern int testCount;
extern bool quitOnError;
extern bool discoveryInBackground;
extern bool discoveryBroadcast;
extern bool copyFrame;
extern bool pressEnterToStart;
extern bool heartbeat;
extern char cameraIPDiscovery[IPV4_BYTES_SIZE];
extern bool runtimeStatSlient;
extern bool lazyMode;
extern bool viewSettingsOnly;
extern size_t userMemSize;
extern char localXml[128];

extern unsigned long long timeStampBaseDelta;
extern unsigned long long timeStampMaxDriftCheck;

extern UNSIGNED_INT_PARAM paramUI[PARAMETER_MAX];
extern int countParamUI;
extern SIGNED_INT_PARAM paramI[PARAMETER_MAX];
extern int countParamI;
extern ENUM_PARAM paramEnum[PARAMETER_MAX];
extern int countParamEnum;

extern STRING_PARAMS paramStringsSet;
extern STRING_PARAMS paramStringsGet;

extern const char* opStrs[];

#define ERROR_EXIT(STR,...) { printf(STR "\n", ##__VA_ARGS__);  exit(-1);}

static void Usage()
{
    printf("multistream [option [parameter]]\n"
        "Options:\n"
        "Grouping:\n"
        "-c count1,count2,count3,count4: camera count of group 1 to 4.\n"
        "-n subnet@ip-ip,subnet@ip-ip,subnet@ip-ip,subnet@ip-ip,subnet@ip-ip: subnet prefix and starting IP of group 1 to 4.\n"
        "   For example, 192.168.66,192.168.67@3-10,192.167@100,193 will group cameras as:\n"
        "   group 1: 192.168.66.*\n"
        "   group 2: 192.168.67.3 - 192.168.67.10\n"
        "   group 3: 192.167.100.* - 192.167.254.*\n"
        "   group 4: 193.*.*.*\n"
        "   Default grouping is 192.168.1.*, 192.168.2.*, 192.168.3.*, 192.168.4.*.\n"

        "\nAffinities:\n"
        "-d cpu1,cpu2,cpu3,cpu4: stream threads of a group are set affinity 1-increasingly from this number. \n"
        "   CPU is 0 based. Default -1: not setting affinity.\n"
        "-e coreCountStream,coreCountSave: number of cores for streaming and saving. The CPU assignment of a group's streaming/save threads wrap around at this number. Default 1.\n"
		
        "\nSave frames:\n"
        "-v cpu1,cpu2,cpu3,cpu4@arrayCount@dir@camsPerDir@dirsPerCam@dirCount.\n"
            "   cpu[1-4]: save thread starting cpu of group 1 to 4, default -1: no synchronized operations.\n"
            "   arrayCount: array count to save the frames to. if arrayCount <= 0, save to files specified by dir@camsPerDir@dirsPerCam.\n"
            "   dir: saving home dir. Sub-dir 0 - N should have been created under this directory. Typically 1 sub-dir is mapped to 1 SSD. \n"
            "   camsPerDir: save multiple cameras to 1 dir. This is used when camera band width is less than of 1 SSD.\n"
            "       For instance of 2, cam 0 and 1 are saved at dir/0/00, dir/0/01, cam 2 and 3 are saved to dir/1/02 and dir/1/03. Default 1.\n"
            "       If camera bandwidth is greater than 1 SSD, 1 camera must be distributed to multiple SSDs. Then set camsPerDir to <= 0. The \n"
            "       app will use the next parameter \"dirsPerCam\".\n"
            "   dirsPerCam: save 1 camera to multiple dirs.\n"
            "       For instance of 4, cam 0 will be saved to dir/0/00, dir/1/00, dir/2/00, dir/3/00; cam 1 will be saved to dir/4/01, dir/5/01, dir/6/01, dir/7/01.\n"
			"   dirCount: total save directories. Wrap up to the first one when reaching the end. Default 100.\n"
        
        "\nCamera parameters:\n"
        "-f fps1,fps2,fps3,fps4: frame rate of group 1 to 4. 0 for not set.\n"
        "-j string: set/get a camera parameter. Format: <type>,<name>[,value][,length]. \n"
        "   u,Width,4096            : set Width (type of unsigned int) to 4096.\n"
        "   i,SensTemp,12045        : set SensTemp (type of signed int) to 12045.\n"
        "   e,PixelFormat,BayerGB10 : set Pixel format (type of enum) to BayerGB10.\n"
        "   s,DeviceUserName,cam1 : set Device User Name(type of string) as cam1. The data of max length of the node will be transmitted.\n"
        "   s,DeviceUserName,cam1234,5 : set Device User Name(type of string) with length(5) as cam12. Only the data of specified length will be transmitted.\n"
        "   If value is \"max\" or \"min\", set the value to accordingly. If it is omitted, the operation is to get.\n"

        "\nGPU-direct:\n"
        "-g gpu1,gpu2,gpu3,gpu4 : GPU device IDs of group 1 - 4. Default -1, no GPU-Direct.\n"

        "\nStreaming options:\n"
        "-p number: trigger mode. -1: HW, 0: trigger mode off(normal streaming); >0: PTP countdown in seconds.\n"
        "-h strings delimited with ','. HW trigger mode options: GPI_4,Rising_Edge,Internal.\n"
        "-a number: The memory size in MB that user allocates for ring buffer.\n"
        "-m base_delta,max_drift: print/check timestamp.\n"
        "   If base_delta == 0, print timestamp and delta of the first camera for reference;\n"
        "   If base_delta != 0 && max_drift == 0, get the maximum drift;\n"
        "   If base_delta != 0 && max_drift != 0, check frame skipping by abs(timestamp_delta - base_delta) > max_drift.\n"

        "\nMiscellaneous:\n"
        "-s number: get frame timeout, milliseconds. Default 30.\n"
        "-t number: test duration in seconds. Default 60.\n"
        "-w number: non-ptp wait time in seconds. Default 0.\n"
        "-i number: test loops. Default 1.\n"
        "-r string of dir: the directory of the files where streaming ports are stored, for communicating with media_reciever. If set, the app won't start Rivermax streaming to receive packets.\n"
        "-u IP address: the IP address of the camera to discover. The app will send unicast discovery packets.\n"
        "-x string: the path of a local Xml to open the camera with. Default NULL, using the xml from camera.\n"
        "-o string of letters: binary options.\n"
        "   v: view test settings only, grouping, cpu etc.. Discover cameras but won't open them;\n"
        "   b: broadcast to discover cameras; p: press enter to start; h: heartbeat;\n"
        "   d: discover in background; w: wrap around; l: lazy mode, don't get frame;\n"
        "   c: copy frame data to app; s: silent runtime statistics; \n"
        "   n: show error code of no frame; q: quit on error or drop frame.\n"
        );
}

void ParseOpts(int argc, char* argv[])
{
    if(argc <= 1) { Usage(); exit(0); }

    int c;
    char strTemp[256]; // temp string for parsing
    char delim[] = ",";
    char strs[10][128]; // temp string setions in an arguement, for calling DelimitedStringToStrings.
    int intTemp = 0;
    char* ptrTemp = nullptr;

    while ((c = getopt (argc, argv, "h:c:a:n:d:v:e:f:p:s:w:t:g:i:j:o:r:u:x:m:")) != -1) {
        switch (c) {
            case 'c':  // camera counts
                strcpy(strTemp, optarg);
                CEvtUtil::DelimitedStringToInts(strTemp, delim, (int*)camTests, CAMERA_GROUP_MAX);
                break;

            case 'a':  // MB of user allocated ringbuffer
                userMemSize = atoi(optarg);
                break;

            case 'n': // subnet grouping
                //"-n subnet@ip,subnet@ip,subnet@ip,subnet@ip,subnet@ip: subnet prefix and starting IP of group 1 to 4.\n"
                strcpy(strTemp, optarg);
                memset(strs, 0, sizeof(strs));
                intTemp = CEvtUtil::DelimitedStringToStrings(strTemp, (char*)",", (char*)strs, 10, 128);
                intTemp = min(intTemp, CAMERA_GROUP_MAX);
                for(int i = 0; i < intTemp; i++) {
                    char* ptr = strs[i]; // subnet@ip
                    if(*ptr) {  // if not empty
                        while(*ptr && *ptr != '@') ptr++;
                        *ptr = 0;
                        strncpy(subnetPrefixes[i], strs[i], IP_MATCHING_BUFF_LEN);
                        ptr++; // should be number after @ or \0
                        if(*ptr) {
                            char* ptr1 = ptr;
                            while(*ptr1 && *ptr1 != '-') ptr1++;
                            *ptr1 = 0;
                            subnetStart[i] = atoi(ptr);
                            ptr1++;
                            if(*ptr1) subnetEnd[i] = atoi(ptr1);
                        }
                    }
                }
                break;

            case 'd':  // starting cpus for streaming
                strcpy(strTemp, optarg);
                CEvtUtil::DelimitedStringToInts(strTemp, delim, cpuStartStream, CAMERA_GROUP_MAX);
                break;

            case 'v':  // starting cpus for saving
                strcpy(strTemp, optarg);
                memset(strs, 0, sizeof(strs));
                intTemp = CEvtUtil::DelimitedStringToStrings(strTemp, (char*)"@", (char*)strs, 10, 128);
                //cpu1,cpu2,cpu3,cpu4@arrayCount@dir@camsPerDir@dirsPerCam, must be 2 for copy, 3 or 4 for saving
                //    0                1         2     3          4
                if(intTemp < 2) {
                    printf("parameter -v is invalid.\n");
                    exit(-1);
                }
                CEvtUtil::DelimitedStringToInts(strs[0], delim, cpuStartSave, CAMERA_GROUP_MAX);
                saveArrayCount = atoi(strs[1]);
                if(saveArrayCount > 0) {
                    if(saveArrayCount > SAVE_ARRAYS_MAX) {
                        printf("parameter -v is invalid.\n");
                        exit(-1);
                    }
                } else {  // save to files
                    if(intTemp < 3) {
                        printf("parameter -v is invalid.\n");
                        exit(-1);
                    }
                    strcpy(saveHomeDir, strs[2]);
                    saveCamsPerDir  = atoi(strs[3]);
                    if(saveCamsPerDir <= 0) { // we should look at saveDirsPerCam
                        if(intTemp < 5) {
                            printf("parameter -v is invalid.\n");
                            exit(-1);
                        }
                        saveDirsPerCam = atoi(strs[4]);

                        if(intTemp >= 6) {  // now get total save dirs
                            saveDirsTotal = atoi(strs[5]);
                        }
                    }
                }
                break;

            case 'e':  // cores per group of streaming and saving
                strcpy(strTemp, optarg);
                memset(strs, 0, sizeof(strs));
                intTemp = CEvtUtil::DelimitedStringToStrings(strTemp, (char*)",", (char*)strs, 10, 128);
                if(intTemp >= 1) coresGrpStream = atoi(strs[0]);
                if(intTemp >= 2) coresGrpSave = atoi(strs[1]);
                break;

            case 'f':  // frame rates
                strcpy(strTemp, optarg);
                CEvtUtil::DelimitedStringToInts(strTemp, delim, (int*)frameRates, CAMERA_GROUP_MAX);
                break;

            case 'p': // trigger mode value
                trigerModeValue = atoi(optarg);
                break;

            case 'h': 
                strcpy(strTemp, optarg);
                CEvtUtil::DelimitedStringToStrings(strTemp, delim, (char*)hwTriggerOpts, HW_TRIGGER_OPTS_COUNT, HW_TRIGGER_OPTS_LEN);
                break;

            case 's': // get Frame Timeout ms
                getFrameTimeoutMs = atoi(optarg);
                break;
            
            case 'w': // non-PTP count down
                nonPtpSeconds = atoi(optarg);
                break;                

            case 't': // test time in seconds
                testTime = atoi(optarg);
                break;

            case 'g': // GPU device IDs
                strcpy(strTemp, optarg);
                CEvtUtil::DelimitedStringToInts(strTemp, delim, (int*)gpuDeviceIds, CAMERA_GROUP_MAX);
                break;

            case 'i': // test loops
                testCount = atoi(optarg);
                break;

            case 'j':
                if(optarg[1] == ',') {
                    if(optarg[0] == 'u') { // unsigned type: u,Width,4096
                        if(countParamUI == PARAMETER_MAX) ERROR_EXIT("Too many unsigned int parameters.");

                        UNSIGNED_INT_PARAM* p = &paramUI[countParamUI];
                        optarg += 2; // skip "u,"
                        strcpy(strTemp, optarg);
                        memset(strs, 0, sizeof(strs));
                        int ret = CEvtUtil::DelimitedStringToStrings(strTemp, delim, (char*)strs, 10, 128);
                        if(ret < 1) ERROR_EXIT("Wrong format of unsigned int parameters");

                        if(ret > 0) strncpy(p->name, strs[0], 64);
                        p->operType = PARAM_READ;
                        if(ret > 1) {
                            if(!strcmp(strs[1], "min")) { p->operType = PARAM_WRITE_MIN; }
                            else {
                                if(!strcmp(strs[1], "max")) { p->operType = PARAM_WRITE_MAX; }
                                else {
                                    if(sscanf(strs[1], "%u", &p->val) != 1) { ERROR_EXIT("Wrong format of unsigned int parameters"); }
                                    else p->operType = PARAM_WRITE_VAL;
                                }
                            }
                        }
                        countParamUI++;
                    } else if(optarg[0] == 'i') { // signed in type: i,SensTemp,12045
                        if(countParamI == PARAMETER_MAX) ERROR_EXIT("Too many int parameters");

                        SIGNED_INT_PARAM* p = &paramI[countParamI];
                        memset(p, 0, sizeof(SIGNED_INT_PARAM));
                        optarg += 2; // skip "i,"
                        strcpy(strTemp, optarg);
                        memset(strs, 0, sizeof(strs));
                        int ret = CEvtUtil::DelimitedStringToStrings(strTemp, delim, (char*)strs, 10, 128);
                        if(ret < 1) ERROR_EXIT("Wrong format of signed int parameters.");

                        if(ret > 0) strncpy(p->name, strs[0], 64);
                        if(ret > 1) {
                            if(sscanf(strs[1], "%d", &p->val) != 1) ERROR_EXIT("Wrong format of signed int parameters");
                            p->opWrite = true;
                        } else p->opWrite = false;
                        countParamI++;
                    } else if(optarg[0] == 'e') { // enum type
                        if(countParamEnum == PARAMETER_MAX) {
                            printf("Too many enum parameters.\n");
                            exit(-1);
                        }
                        ENUM_PARAM* p = &paramEnum[countParamEnum];
                        optarg += 2; // skip "e,"
                        int ret = sscanf(optarg, "%[^,],%s", p->name, p->val);
                        if(ret != 2) {
                            printf("Wrong format of enum parameters.\n");
                            exit(-1);
                        }
                        countParamEnum++;
                    } else if(optarg[0] == 's') { // string type
                        optarg += 2; // skip "s,"
                        char* name = nullptr;
                        char* val = nullptr;
                        unsigned int length = 0;

                        strcpy(strTemp, optarg);
                        memset(strs, 0, sizeof(strs));
                        int ret = CEvtUtil::DelimitedStringToStrings(strTemp, delim, (char*)strs, 10, 128);
                        if(ret > 0) name = strs[0];
                        if(ret > 1) val = strs[1];
                        if(ret > 2) length = atoi(strs[2]);

                        if(!name) { // 
                            printf("Name of string parameters is missing.\n");
                            exit(-1);
                        }
                        if(length > STRING_MAX) {
                            printf("%u exceeds the maximum size %u.\n", length, STRING_MAX);
                            exit(-1);
                        }

                        STRING_PARAMS* ps = nullptr;
                        if(val && val[0]) { // set
                            ps = &paramStringsSet;
                        } else { //get
                            ps = &paramStringsGet;
                        }
                        int count = ps->count;
                        if(count == PARAMETER_MAX) {
                            printf("Too many string parameters.\n");
                            exit(-1);
                        }

                        STRING_PARAM* p = &ps->params[count];
                        memset(p, 0, sizeof(*p));
                        strncpy(p->name, name, 64);
                        if(val) strncpy(p->val, val, STRING_MAX);
                        p->length = length;
                        ps->count++;
                    }
                }
                break;

            case 'r': 
                strcpy(streamPortsDir, optarg);
                break;

            case 'u':
                strncpy(cameraIPDiscovery, optarg, sizeof(cameraIPDiscovery));
                break;

            case 'x':
                strncpy(localXml, optarg, sizeof(localXml));
                break;

            case 'm':
                strcpy(strTemp, optarg);
                memset(strs, 0, sizeof(strs));
                intTemp = CEvtUtil::DelimitedStringToStrings(strTemp, delim, (char*)strs, 10, 128);
                if(intTemp < 1) ERROR_EXIT("Wrong parameter of -m.");
                if(sscanf(strs[0], "%lld", &timeStampBaseDelta) != 1) ERROR_EXIT("Wrong parameter of -m.");
                if(timeStampBaseDelta != 0) {
                    if(intTemp < 2) ERROR_EXIT("Wrong parameter of -m.");
                    if(sscanf(strs[1], "%lld", &timeStampMaxDriftCheck) != 1) ERROR_EXIT("Wrong parameter of -m.");
                }
                break;

            case 'o': // options
                ptrTemp = optarg;
                while(*ptrTemp) {
                    char c = *ptrTemp;
                    if(c == 'b') discoveryBroadcast = true;
                    else if(c == 'd') discoveryInBackground = true;
                    else if(c == 'h') heartbeat = true;
                    else if(c == 'c') copyFrame = true;
                    else if(c == 'w') wrapAround = true;
                    else if(c == 'q') quitOnError = true;
                    else if(c == 'n') showNoFrame = true;
                    else if(c == 'p') pressEnterToStart = true;
                    else if(c == 's') runtimeStatSlient = true;
                    else if(c == 'l') lazyMode = true;
                    else if(c == 'v') viewSettingsOnly = true;
                    ptrTemp++;
                }
                break;
        }
    }

    // print all options
    printf("Parameters:\n");
    for(int i = 0; i < CAMERA_GROUP_MAX; i++)
    printf("cam group %d: subnet: %s@%u-%u, count %02d, fps: %03d, straming cpu: %02d, saving cpu: %02d, GPU: %d\n", i, subnetPrefixes[i], subnetStart[i], subnetEnd[i], camTests[i], frameRates[i], cpuStartStream[i], cpuStartSave[i], gpuDeviceIds[i]);
    printf("coresGrpStream: %d\n", coresGrpStream);
    printf("coresGrpSave: %d\n", coresGrpSave);
    printf("saveArrayCount: %d\n", saveArrayCount);
    printf("saveHomeDir: %s\n", saveHomeDir);
    printf("saveCamsPerDir: %d\n", saveCamsPerDir);
    printf("saveDirsPerCam: %d\n", saveDirsPerCam);
    printf("saveDirsTotal: %d\n", saveDirsTotal);
    printf("discoveryInBackground: %d\n", discoveryInBackground);
    printf("heartbeat: %d\n", heartbeat);
    printf("copyFrame: %d\n", copyFrame);
	printf("wrapAround : %d\n", wrapAround);
    printf("trigerModeValue: %d\n", trigerModeValue);
    printf("nonPtpSeconds: %d\n", nonPtpSeconds);
    printf("HW trigger options: %s, %s, %s\n", hwTriggerOpts[0], hwTriggerOpts[1], hwTriggerOpts[2]);
    printf("userMemSize: %zu\n", userMemSize);

    printf("countParamUI: %d, params: ", countParamUI);
    //for(int i = 0; i < countParamUI; i++) printf("[%s: %u, %s]\t", paramUI[i].name, paramUI[i].val, opStrs[paramUI[i].operType]);

    printf("\ncountParamEnum: %d, params: ", countParamEnum);
    for(int i = 0; i < countParamEnum; i++) printf("[%s: %s]\t", paramEnum[i].name, paramEnum[i].val);
    
    STRING_PARAMS* ps = &paramStringsSet;
    printf("\nSet %d strings, params: ", ps->count);
    for(int i = 0; i < ps->count; i++) printf("[%s: %s, %d]\t", ps->params[i].name, ps->params[i].val, ps->params[i].length);
    ps = &paramStringsGet;
    printf("\nGet %d strings, params: ", ps->count);
    for(int i = 0; i < ps->count; i++) printf("[%s: %s, %d]\t", ps->params[i].name, ps->params[i].val, ps->params[i].length);

    printf("streamPortsDir: %s\n", streamPortsDir);
    printf("getFrameTimeoutMs: %d\n", getFrameTimeoutMs);
    printf("testTime: %d\n", testTime);
    printf("testCount: %d\n", testCount);
	printf("quitOnError: %d\n", quitOnError);
    printf("showNoFrame: %d\n", showNoFrame);
    printf("cameraIPDiscovery: %s\n", cameraIPDiscovery);
    printf("runtimeStatSlient: %d\n", runtimeStatSlient);
    printf("lazyMode: %d\n", lazyMode);
    printf("viewSettingsOnly: %d\n", viewSettingsOnly);
    printf("localXml: %s\n", localXml);
    printf("timeStampBaseDelta: %lld timeStampMaxDriftCheck: %lld\n", timeStampBaseDelta, timeStampMaxDriftCheck);
    
    printf("End of Parameters\n\n");
//    exit(-1);
 }