 #############################################################################
 #
 # Project:      EVT_Py
 # Description:  Python Wrapper for eSDK
 #
 # (c) 2024      by Emergent Vision Technologies Inc
 #               www.emergentvisiontec.com
 #############################################################################

 # Loads the FFI for eSDK
 # Note that this could be compiled as a seperate step to reduce initialization overhead

from cffi import FFI

def _cdef_gigevision_device_info(ffibuilder: FFI):
    ffibuilder.cdef("""
        const unsigned int IPV4_BYTES_SIZE = 16;
        const unsigned int NIC_FRIENDLY_NAME_LEN = 128;
        const unsigned int NIC_DESCRIPTION_LEN = 128;
        const unsigned int NIC_ADAPTER_NAME_LEN = 128;

        typedef struct
        {
            unsigned int vendorID; // Linux: it is supported NIC if vendorID matches /sys/class/net/%s/device/vendor
            const char* name; // Windows: it is supported NIC if name matches NetworkInterfaceController.description
            const char* fullName;
        } NicVendorInfo;

        const unsigned int NIC_VENDOR_DATA_SIZE = 1024;

        typedef struct
        {
            char ip4Address[IPV4_BYTES_SIZE];
            char ip4AddressBroadcast[IPV4_BYTES_SIZE];
            char friendlyName[NIC_FRIENDLY_NAME_LEN]; // Windows: Ethernet 3, Linux: enp1s0f0
            char description[NIC_DESCRIPTION_LEN];  //  Windows: Myri-10G PCIe NIC with MVA #5, Linux: enp1s0f0
            char adapterName[NIC_ADAPTER_NAME_LEN];  //  Windows: {0BF475AD-EE60-4638-9160-FC8EBC29AB83}, Linux: enp1s0f0
            unsigned int subnetMask; // subnet mask. Ip(struct in_addr, s_addr) and mask are little endian, 192.168.2.100 is: 0x6402a8c0,  255.255.255.0 is 0x00ffffff
            const struct NicVendorInfo* nicVendorInfo;
            unsigned char vendorData[NIC_VENDOR_DATA_SIZE]; // extra data for some special case, for EVT NIC, it is a struct of EVT_NIC_PROP
        } NetworkInterfaceController;

        typedef struct
        {
            char currentIp[16];
            char macAddress[18];
            char serialNumber[16];
        } LabviewDeviceInfo;

        typedef struct
        {
            int countDevice;
            uint64_t privateData;
        } LabviewDeviceSet;

    """)

def _cdef_evt_system_funcs(ffibuilder: FFI):
    ffibuilder.cdef("""
        typedef void* Camera_Handle;

        typedef struct {
            bool disableHeartbeat;  // disable Heartbeat, default false;
            unsigned long long lineReorderCPUMask; //the bit mask for reorder threads and cores. default 0x0000000F: 4 threads pinned to core 0 - 3; 0: line reorering is disabled.
            int gpuDirectDeviceId;  // not use GPU-direct if < 0; specify GPU Id if >= 0, default -1
        } CameraOpenParams;

        void EVT_CameraOpenParams_Init(CameraOpenParams* params);

        int EVT_ListDevices(LabviewDeviceSet* labviewDeviceSet);
        int EVT_GetDeviceInfo(LabviewDeviceSet* labviewDeviceSet, LabviewDeviceInfo* labviewDeviceInfo, int index);
        void EVT_DeviceInfoSetCleanup(LabviewDeviceSet* labviewDeviceSet);

        int EVT_CameraOpen(Camera_Handle* cameraHandle, LabviewDeviceSet* labviewDeviceSet, int index); 
        int EVT_CameraOpen_Ex(Camera_Handle* cameraHandle, LabviewDeviceSet* labviewDeviceSet, int index, const CameraOpenParams* params);
        int EVT_CameraClose(Camera_Handle cameraHandle); 
    """)

def _cdef_evt_camera_funcs(ffibuilder: FFI):
    ffibuilder.cdef("""
        typedef struct EmergentFrame
        {
            int pixel_type;
            unsigned int size_x;      
            unsigned int size_y;      
            int stride;
            unsigned int offset_x;      
            unsigned int offset_y;      
            unsigned int padding_x;      
            unsigned int padding_y;      
            unsigned int trailer_size_y;
            unsigned short frame_id;
            unsigned char* imagePtr;
            unsigned int bufferSize; 
            unsigned long long timestamp;
            unsigned long long nsecs;
            int convertColor;
            int convertBitDepth;
            unsigned char privateData[256]; // place holder of c++ class: Emergent::CEmergentFrame
        } EmergentFrame;

        typedef void* LINES_REORDER_HANDLE;

        int EVT_CameraOpenStream(Camera_Handle camera);
        int EVT_CameraCloseStream(Camera_Handle camera);

        int EVT_AllocateFrameBuffer(Camera_Handle camera, EmergentFrame* frame, int bufferType);
        int EVT_ReleaseFrameBuffer(Camera_Handle camera, EmergentFrame* frame);
        int EVT_CameraQueueFrame(Camera_Handle camera, EmergentFrame* frame);
        int EVT_CameraGetFrame(Camera_Handle camera, EmergentFrame* frame, int timeout);
        
        int EVT_CameraGetUInt32Param(Camera_Handle camera, const char* name, unsigned int* val);
        int EVT_CameraSetUInt32Param(Camera_Handle camera, const char* name, unsigned int val);
        int EVT_CameraGetUInt32ParamMax(Camera_Handle camera, const char* name, unsigned int* max);
        int EVT_CameraGetUInt32ParamMin(Camera_Handle camera, const char* name, unsigned int* min);

        int EVT_CameraGetStringEnumParam(Camera_Handle camera, const char* name, char* buffer, unsigned long bufferSize, unsigned long* valueSize);
        int EVT_CameraSetEnumParam(Camera_Handle camera, const char* name, const char* buffer);

        int EVT_CameraGetValueEnumParam(Camera_Handle camera, const char* name, unsigned int* value);

        int EVT_CameraExecuteCommand(Camera_Handle camera, const char* name);

        LINES_REORDER_HANDLE EVT_GetLinesReorderHandle(Camera_Handle CamHandle);
        int EVT_FrameConvert(EmergentFrame* frameSrc, EmergentFrame* frameDst, int convertBitDepth, int convertColor, LINES_REORDER_HANDLE handle);
    """)

def load_cffi() -> FFI:
    ffi = FFI()

    _cdef_gigevision_device_info(ffi)

    _cdef_evt_system_funcs(ffi)

    _cdef_evt_camera_funcs(ffi)

    return ffi