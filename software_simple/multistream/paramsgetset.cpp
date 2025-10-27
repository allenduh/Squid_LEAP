#include "common.h"
extern UNSIGNED_INT_PARAM paramUI[PARAMETER_MAX];
extern int countParamUI;
extern SIGNED_INT_PARAM paramI[PARAMETER_MAX];
extern int countParamI;
extern ENUM_PARAM paramEnum[PARAMETER_MAX];
extern int countParamEnum;

extern STRING_PARAMS paramStringsSet;
extern STRING_PARAMS paramStringsGet;

const char* opStrs[] = {"read", "write_min", "write_val", "write_max"};
#define GET_SET_STR(write) write ? "Set" : "Get"
// return false for any error
bool ParamsGetSet(CEmergentCamera *camera)
{
    printf("Paramaters: %s\n", camera->ConnectedDeviceInfo()->currentIp);
    int count = 0;
    EVT_ERROR ret = EVT_SUCCESS;

    // set unsigned int parameters
    for(count = 0; count < countParamUI; count++) {
        UNSIGNED_INT_PARAM* param = &paramUI[count];
        if(param->operType == PARAM_READ) {
            unsigned int min = 0, max = 0xFFFFFFFF;
            ret = EVT_CameraGetUInt32Param(camera, param->name, &param->val);
            ret = EVT_CameraGetUInt32ParamMin(camera, param->name, &min);
            ret = EVT_CameraGetUInt32ParamMax(camera, param->name, &max);
            printf("%s [%s = %u (%u - %u)]: ", opStrs[param->operType], param->name, param->val, min, max);
        } else {
            if(param->operType == PARAM_WRITE_MIN) {
                ret = EVT_CameraGetUInt32ParamMin(camera, param->name, &param->val);
            } else if(param->operType == PARAM_WRITE_MAX) {
                ret = EVT_CameraGetUInt32ParamMax(camera, param->name, &param->val);
            }
            ret = EVT_CameraSetUInt32Param(camera, param->name, param->val);
            printf("%s [%s = %u]: ", opStrs[param->operType], param->name, param->val);
        }

        if(EVT_ERROR_SUCCESS(ret)) printf("\n");
        else {
            printf("failed, ret = %d\n", ret);
            break;
        }
    }
    if(count != countParamUI) return false;

    // set int parameters
    for(count = 0; count < countParamI; count++) {
        SIGNED_INT_PARAM* param = &paramI[count];
        ret = param->opWrite ? EVT_CameraSetInt32Param(camera, param->name, param->val) :
                               EVT_CameraGetInt32Param(camera, param->name, &param->val);
        printf("%s [%s = %d]", GET_SET_STR(param->opWrite), param->name, param->val);
        if(EVT_ERROR_SUCCESS(ret)) printf("\n");
        else {
            printf(" failed, ret = %d\n", ret);
            break;
        }
    }
    if(count != countParamI) return false;


    // set enum parameters
    for(count = 0; count < countParamEnum; count++) {
        ENUM_PARAM* param = &paramEnum[count];
        ret = EVT_CameraSetEnumParam(camera, param->name, param->val);
        printf("Set [%s = %s]: ", param->name, param->val);
        if(EVT_ERROR_SUCCESS(ret)) printf("\n");
        else {
            printf("failed, ret = %d\n", ret);
            break;
        }
    }
    if(count != countParamEnum) return false;

    // set string parameters
    STRING_PARAMS* ps = &paramStringsSet;
    for(count = 0; count < ps->count; count++) {
        STRING_PARAM* param = &ps->params[count];
#ifdef OLD_ESDK
        ret = EVT_CameraSetStringParam(camera, param->name, param->val);
#else        
        ret = EVT_CameraSetStringParam(camera, param->name, param->val, param->length);
#endif        
        if(!EVT_ERROR_SUCCESS(ret)) {
            printf("Set [%s: %s] fail, ret = %d\n", param->name, param->val, ret);
            break;
        }
    }
    if(count != ps->count) return false;

    printf("Getting parameters:\n");
    // get string parameters
    ps = &paramStringsGet;
    for(count = 0; count < ps->count; count++) {
        STRING_PARAM* param = &ps->params[count];
        unsigned long valueSize = 0;
#ifdef OLD_ESDK        
        ret = EVT_CameraGetStringParam(camera, param->name, param->val, STRING_MAX, &valueSize);
#else
        ret = EVT_CameraGetStringParam(camera, param->name, param->val, STRING_MAX, &valueSize, param->length);        
#endif        
        if(EVT_ERROR_SUCCESS(ret)) {
            printf("Got [%s: %.*s] length: %u valueSize: %lu\n", param->name, (int)valueSize, param->val, param->length, valueSize);
        } else {
            printf("Get [%s: %s] fail, ret = %d\n", param->name, param->val, ret);
             break;
        }
    }
    printf("\n");
    if(count != ps->count) return false;

    return true;
}