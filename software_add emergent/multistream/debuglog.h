#pragma once
#include <stdio.h>
#include <time.h>

//#define DEBUG_LOG_FILE // always log to file: physical file or stdout: by the way fp_log is opened.

extern FILE* fp_log;  // decleared in main.cpp
//FILE* fp = fopen(FILE_NAME, "a");
//fclose(fp)
#define LOG_TO_FILE(STR,...) \
    {\
        if(fp_log) \
        { \
            if(ftell(fp_log) > 100000) fseek(fp_log, 0, SEEK_SET); \
            time_t t = time(NULL); \
            struct tm tm = *localtime(&t);  \
            fprintf(fp_log, "%d-%02d-%02d %02d:%02d:%02d: ", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec); \
            fprintf(fp_log, STR,##__VA_ARGS__); \
        } \
    }

#if 0
    #define LOG_DEFINE_LOG_FILE FILE* fp_log = NULL;
    #define LOG_OPEN_LOG_FILE(file_name) fp_log = fopen(file_name, "wb"); if(fp_log) setbuf(fp_log, NULL);
    #define LOG_CLOSE_LOG_FILE fclose(fp_log);

    #ifdef DEBUG_LOG_FILE
        #define LOG(STR,...) LOG_TO_FILE(STR, ##__VA_ARGS__) 
        #define INIT_LOG_STRING
        #define SAVE_LOG_STRING
    #else
   		#ifdef DEBUG_LOG_STRING
            static char* log_str = NULL, *log_str0 = NULL;
            #define INIT_LOG_STRING { log_str0 = log_str = new char[100000]; *log_str0 = 0;}
            #define RESET_LOG_STRING { log_str = log_str0; *log_str0 = 0;}
			#define LOG(STR,...) { log_str += sprintf(log_str, STR, ##__VA_ARGS__);}
            #define SAVE_LOG_STRING  { LOG_TO_FILE(log_str0);}
            
            static char* log1_str = NULL, *log1_str0 = NULL;
            #define INIT_LOG_STRING1 { log1_str0 = log1_str = new char[100000]; *log1_str0 = 0;}
            #define RESET_LOG_STRING1 { log1_str = log1_str0; *log1_str0 = 0;}
			#define LOG1(STR,...) { log1_str += sprintf(log1_str, STR, ##__VA_ARGS__);}
            #define SAVE_LOG_STRING1  { LOG_TO_FILE(log1_str0);}
            
		#else
   			#ifdef DEBUG_LOG_CONSOLE
                #define LOG printf
                #define LOG_LINE(FORMAT,...) LOG("+++++++++++ %s %s %d: " FORMAT "\n", __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__) 
                #define INIT_LOG_STRING 
                #define SAVE_LOG_STRING
            #else
                #define LOG(STR,...)
                #define INIT_LOG_STRING 
                #define SAVE_LOG_STRING
            #endif
		#endif
    #endif
#else
    #define LOG_DEFINE_LOG_FILE
    #define LOG_OPEN_LOG_FILE
    #define LOG_CLOSE_LOG_FILE

    #define LOG(STR,...)
    #define LOG_LINE(STR,...)
    #define INIT_LOG_STRING 
    #define SAVE_LOG_STRING    
#endif
