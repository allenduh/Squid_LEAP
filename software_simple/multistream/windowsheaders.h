#pragma once

/* Include all Windows header files here because WinSock2.h must be placed prior to Windows.h, otherwise Windows.h will include WinSock1.
   All other code should include this file for Windows to keep the order. 
   Also import Ws2_32.lib and  Iphlpapi.lib. */
#include <WinSock2.h>
#include <WS2tcpip.h>
#include <Windows.h>
#include <iphlpapi.h>
#include <io.h>
#include <fcntl.h>
#include <direct.h>
#include <sys/stat.h>
#include <stdio.h>
#include <string.h>
#include <time.h>       /* time_t, struct tm, difftime, time, mktime */
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Iphlpapi.lib")
