//#include "stdafx.h"
#include <Windows.h>

#include <cstdarg>
#include <string>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <stdarg.h>

#include <mutex>
#include <thread>
#include <string>
#include <list>
#include <fstream>
#include <chrono>

extern "C"
{
#include "libavutil/time.h"
}

#include "P2PLog.h"
#define LOG_BUF_SIZE 512

volatile bool endOfLog = false;
class static_
{
public:
    template<class T>
    static T& var()
    {
        static T instance;
        return instance;
    }
private:
    ~static_() {};
};
std::thread * logtoFileThread = nullptr;
std::list<std::string> *logList = nullptr;
void logToFileLoop()
{
    //create log file
    std::ofstream logFileStream;
    time_t rawtime;
    struct tm  timeinfo;
    time(&rawtime);
    localtime_s(&timeinfo,&rawtime);
    char filename[80];
    strftime(filename, 80, "%F-%H-%M-%S.log", &timeinfo);
    printf("create log file %s\n", filename);
    logFileStream.open(filename);

    while (!endOfLog)
    {
        static_::var<std::mutex>().lock();
        while (logList->size() > 0)
        {
            logFileStream << logList->front();
            logList->pop_front();
            //write to file
        }
        static_::var<std::mutex>().unlock();
        logFileStream.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    //delete log buffer
    static_::var<std::mutex>().lock();
    delete logList;
    logList = nullptr;
    static_::var<std::mutex>().unlock();

    //close file
    logFileStream.close();
}

void p2p_log_start()
{
    logList = new std::list<std::string>();
    logtoFileThread = new std::thread(logToFileLoop);
    logtoFileThread->detach();
}

void p2p_log_end()
{
    endOfLog = true;
}

void appendLog(std::string& log)
{
    static_::var<std::mutex>().lock();
    if (logList == nullptr) {
        p2p_log_start();
    }
    logList->push_back(log);
    static_::var<std::mutex>().unlock();
}

int p2p_log_print(int prio, const char *file, int line, const char *fmt, ...)
{
    va_list ap;
    char buf[LOG_BUF_SIZE];

    va_start(ap, fmt);
    vsnprintf(buf, LOG_BUF_SIZE, fmt, ap);
    va_end(ap);

#ifdef LOGTOFILE
    char bufOut[LOG_BUF_SIZE * 2];
    SYSTEMTIME sys;
    GetLocalTime(&sys);
    sprintf_s(bufOut, "%4d/%02d/%02d %02d:%02d:%02d.%03d %24s %4d %s\n", sys.wYear, sys.wMonth, sys.wDay, sys.wHour, sys.wMinute, sys.wSecond, sys.wMilliseconds, file, line, buf);
    appendLog(std::string(bufOut));
#else
    SYSTEMTIME sys;
    GetLocalTime(&sys);
    printf("%4d/%02d/%02d %02d:%02d:%02d.%03d %24s %4d %s\n", sys.wYear, sys.wMonth, sys.wDay, sys.wHour, sys.wMinute, sys.wSecond, sys.wMilliseconds, file, line, buf);
#endif
    return 0;
}