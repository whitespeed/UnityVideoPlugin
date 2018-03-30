//========= Copyright 2015-2018, HTC Corporation. All rights reserved. ===========

#include "Logger.h"
#include <memory>
#include <string>
#include <consoleapi.h>
#pragma warning(disable:4996)
Logger* Logger::_instance;
UnityLog Logger::_unity = NULL;
Logger::~Logger()
{
	fclose(file);
	file = nullptr;
}
Logger::Logger() {
	fclose(stdout);
	AllocConsole();
	freopen_s(&file, "CONOUT$", "wb", stdout);
	//file = freopen("NativeLog.txt", "a", stdout);
}

Logger* Logger::instance() {
	if (!_instance) {
		_instance = new Logger();
	}
	return _instance;
}

void Logger::log(const char* str, ...) {
	va_list args;
	va_start(args, str);
	char msg[500];
	size_t size = vsprintf(msg,str, args);
	if (_unity != NULL)
	{
		_unity(msg);
	}
	else
	{
		printf(msg);
	}
	va_end(args);

	fflush(stdout);
}