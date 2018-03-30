#include "FPSCounter.h"

FPSCounter::FPSCounter()
{
}

FPSCounter::~FPSCounter()
{
}

float FPSCounter::getFps()
{
	return fps;
}

bool FPSCounter::NewFrame()
{
	int64_t curtime = av_gettime() / 1000;
	if (previousMilliseconds == 0)
	{//first time call		
		previousMilliseconds = curtime;
		return false;
	}

	frames++;
	milliseconds += curtime - previousMilliseconds;
	previousMilliseconds = curtime;

	if (milliseconds >= interval)
	{
		fps = frames * 1000.0f / (float)milliseconds;
		frames = 0;
		milliseconds = 0;
		//milliseconds -= interval;
		return true;
	}
	return false;
}

void FPSCounter::setInterval(int interval)
{
	this->interval = interval;
}

int FPSCounter::getInterval()
{
	return this->interval;
}
