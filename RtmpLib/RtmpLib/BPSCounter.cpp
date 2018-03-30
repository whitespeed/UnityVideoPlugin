#include "BPSCounter.h"

BPSCounter::BPSCounter()
{
}

BPSCounter::~BPSCounter()
{
}

int BPSCounter::getkbps()
{
	return kbps;
}

bool BPSCounter::NewPacket(int pktsize)
{
	int64_t curtime = av_gettime() / 1000;
	if (previousMilliseconds == 0)
	{//first time call		
		previousMilliseconds = curtime;
		return false;
	}

	size += pktsize;
	milliseconds += curtime - previousMilliseconds;
	previousMilliseconds = curtime;

	if (milliseconds >= interval)
	{
		kbps = size * 8 / milliseconds; // * 1000 kbps / 1000ms
		size = 0;
		milliseconds -= interval;
		return true;
	}
	return false;
}

void BPSCounter::setInterval(int interval)
{
	this->interval = interval;
}

int BPSCounter::getInterval()
{
	return this->interval;
}
