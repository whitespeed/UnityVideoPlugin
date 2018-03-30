#pragma once
extern "C"
{
#include "libavutil\time.h"
}


class BPSCounter
{

public:
	BPSCounter();
	~BPSCounter();

	int getkbps();
	bool NewPacket(int pktsize);
	void setInterval(int interval);
	int getInterval();
private:
	int size = 0;
	int64_t previousMilliseconds = 0;
	int milliseconds = 0;
	int interval = 1000;
	int kbps = 0;
};