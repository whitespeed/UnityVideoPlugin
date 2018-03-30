#pragma once
extern "C"
{
#include "libavutil\time.h"
}


class FPSCounter
{

public:
	FPSCounter();
	~FPSCounter();

	float getFps();
	bool NewFrame();
	void setInterval(int interval);
	int getInterval();
private:
	int frames = 0;
	int64_t previousMilliseconds = 0;
	int milliseconds = 0;
	int interval = 5000;
	float fps = 0;
};