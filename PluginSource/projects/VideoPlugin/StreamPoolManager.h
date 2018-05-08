#pragma once
extern "C" {
#include <libavformat\avformat.h>
}

#include <iostream>   
#include <list>
#include "IStreamInput.h"
#include "FFMpegDecoder.h"
#include <stdio.h>
using namespace std;

class StreamPoolManager
{

	enum BufferState {EMPTY, NORMAL, FULL};

public:
	std::list<AVFrame*> FixStreamList;
	std::list<AVFrame*> DynamicStreamList;
	H264Queue H264Queue;

	int StreamPoolManager(IStreamInput *input);

	int RegisterAVIOContext(AVIOContext *);
protected:
	void updateBufferState(FFMpegDecoder.VideoInfo &mVideoInfo, AVFrame* mVideoFrames);
	bool isH264QueueReady();
	bool isBuffBlocked();
};