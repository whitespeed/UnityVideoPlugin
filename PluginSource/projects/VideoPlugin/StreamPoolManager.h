#pragma once
extern "C" {
#include <libavformat\avformat.h>
}
#include "H264Queue.h"
#include <iostream>   
#include <list>
#include "IStreamInput.h"
#include "FFMpegDecoder.h"
#include <stdio.h>
using namespace std;

class StreamPoolManager
{

public:
	std::list<AVFrame*> FixStreamList;
	std::list<AVFrame*> DynamicStreamList;
	H264Queue H264Queue;

	StreamPoolManager(IStreamInput *input);
	~StreamPoolManager();

	int RegisterAVIOContext(AVIOContext *);
protected:
	void updateBufferState(FFMpegDecoder::VideoInfo &mVideoInfo, AVFrame* mVideoFrames);
	bool isH264QueueReady();
	bool isBuffBlocked();
};