#pragma once
extern "C" {
#include <libavformat\avformat.h>
}
#include "StreamPoolManager.h"
class IStreamInput
{
public:

	virtual ~IStreamInput() {};

	virtual bool InitAVFormatContext(char * path) = 0;

	AVFormatContext *mFormatContext;
	AVIOContext *mIOContext;

	uint8_t *mBuffer;
	int mBufferSize;

protected:
	StreamPoolManager *PoolManager;
	static IStreamInput* self;
};