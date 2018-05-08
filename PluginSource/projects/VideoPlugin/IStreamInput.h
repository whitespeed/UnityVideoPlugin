#pragma once
extern "C" {
#include <libavformat\avformat.h>
}
#include "StreamPoolManager.h"
class IStreamInput
{
public:
	virtual IStreamInput(){};
	virtual ~IStreamInput() {};

	virtual bool InitAVFormatContext(AVFormatContext *) = 0;

	AVIOContext *mIOContext;

	uint8_t *mBuffer;
	int mBufferSize;

	uint8_t *mBuffer;
	int mBufferSize;
protected:
	StreamPoolManager *PoolManager;
};