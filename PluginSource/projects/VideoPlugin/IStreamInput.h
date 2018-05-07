#pragma once
extern "C" {
#include <libavformat\avformat.h>
}
#include "StreamPoolManager.h"
class IStreamInput
{
public:
	virtual ~IStreamInput() {}

	virtual bool InitAVFormatContext(AVFormatContext *) = 0;
protected:
	StreamPoolManager *PoolManager;
};