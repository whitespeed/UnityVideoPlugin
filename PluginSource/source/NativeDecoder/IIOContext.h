#pragma once

extern "C" {
#include <libavformat\avformat.h>
}
class IIOContext
{
public:
	virtual ~IIOContext() {}

	virtual bool initAVFormatContext(AVFormatContext *) = 0;
};