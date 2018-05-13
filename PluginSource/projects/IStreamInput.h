#pragma once
extern "C" {
#include <libavformat\avformat.h>
}
class IStreamInput
{
public:
	virtual ~IStreamInput() {}

	virtual bool InitData() = 0;
};