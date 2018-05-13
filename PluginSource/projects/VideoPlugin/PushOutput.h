#pragma once
#include <stdio.h>
#include <string>
#include "IStreamOutput.h"
#include <functional>
#include <thread>
extern "C"
{
#include "libavformat/avformat.h"  
#include "libavutil/mathematics.h"  
#include "libavutil/time.h"  
};


class PushOutput : public virtual IStreamOutput
{
public:
	PushOutput();

	virtual ~PushOutput() {};
	virtual int InitAVFormatContext(char * outputFile);

	virtual int DoOutput(AVFormatContext *ifmt_ctx);

private:
	int videoIndex = -1;
	int frame_index = 0;
	int ret, i;
	int64_t start_time = 0;
};