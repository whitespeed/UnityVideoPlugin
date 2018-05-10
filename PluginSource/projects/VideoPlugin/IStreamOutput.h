#pragma once
#include <IOStream>


extern "C" {
#include <libavformat\avformat.h>
}

class IStreamOutput
{
public:
	virtual ~IStreamOutput() {};

	//Push Or Save to file

	virtual int DoOutput(AVFormatContext *ifmt_ctx);
	virtual int InitAVFormatContext(char * outputFile);

	AVIOContext *mIOContext;
	AVFormatContext *mFormatContext;
	AVOutputFormat *outFormat;

protected:
	uint8_t *mBuffer;
	int mBufferSize;
	AVPacket pkt;
	std::string mUrl;
	static IStreamOutput * self;
};