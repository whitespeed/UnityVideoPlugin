#pragma once

extern "C" {
#include <libavutil\error.h>
#include <libavformat\avio.h>
#include <libavformat\avformat.h>
}

#include <stdio.h>
#include <string>
#include "IIOContext.h"
class FileIOContext:public virtual IIOContext
{
public:
	std::string mPath;
	AVIOContext *mIOContext;
	uint8_t *mBuffer; // internal buffer for ffmpeg
	int mBufferSize;
	FILE *mFileHandler;
public:
	FileIOContext(const char* url);
	virtual ~FileIOContext();

	bool initAVFormatContext(AVFormatContext *);
};

static int IOReadFunc(void *data, uint8_t *buf, int buf_size) {
	FileIOContext *hctx = (FileIOContext*)data;
	size_t len = fread(buf, 1, buf_size, hctx->mFileHandler);
	if (len == 0) {
		// Let FFmpeg know that we have reached EOF, or do something else
		return AVERROR_EOF;
	}
	return (int)len;
}

// whence: SEEK_SET, SEEK_CUR, SEEK_END (like fseek) and AVSEEK_SIZE
static int64_t IOSeekFunc(void *data, int64_t pos, int whence) {
	if (whence == AVSEEK_SIZE) {
		// return the file size if you wish to
	}
	FileIOContext *hctx = (FileIOContext*)data;
	long fpos = ftell(hctx->mFileHandler); // int64_t is usually long long
	if (whence > fpos)
	{
		return -1;
	}
	int rs = fseek(hctx->mFileHandler, (long)pos, whence);
	if (rs != 0) {
		return -1;
	}
	fpos = ftell(hctx->mFileHandler); // int64_t is usually long long
	return (int64_t)fpos;
}