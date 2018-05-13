#include "FileStreamInput.h"
#include "Logger.h"

#ifndef SAFE_DELETE
#define SAFE_DELETE(p) {if(p){delete(p); (p)=NULL;}}
#endif

FileStreamInput::FileStreamInput()
{
	mBufferSize = 4096;

	PoolManager = new StreamPoolManager(this);
}


FileStreamInput::~FileStreamInput() {
	if (mFileHandler)fclose(mFileHandler); 
	av_free(mIOContext); 
	av_free(mBuffer); 
	avformat_free_context(mFormatContext);
	self = NULL;
}

bool FileStreamInput::initAVFormatContext(char * path) {
	av_register_all();
	av_log_set_level(AV_LOG_DEBUG);

	if (NULL == mFormatContext) {
		mFormatContext = avformat_alloc_context();
	}

	mFormatContext-> pb = mIOContext;
	mFormatContext-> flags |= AVFMT_FLAG_CUSTOM_IO;

	mPath.assign(path);

	// allocate buffer
	mBufferSize = 4096;
	mBuffer = (uint8_t *)av_malloc(mBufferSize); // see destructor for details

												 // open file
	mFileHandler = fopen(mPath.c_str(), "rb");
	if (!mFileHandler) {
		LOG("FileIOContext: failed to open file %s\n",
			mPath.c_str());
	}

	// allocate the AVIOContext
	mIOContext = avio_alloc_context(
		mBuffer, mBufferSize, // internal buffer and its size
		0, // write flag (1=true, 0=false) 
		(void *)this, // user data, will be passed to our callback functions
		IOReadFunc,
		0, // no writing
		NULL);

	// you can specify a format directly
	//pCtx->iformat = av_find_input_format("h264");

	// or read some of the file and let ffmpeg do the guessing
	size_t len = fread(mBuffer, 1, mBufferSize, mFileHandler); 
	if (len == 0)return false; 
	fseek(mFileHandler, 0, SEEK_SET); // reset to beginning of file

	return true; 
}
