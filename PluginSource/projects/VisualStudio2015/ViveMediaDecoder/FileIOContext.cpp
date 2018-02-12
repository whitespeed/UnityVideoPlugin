#include "FileIOContext.h"
#include "Logger.h"



FileIOContext::FileIOContext(const char* s) {
	mPath.assign(s); 

	// allocate buffer
	mBufferSize = 4096; 
	mBuffer = (uint8_t * )av_malloc(mBufferSize); // see destructor for details

											   // open file
	mFileHandler = fopen(mPath.c_str(), "rb"); 
	if ( ! mFileHandler) {
		LOG("FileIOContext: failed to open file %s\n", 
			mPath.c_str()); 
	}

	// allocate the AVIOContext
	mIOContext = avio_alloc_context(
		mBuffer, mBufferSize, // internal buffer and its size
		0, // write flag (1=true, 0=false) 
		(void * )this, // user data, will be passed to our callback functions
		IOReadFunc, 
		0, // no writing
		NULL); 
}


FileIOContext::~FileIOContext() {
	if (mFileHandler)fclose(mFileHandler); 
	av_free(mIOContext); 
	av_free(mBuffer); 
}

bool FileIOContext::initAVFormatContext(AVFormatContext * pCtx) {
	pCtx -> pb = mIOContext; 
	pCtx -> flags |= AVFMT_FLAG_CUSTOM_IO; 

	// you can specify a format directly
	//pCtx->iformat = av_find_input_format("h264");

	// or read some of the file and let ffmpeg do the guessing
	size_t len = fread(mBuffer, 1, mBufferSize, mFileHandler); 
	if (len == 0)return false; 
	fseek(mFileHandler, 0, SEEK_SET); // reset to beginning of file

	return true; 
}
