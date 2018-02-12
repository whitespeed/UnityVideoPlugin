#include "SspIOContext.h"
#include <functional>
#include "Logger.h"

using namespace std::placeholders;

#ifndef SAFE_DELETE 
#define SAFE_DELETE(p) { if(p){delete(p);  (p)=NULL;} }
#endif
SspIOContext::SspIOContext(const char* path)
{
	mUrl.assign(path);
	mBufferSize = 4096;
	mBuffer = (uint8_t *)av_malloc(mBufferSize
	); // see destructor for details
	mThreadLooper = new imf::ThreadLoop(std::bind(&SspIOContext::setup,this,_1,path));
	mThreadLooper->start();

	mIOContext = avio_alloc_context(
		mBuffer, mBufferSize, // internal buffer and its size
		0, // write flag (1=true, 0=false) 
		(void *)this, // user data, will be passed to our callback functions
		IOReadFunc,
		0, // no writing
		NULL);
	mIsConnected = false;
}


SspIOContext::~SspIOContext()
{
	av_free(mIOContext);
	av_free(mBuffer);
	mThreadLooper->stop();
	//SAFE_DELETE(mThreadLooper);
	//TODO: delete client
}

bool SspIOContext::initAVFormatContext(AVFormatContext *avFormatContext)
{
	avFormatContext->pb = mIOContext;
	avFormatContext->flags |= AVFMT_FLAG_FLUSH_PACKETS;
	avFormatContext->flags |= AVFMT_FLAG_CUSTOM_IO;
	return true;
}

void SspIOContext::on_264(uint8_t * data, size_t len, uint64_t pts, uint32_t frm_no, uint32_t type)
{
	LOG("on H264 %d [%d] [%lld]\n", frm_no, type, len);
}

void SspIOContext::on_meta(struct imf::SspVideoMeta *v, struct imf::SspAudioMeta *a, struct imf::SspMeta *s)
{
	LOG("on meta");
	mIsConnected = true;
	memcpy(&mVideoMeta, v, sizeof(imf::SspVideoMeta));
	memcpy(&mAudioMeta, a, sizeof(imf::SspVideoMeta));
	memcpy(&mSSpMeta, s, sizeof(imf::SspVideoMeta));
}

void SspIOContext::on_disconnect()
{
	LOG("on disconnet");
	mIsConnected = false;
}

void SspIOContext::setup(imf::Loop *loop, const char* url)
{
	std::string ip(url);
	mSspClient = new imf::SspClient(ip, loop, 0x400000);
	mSspClient->init();
	mSspClient->setOnH264DataCallback(std::bind(&SspIOContext::on_264, this,_1, _2, _3, _4, _5));
	mSspClient->setOnMetaCallback(std::bind(&SspIOContext::on_meta,this, _1, _2, _3));
	mSspClient->setOnDisconnectedCallback(std::bind(&SspIOContext::on_disconnect,this));
	mSspClient->start();
}
