#include "CameraInput.h"
#include <functional>
#include "Logger.h"

using namespace std::placeholders;

#ifndef SAFE_DELETE
#define SAFE_DELETE(p) {if(p){delete(p); (p)=NULL;}}
#endif

CameraInput::CameraInput(const char* data)
{
	mUrl.assign(data);
	mBufferSize = 4096;
	mBuffer = (uint8_t *)av_malloc(mBufferSize);
	mThreadLooper = new imf::ThreadLoop(std::bind(&CameraInput::setup, this, _1, data));
	mThreadLooper->start();

	mIOContext = avio_alloc_context(mBuffer, mBufferSize, 0, (void *)this, IOReadCall, 0, NULL);
	mIsConnected = false;
}

CameraInput::~CameraInput()
{
	av_free(mIOContext);
	av_free(mBuffer);
	mThreadLooper->stop();
	SAFE_DELETE(mThreadLooper);
	SAFE_DELETE(mSspClient);
}

bool CameraInput::InitAVFormatContext(AVFormatContext *ctx)
{
	ctx->pb = mIOContext;
	ctx->flags |= AVFMT_FLAG_FLUSH_PACKETS;
	ctx->flags |= AVFMT_FLAG_CUSTOM_IO;
	return true;
}

void CameraInput::on_264(uint8_t * data, size_t len, uint64_t pts, uint32_t frm_no, uint32_t type)
{
	LOG("on H264 %d [%d] [%lld]\n", frm_no, type, len);
}

void CameraInput::on_meta(struct imf::SspVideoMeta *v, struct imf::SspAudioMeta *a, struct imf::SspMeta *s)
{
	LOG("on meta");
	mIsConnected = true;
	memcpy(&mVideoMeta, v, sizeof(imf::SspVideoMeta));
	memcpy(&mAudioMeta, a, sizeof(imf::SspVideoMeta));
	memcpy(&mSspMeta, s, sizeof(imf::SspVideoMeta));
}

void CameraInput::on_disconnect()
{
	LOG("on disconnet");
	mIsConnected = false;
}

void CameraInput::setup(imf::Loop *loop, const char* url)
{
	std::string ip(url);
	mSspClient = new imf::SspClient(ip, loop, 0x400000);
	mSspClient->init();
	mSspClient->setOnH264DataCallback(std::bind(&CameraInput::on_264, this, _1, _2, _3, _4, _5));
	mSspClient->setOnMetaCallback(std::bind(&CameraInput::on_meta, this, _1, _2, _3));
	mSspClient->setOnDisconnectedCallback(std::bind(&CameraInput::on_disconnect, this));
	mSspClient->start();
}