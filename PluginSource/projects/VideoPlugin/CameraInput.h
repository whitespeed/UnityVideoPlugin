#pragma once

extern "C" {
#include <libavutil\error.h>
#include <libavformat\avio.h>
#include <libavformat\avformat.h>
}

#include <stdio.h>
#include <string>
#include "IStreamInput.h"
#include <functional>
#include <thread>

#include "imf/net/loop.h"
#include "imf/net/threadloop.h"
#include "imf/ssp/sspclient.h"

#ifdef _DEBUG
#pragma comment (lib, "libsspd.lib")
#else
#pragma comment (lib, "libssp.lib")
#endif

class CameraInput : public virtual IStreamInput
{
public:
		CameraInput(const char* data);
		virtual ~CameraInput();
		virtual bool InitAVFormatContext(AVFormatContext *);
		static int IOReadCall(void *data, uint8_t *buf, int buf_size) { return -1; }

private:
	void on_264(uint8_t *data, size_t len, uint64_t handle, uint32_t frm_no, uint32_t type);
	void on_meta(struct imf::SspVideoMeta *V, struct imf::SspAudioMeta *A, struct imf::SspMeta *m);
	void on_disconnect();
	void setup(imf::Loop *loop, const char* addr);

	uint8_t *mBuffer;
	int mBufferSize;
	std::string mUrl;
	imf::SspClient *mSspClient;
	imf::ThreadLoop *mThreadLooper;
	AVIOContext *mIOContext;
	imf::SspVideoMeta mVideoMeta;
	imf::SspAudioMeta mAudioMeta;
	imf::SspMeta mSspMeta;
	bool mIsConnected;
};