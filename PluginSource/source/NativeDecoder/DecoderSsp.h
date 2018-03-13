#pragma once
#include "IDecoder.h"
#include <list>
#include <mutex>
#include "imf/net/loop.h"
#include "imf/net/threadloop.h"
#include "imf/ssp/sspclient.h"
#include <string>
extern "C" {
#include <libavformat\avformat.h>
#include <libswresample\swresample.h>
#include <libavutil\pixdesc.h>
}

struct H264Data
{
	uint8_t* data;
	size_t len;
	uint32_t type;
	uint32_t pts;
	uint32_t frm_no;
	virtual ~H264Data() {
		len = 0;
		frm_no = -1;
		type = -1;
		pts = -1;
		if (data != NULL)
			delete[] data;
		data = NULL;
	}
}; H264Data;


class H264Queue
{
public:
	H264Queue()
	{
		
	}
	~H264Queue()
	{
		release();
	}
	void queue(H264Data* data)
	{
		std::lock_guard<std::mutex> lock(mOpMutex);
		mDataList.push_back(data);
	}
	H264Data* dequeue()
	{
		std::lock_guard<std::mutex> lock(mOpMutex);
		if (mDataList.size() <= 0)
			return NULL;

		H264Data* data = mDataList.front();
		mDataList.pop_front();
		return data;
	}
	size_t size()
	{
		std::lock_guard<std::mutex> lock(mOpMutex);
		return mDataList.size();
	}
	void release()
	{
		std::lock_guard<std::mutex> lock(mOpMutex);
		if (!mDataList.empty())
			mDataList.clear();
	}
private:
	std::list<H264Data*> mDataList;
	std::mutex mOpMutex;
};

class DecoderSsp :
	public IDecoder
{
public:
	DecoderSsp();
	~DecoderSsp();

	virtual bool init(const char* filePath) override;

	virtual bool decode() override;

	void pushVideoFrame(AVFrame* frame);

	virtual void seek(double time) override;

	virtual void destroy() override;

	virtual VideoInfo getVideoInfo() override;

	virtual AudioInfo getAudioInfo() override;

	virtual void setVideoEnable(bool isEnable) override;

	virtual void setAudioEnable(bool isEnable) override;

	virtual void setAudioAllChDataEnable(bool isEnable) override;

	virtual double getVideoFrame(unsigned char** outputY, unsigned char** outputU, unsigned char** outputV) override;

	virtual double getAudioFrame(unsigned char** outputFrame, int& frameSize) override;

	virtual void freeVideoFrame() override;

	virtual void freeAudioFrame() override;

	virtual int getMetaData(char**& key, char**& value) override;

private:
	std::mutex mVideoMutex;
	std::mutex mAudioMutex;

	bool mIsInitialized;
	bool mIsAudioAllChEnabled;
	bool mUseTCP;				//	For RTSP stream.
	bool mIsConnected;
	bool mIsSeekToAny;

	int mFrameBufferNum;

	AVFormatContext* mAVFormatContext;
	AVCodec*		mVideoCodec;
	AVCodec*		mAudioCodec;
	AVCodecContext*	mVideoCodecContext;
	AVCodecContext*	mAudioCodecContext;
	SwrContext*	mSwrContext;

	imf::SspClient * mSspClient;
	imf::ThreadLoop * mThreadLooper;

	AVPacket	mPacket;
	H264Queue mH264Queue;
	std::list<AVFrame*> mVideoFrames;
	std::list<AVFrame*> mAudioFrames;
	unsigned int mVideoBuffMax;
	unsigned int mAudioBuffMax;
	unsigned int mQueueMaxSize;
	std::string mUrl;
	imf::SspVideoMeta mVideoMeta;
	imf::SspAudioMeta mAudioMeta;
	imf::SspAudioMeta mSSpMeta;


	VideoInfo		mVideoInfo;
	AudioInfo	mAudioInfo;

	int initSwrContext();
	void updateBufferState();
	bool isH264QueueReady();
	bool isBuffBlocked();
	void freeFrontFrame(std::list<AVFrame*>* frameBuff, std::mutex* mutex);
	void flushBuffer(std::list<AVFrame*>* frameBuff, std::mutex* mutex);
	int	loadConfig();
	void printErrorMsg(int errorCode);
	void setup(imf::Loop *loop, const char* ip);
	void on_264(uint8_t * data, size_t len, uint64_t pts, uint32_t frm_no, uint32_t type);
	void on_meta(struct imf::SspVideoMeta *v, struct imf::SspAudioMeta *a, struct imf::SspMeta *);
	void on_disconnect();
};

