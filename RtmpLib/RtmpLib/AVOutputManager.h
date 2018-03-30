#pragma once
#include <string>
#include <queue>
#include <cstdio>

#include "PCMFrameFilter.h"
#include "VideoFrameFilter.h"
#include "AVStreamerNew.h"
#include <cuda_runtime.h>
#include "UDPClient.h"

typedef singleton<UDPClient>  _UDPCLIENT;

class AVOutputManager : public PCMFrameSink, public VideoFrameSink, public VideoFrameSource, public PCMFrameSource, public VROutputCallback
{
public:
	AVOutputManager();
	~AVOutputManager();
	virtual int onPCMFrame(PCMFrame* packet);
	virtual int onVideoFrame(VideoFrame *packet, VideoFrameSource* src);
    int startUdpClient(char *server_ip, unsigned short port);
	void startOutputRtmp(VrType vrType, const std::string& url, StreamOpt streamOpt, VROutputCallback *pCallback);

	int stopOutputRtmp();
    int stopUdpClient();
	void setFrameSourceGroup(StreamSourceGroup &ssg);
private:
	int stop();
	int wait();

	static void* run(void* data);
	void doVideoOutput();

	void OnMsg(OutputType nType, ResultType nResult, char *msg);
	//void stopEncoder();
    void stopEncoder(AVEncoder* avEncoder);
	void registerAndGetMixType();
	void unregisterAndStop();
	void doStopOutputRtmp();
	StreamSourceGroup m_ssg;
public:
	VROutputCallback *m_pCallback = NULL;
	bool m_isRunning = false;
	pthread_t outputThread;
	bool output_start = false;
	VideoFramePool* vpool = 0;
	VideoFrame *m_pVideoFrame = NULL;
	int m_video_width = 0, m_video_height = 0;
	bool videoOutputInit = false;

	bool in_picture_get = false;
	bool first_picture = false;
	//int64_t first_timestamp = 0;

	int g_encode_type = 0;
	StreamType m_streamType;

	AVEncoder* m_encoder_file = NULL;
	AVEncoder* m_encoder = NULL;
	AVStreamerNew* m_streamer_new_file = NULL;
	AVStreamerNew* m_streamer_new_rtmp = NULL;
	bool m_is_reuse_encoder = false;
	VrType m_VRType;
	SourceType m_data_source = NONE_SOURCE;
	OUTPUT_STATUS m_output_state_rtmp = STATUS_IDLE;
	OUTPUT_STATUS m_state = STATUS_IDLE;
	FPSCounter* mFPSCounter = NULL;
	cudaStream_t m_cudaStream = NULL;
};
