#pragma once
#include <string>
#include <queue>
#include <cstdio>

#include "EncodedPacketFilter.h"
//#include "tcp_connect.h"
#include "AVEncoder.h"

class AVStreamerNew : public EncodedPacketSink
{
public:
    AVStreamerNew(const std::string& url, AVEncoder *encoder);
    ~AVStreamerNew();
    int static instance_ref_count;
    int start(/*int mix_type,*/ VROutputCallback *pCallback);
    int stop();
    //int wait();
    virtual int onEncodedPacket(EncodedPacket *packet, EncodedPacketSource *nSource);
    std::string get_error_text(int error);
    //bool getDomain(const char* url, string& domain);
private:
    int openAudioStream();
    int openVideoStream();
    static void* run(void* data);
    int doStream();
    void OnMsg(OutputType nType, ResultType nResult, char *msg);
public:
    AVEncoder *m_encoder = NULL;
    VROutputCallback *m_pCallback = NULL;
    std::string url;
    AVFormatContext *ofmt_ctx = NULL;
    AVStream *video_st = NULL, *audio_st = NULL;

    bool stopFlag = false;
    pthread_t newStreamThread;
    pthread_mutex_t packetQueueMutex;

    bool audioStreamInit = false;
    bool stream_start = false;

    EncodedPacketPool* packetPool = 0;
    bool videoStreamInit = false;

    bool in_packet_get = false;
    OutputType m_output_type = OT_RTMP;
    //VrType g_enc_vrtype = VR360;
    //int g_mix_type = 0;
    //FPSCounter* mFPSCounter;
    //CPing		ItemPing2;
    bool realstopFlag = false;
    //int64_t m_time_retry = 0;
};
