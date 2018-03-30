#pragma once
#include <string>
#include <queue>
#include <cstdio>

extern "C"
{
#include "libavcodec/avcodec.h"
#include "libavdevice/avdevice.h"
#include "libavformat/avformat.h"
#include "libavfilter/avfiltergraph.h"
#include "libavfilter/avfilter.h"
#include "libavfilter/buffersink.h"
#include "libavfilter/buffersrc.h"
#include "libavutil/avutil.h"
#include "libavutil/audio_fifo.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/time.h"
#include "libavutil/imgutils.h"
#include "libswresample/swresample.h"
#include "pthread.h"
}
//#include "CaptureDevice.h"
#include "PCMFrameFilter.h"
#include "VideoFrameFilter.h"
#include "EncodedPacketFilter.h"
#include "Constants.h"

class FPSCounter;
class BPSCounter;

class AVEncoder : public PCMFrameSink, public VideoFrameSink, public EncodedPacketSource
{
public:
    AVEncoder(StreamType streamType);
    ~AVEncoder();
    int start(/*int mix_type, */VROutputCallback *pCallback);
    int stop();
    int wait();
    int initVideoParams(int64_t bit_rate);
    int initVideoParams(int64_t bit_rate, int width, int height);
    void setPtsStratage(PtsStratage stratage);
    PtsStratage getPtsStratage();

    int initAudioParams(int64_t bit_rate);
    void setAudioDelay(int millisecond);
    void setLive(bool mode);

    virtual int onPCMFrame(PCMFrame* packet);
    virtual int onVideoFrame(VideoFrame *packet, VideoFrameSource* src);

    std::string get_error_text(int error);

private:
    int openAudioEncoder();
    int openVideoEncoder();
    int check_sample_fmt(AVCodec *codec, enum AVSampleFormat sample_fmt);
    int select_sample_rate(AVCodec *codec);
    int64_t select_channel_layout(AVCodec *codec);
    static void* run(void* data);
    void doEncode();

    int encAudio(AVFrame * frame);
    int encVideo(AVFrame * frame);
    void OnMsg(OutputType nType, ResultType nResult, char *msg);
public:
    VROutputCallback *m_pCallback = NULL;
    //for audio mixing:
    PCMFrameFilter *pcm = NULL;
    int audioDelay = 0;
    int videoDelay = 0;
    AVCodecContext *pVOCodecCtx = NULL, *pAOCodecCtx = NULL;
    AVCodec *pVOCodec = NULL, *pAOCodec = NULL;

    bool stopFlag = false;
    bool liveMode = false;
    pthread_t encodeThread;

    AVAudioFifo *fifo = NULL;
    pthread_mutex_t audioQueueMutex;
    pthread_cond_t	audioQueueCond;
    int64_t audio_next_pts = 0;
    AVFrame *audio_frame;
    int audio_stream_index;
    int64_t audio_bit_rate;
    bool audioEncodeInit = false;
    bool encode_start = false;

    bool in_audio_get = false;

    VideoFramePool* vpool = 0;
    PtsStratage mPtsStratage = ptsFromPacket;// ptsAutoGen;
    int64_t video_next_pts = 0;
    int64_t v_timestamp = 0;
    AVFrame *video_frame;
    int picture_size;
    int video_stream_index;
    uint8_t* pData;
    AVPixelFormat video_pix_fmt = AV_PIX_FMT_YUV420P;
    int video_width = 0, video_height = 0;
    int64_t video_bit_rate = 4 * 1024 * 1024;
    bool videoParamInit = false;
    bool videoEncodeInit = false;

    bool in_picture_get = false;
    bool first_picture = false;
    AVPixelFormat in_picture_pix_fmt;
    int in_picture_width = 0;
    int in_picture_height = 0;
    int64_t first_timestamp = 0;
    int64_t packet_pre_timestamp = -1;

    int post_packet_num = 0;
    uint64_t post_packet_time = 0;

    //for audio mix:
    bool get_mixflag = false;
    MixType audiomix = SINGLE;
    int first_mix_audiotype = 0;

    int g_encode_type = 0; // 0 for file ,1 for rtmp
    //int g_mix_type = 0;

    FPSCounter* mFPSCounter;
    BPSCounter* mBPSCounter;

    StreamType m_streamType;

    int  m_sample_rate;
    AVSampleFormat  m_sample_fmt;
    int m_channels;
    int m_channel_layout;

    SwrContext *m_pSwr_ctx = NULL;
    uint8_t** m_resample_buf = NULL;
    int m_resample_buf_size = 1024 * 1024;
};
