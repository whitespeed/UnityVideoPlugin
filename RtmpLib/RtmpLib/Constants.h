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

#define PROC_TIMEBASE_DEN   1000
#define DEST_FPS  25

//color block sizes
#define PIS_BLOCKS_MAX_SIZE 366 //366=((1500-20-8) -6)/4, 1500字节是链路层的MTU(最大传输单元),20字节是不是IP数据报的首部,8字节是UDP数据报的首部
#define PIS_HEADER_SIZE 6

//color block message type
#define MSG_TYPE_NONE -1
#define MSG_TYPE_HEART 0
#define MSG_TYPE_DISCONNECT 1
#define MSG_TYPE_DRAW 2
#define MSG_TYPE_CLEAR 3
#define MSG_TYPE_SERVER_EXCEPTION 4

//nType for VR360 temp:
//0 启动结果信息  nResult 0单路成功 1双路中一路成功 -1失败
//1 警告信息
//2 日志信息
//3 停止结果信息  nResult 0单路成功 -1失败

//for VR180:
//FIX ME:
//nType 0 FILE输出信息，  nResult 0成功 -1失败 1停止
//nType 1 RTMP输出信息，  nResult 0成功 -1失败 1停止
//nType 2 SDI输出信息， nResult 0成功 -1失败 1停止
//
typedef enum SourceType
{
    NONE_SOURCE = -1,
    PANO_SOURCE = 0,
    PROCPOST_SOURCE = 1,
}SourceType;

typedef enum SdiScaleType
{
    ORI_SCALE = 0, //保持原始比例不拉伸之后填充，360度保持底部色，180度默认底部和两边色其他色可选 填充
    CHANGE_SCALE, //拉伸至高1080 or 2160后填充，360度是满屏，180度两边默认黑色其他色可选 拉伸
}SdiScaleType;
struct StreamOpt {
    int w;//分辨率
    int h;
    int vb;//视频码率
    int ab;//音频码率
    int vc;//视频编码
    int ac;//音频编码
    int sdiad=0; //整体延迟
    int devid=0; //SDI编号
    long adms; //编码推流延迟 音频延迟0
    long fillcol=0; //vr180填充颜色
    SourceType source= PROCPOST_SOURCE;
	char uri[1024] = {0};
    //char file[1024];
	char sdilogo[1024] = { 0 };
    SdiScaleType scaletype= ORI_SCALE;
	char ip[64] = {0};
    short port;
};

struct StreamSourceGroup
{
    VideoFrameSource *_VideoCapture_ = NULL;
    PCMFrameSource   *_AudioCapture_ = NULL;
};
//
typedef enum OutputType
{
    OT_UDPCLIENT = -4,
    OT_VIPSTREAMER = -3,
    OT_MANAGER = -2,
    OT_ENCODER = -1,
    OT_RTMP = 1,
}OutputType;

//
typedef enum ResultType
{
    RT_FAIL = -1,
    RT_OK = 0,
    RT_RETRYING,
    RT_END
}ResultType;

class VROutputCallback
{
public:
    virtual void OnMsg(OutputType nType, ResultType nResult, char *msg) {};
};

enum PtsStratage {
    ptsFromPacket = 0,
    ptsAutoGen,
};

#define OUTPUT_AUDIO_SAMPLE_RATE 44100
#define OUTPUT_AUDIO_SAMLE_FMT AV_SAMPLE_FMT_FLTP
#define OUTPUT_AUDIO_CHANNEL_LAYOUT  AV_CH_LAYOUT_STEREO

#define ENCODER_MAX_AUDIO_FIFO_SECONDS 5 //AUDIO FIFO SIZE 5
#define ENCODER_MAX_AUDIO_FIFO_SAMPLES 220500   //about 5 seconds of PCM for 44100hz

typedef enum OUTPUT_STATUS
{
    STATUS_IDLE = 0,
    STATUS_PREPARING,
    STATUS_PREPARED,
    STATUS_START,
    STATUS_RETRYING,
    STATUS_STOP,
    STATUS_END,
    STATUS_FAIL
}OUTPUT_STATUS;

#ifndef _SINGLETON_H_
#define _SINGLETON_H_

template <class T>
class singleton
{
protected:
    singleton() {};
private:
    singleton(const singleton&) {};
    singleton& operator=(const singleton&) {};
    static T* m_instance;
public:
    template <typename... Args>
    static T* GetInstance(Args&&... args)
    {
        if (m_instance == NULL)
            m_instance = new T(std::forward<Args>(args)...);
        return m_instance;
    }

    static void DestroyInstance()
    {
        if (m_instance)
            delete m_instance;
        m_instance = NULL;
    }
};

template <class T>
T* singleton<T>::m_instance = NULL;

#endif
