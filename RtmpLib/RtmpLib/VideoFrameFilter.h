#pragma once

#include <stdint.h>
#include <list>
#include <queue>
#include <cuda.h>
extern "C"
{
#include "libavutil/avutil.h"
#include "libavutil/pixdesc.h"

#include "pthread.h"
}
#define INIT_VIDEO_QUEUE_SIZE 3
#define MAX_VIDEO_QUEUE_SIZE 100
//#define FPS_DEBUG
#ifdef FPS_DEBUG
#include "FPSCounter.h"
#endif

class VideoFrame;
class VideoFrameSink;
#define CUDA_ACCEL

typedef enum _HWACCELTYPE_
{
    CPU = 0,
#ifdef CUDA_ACCEL
    CUDA,
    CUDAHOST,
#endif
}HWACCELTYPE;
//区分360度和180度直播的类型
typedef enum VrType
{
    VR180 = 0,
    VR360,
}VrType;

//区分输出流类型（RTMP和FILE）
typedef enum StreamType
{
    ST_FILE = 0,
    ST_RTMP = 1,
}StreamType;
class VideoFrame
{
public:
    VideoFrame(AVPixelFormat format, int width, int height, int alignment = 1, HWACCELTYPE _hw = CPU);
    ~VideoFrame();
    VideoFrame* dump();

    uint8_t *data[4];// video data, alloc and free at data[0]
    int linesize[4];//linesize is always >= width
    int64_t size;  // total buffer size
    int width;
    int height;
    int nFrameNum;

    enum AVPixelFormat pix_fmt;
    int64_t pts;
    int timebase_num;
    int timebase_den;
    HWACCELTYPE hw;
    volatile bool isUsed;
private:
#ifdef CUDA_ACCEL
    int cuda_alloc();
    void cuda_dealloc();
#endif
};

class VideoFrameGroup
{
    VideoFrame* frames;
    int nb_frames;

    int width;
    int height;
    enum AVPixelFormat pix_fmt;
    int64_t pts;
    int timebase_num;
    int timebase_den;
};

class VideoFrameFilter
{
public:
    VideoFrameFilter();
    ~VideoFrameFilter();
};

class VideoFrameSource
{
public:
    VideoFrameSource();
    ~VideoFrameSource();
    virtual void registerVideoFrameSink(VideoFrameSink* sink);
    virtual void unregisterVideoFrameSink(VideoFrameSink* sink);
    virtual void writeVideoFrame(VideoFrame *packet);
	int count();
private:
    pthread_mutex_t handlers_mutex;
    std::list<VideoFrameSink*> handlers;
};

class VideoFrameSink
{
public:
    virtual int onVideoFrame(VideoFrame *packet, VideoFrameSource *nSource) = 0;
};

class VideoFramePool
{
public:
    VideoFramePool(int initSize, int maxSize, AVPixelFormat format, int width, int height, int alignment = 1, HWACCELTYPE _hw = CPU);
    ~VideoFramePool();
    int getValid();
    int getCurrent();
    int gatMax();

    void pushValid(VideoFrame* frame);
    VideoFrame* popValid();

    void pushEmpty(VideoFrame* frame);
    VideoFrame* popEmpty();
    void resetVaild();
private:
    int width;
    int height;
    enum AVPixelFormat pix_fmt;
    HWACCELTYPE hw;
    int alignment;

    std::queue<VideoFrame *> validQueue;
    std::queue<VideoFrame *> emptyQueue;
    pthread_mutex_t videoQueueMutex;
    pthread_mutex_t emptyQueueMutex;
    volatile int validSize = 0;
    volatile int currentSize = 0;
    int maxSize = INIT_VIDEO_QUEUE_SIZE;
};
