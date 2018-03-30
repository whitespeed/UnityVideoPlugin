#pragma once
#include "VideoFrameFilter.h"
#include <cuda_runtime.h>
template <typename TYPE, void (TYPE::*onWriteFrameThread)()>
void *template_writeframe(void* param) {
    TYPE* This = (TYPE*)param;
    This->onWriteFrameThread();
    return NULL;
}

class FrameProxy : public VideoFrameSource, public VideoFrameSink
{
public:
    FrameProxy();
    ~FrameProxy();
    int onVideoFrame(VideoFrame *pPakcket, VideoFrameSource* src);
    //static FrameProxy* GetInstance() {
    //    return sInstance;
    //}
    void onWriteFrameThread();
private:
    //static FrameProxy *sInstance;
    VideoFramePool* videopool;
    bool running;
    pthread_t mthread;
    pthread_mutex_t incoming_mutex;  //video packets like
    pthread_cond_t incomingcond;
    AVPixelFormat pix_fmt;
    int width;
    int height;
	cudaStream_t stream;
};