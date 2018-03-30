#include "FrameProxy.h"
#include <cuda_runtime.h>
#include "P2PLog.h"
#include <Windows.h>

//FrameProxy* FrameProxy::sInstance = new FrameProxy;

FrameProxy::FrameProxy()
{
    videopool = NULL;
    running = true;
	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    pthread_mutex_init(&incoming_mutex, NULL);
    pthread_cond_init(&incomingcond, NULL);
}
FrameProxy::~FrameProxy()
{
    running = false;
    pthread_cond_signal(&incomingcond);

    void* exit_ret;
    pthread_join(mthread, &exit_ret);
    pthread_mutex_lock(&incoming_mutex);
    if (videopool != NULL)
    {
        delete videopool;
    }
    pthread_mutex_unlock(&incoming_mutex);
    pthread_mutex_destroy(&incoming_mutex);
    pthread_cond_destroy(&incomingcond);
}
int FrameProxy::onVideoFrame(VideoFrame * pPakcket, VideoFrameSource * src)
{
    if (!videopool)
    {
        pix_fmt = pPakcket->pix_fmt;
        width = pPakcket->width;
        height = pPakcket->height;
        pthread_create(&mthread, NULL, template_writeframe<FrameProxy, &FrameProxy::onWriteFrameThread>, this);
        return 0;
    }
    VideoFrame* tmp = videopool->popEmpty();
    if (tmp == nullptr)
    {
        LOGI("FrameProxy:popEmpty empty!");
        return -1;
    }
    //cudaMemcpy(tmp->data[0], pPakcket->data[0], pPakcket->size, cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(tmp->data[0], pPakcket->data[0], pPakcket->size, cudaMemcpyDeviceToHost, stream);
    tmp->pts = pPakcket->pts;
    tmp->hw = CUDAHOST;//CPU;
    tmp->pix_fmt = pPakcket->pix_fmt;
    tmp->timebase_num = pPakcket->timebase_num;
    //FIX ME:
    tmp->timebase_den = 1000;// packet->timebase_den;//timebaseÊÇmsµ¥Î»
    tmp->size = pPakcket->size;
    tmp->linesize[0] = pPakcket->linesize[0];
    tmp->nFrameNum = pPakcket->nFrameNum;
	cudaStreamSynchronize(stream);
    pthread_mutex_lock(&incoming_mutex);
    videopool->pushValid(tmp);
    pthread_cond_signal(&incomingcond);
    pthread_mutex_unlock(&incoming_mutex);
    return 0;
}
void FrameProxy::onWriteFrameThread()
{
    videopool = new VideoFramePool(INIT_VIDEO_QUEUE_SIZE, 10, pix_fmt, width, height,1,CUDAHOST);
    while (running)
    {
        pthread_mutex_lock(&incoming_mutex);
        while (videopool->getValid() <= 0 && running)
        {
            struct timespec tv;
            int err = 0;
            tv.tv_sec = time(NULL) + 1;
            tv.tv_nsec = 0;

            err = pthread_cond_timedwait(&incomingcond, &incoming_mutex, &tv);
            if (err == ETIMEDOUT && running == false)
            {
                pthread_mutex_unlock(&incoming_mutex);
                goto out;
            }
        }
        VideoFrame *tmp = videopool->popValid();
        pthread_mutex_unlock(&incoming_mutex);

        if (tmp == nullptr) continue;
        writeVideoFrame(tmp);
        videopool->pushEmpty(tmp);
        
    }
out:
    pthread_exit(0);
}