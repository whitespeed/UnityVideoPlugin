#include "VideoFrameFilter.h"
#ifdef CUDA_ACCEL
#include <cuda_runtime.h>
#endif
extern "C"
{
#include "libavutil/Imgutils.h"
}

#define TAG "VideoFrame"
#include "P2PLog.h"

VideoFrame::VideoFrame(AVPixelFormat format, int width, int height, int alignment, HWACCELTYPE _hw) :
	pts(0), timebase_num(1), timebase_den(1000), nFrameNum(1)
{
	this->pix_fmt = format;
	this->width = width;
	this->height = height;
	this->hw = _hw;
	for (int i = 0; i < 4; i++)
	{
		data[i] = NULL;
	}
	switch (_hw)
	{
#ifdef CUDA_ACCEL
	case CUDA:
	case CUDAHOST:
		size = cuda_alloc();
		break;
#endif
	case CPU:
	default:
		size = av_image_alloc(data, linesize, width, height, format, alignment);
		this->hw = CPU;
		break;
	}
}

VideoFrame::~VideoFrame()
{
	switch (hw)
	{
	case CPU:
		if (size > 0)
		{
			LOGV("~VideoFrame");
			av_freep(&data[0]);
			size = 0;
			LOGV("~VideoFrame end");
		}
		break;
#ifdef CUDA_ACCEL
	case CUDA:
	case CUDAHOST:
		cuda_dealloc();
		break;
#endif
	default:
		break;
	}
}

VideoFrame* VideoFrame::dump()
{
	VideoFrame* ret = NULL;
	if (nFrameNum > 1)
	{
		ret = new VideoFrame(pix_fmt, width, height*nFrameNum);
		ret->width = width;
		ret->height = height;
	}
	else
	{
		ret = new VideoFrame(pix_fmt, width, height);
	}

	cudaMemcpy(ret->data[0], data[0], size, cudaMemcpyDeviceToHost);
	//memcpy(ret->data[0], data[0], size);
	ret->hw = CPU;
	ret->pix_fmt = pix_fmt;
	ret->pts = pts;
	ret->timebase_num = timebase_num;
	ret->timebase_den = timebase_den;
	ret->size = size;
	ret->linesize[0] = linesize[0];
	ret->nFrameNum = nFrameNum;
	return ret;
}

#ifdef CUDA_ACCEL
int VideoFrame::cuda_alloc()
{
	int buffer_size = -1;
	switch (pix_fmt)
	{
	case AV_PIX_FMT_YUV420P:
		buffer_size = width*height * 3 / 2;
		break;
	case AV_PIX_FMT_UYVY422:
		buffer_size = width*height * 2;
		break;
	case AV_PIX_FMT_RGB24:
	case AV_PIX_FMT_BGR24:
		buffer_size = width*height * 3;
		break;
	case AV_PIX_FMT_ARGB:
	case AV_PIX_FMT_RGBA:
	case AV_PIX_FMT_ABGR:
	case AV_PIX_FMT_BGRA:
		buffer_size = width*height * 4;
		break;
	default:
		LOGE("can't support this pix format!");
		return -1;
	}
	cudaError_t st;
	if (hw == CUDA)
		st = cudaMalloc((void**)&data[0], buffer_size * sizeof(uint8_t));
	else
		st = cudaHostAlloc((void**)&data[0], buffer_size * sizeof(uint8_t), cudaHostAllocDefault);
	if (st || NULL == data[0])
		printf("VIDEOFRAMEFILTER ALLOC ERROR %d\n", st);
	if (pix_fmt == AV_PIX_FMT_YUV420P)
	{
		data[1] = data[0] + width*height;
		data[2] = data[0] + width*height * 5 / 4;
	}
	av_image_fill_linesizes(linesize, pix_fmt, width);
	return buffer_size;
}

void VideoFrame::cuda_dealloc()
{
	if (size > 0)
	{
		if (hw == CUDA)
			cudaFree(data[0]);
		else
			cudaFreeHost(data[0]);
	}
}
#endif
VideoFrameFilter::VideoFrameFilter()
{
}

VideoFrameFilter::~VideoFrameFilter()
{
}

VideoFrameSource::VideoFrameSource()
{
	pthread_mutex_init(&handlers_mutex, NULL);
}

VideoFrameSource::~VideoFrameSource()
{
	pthread_mutex_destroy(&handlers_mutex);
}

void VideoFrameSource::registerVideoFrameSink(VideoFrameSink* sink)
{
	bool already_in = false;
	pthread_mutex_lock(&handlers_mutex);
	handlers.push_back(sink);
	handlers.sort();
	handlers.unique();
	pthread_mutex_unlock(&handlers_mutex);
}

void VideoFrameSource::unregisterVideoFrameSink(VideoFrameSink* sink)
{
	pthread_mutex_lock(&handlers_mutex);
	if (sink)
		handlers.remove(sink);
	else
		handlers.clear();
	pthread_mutex_unlock(&handlers_mutex);
}
int VideoFrameSource::count()
{
	return handlers.size();
}
void VideoFrameSource::writeVideoFrame(VideoFrame *packet)
{
	pthread_mutex_lock(&handlers_mutex);
	std::list<VideoFrameSink*>::iterator iter = handlers.begin();
	for (; iter != handlers.end(); iter++) {
		if ((*iter)->onVideoFrame(packet, this) < 0)
		{
			iter = handlers.erase(iter);
			if (iter == handlers.end())
			{
				break;
			}
		}
	}
	pthread_mutex_unlock(&handlers_mutex);
}

VideoFramePool::VideoFramePool(int initSize, int maxSize, AVPixelFormat format, int width, int height, int alignment, HWACCELTYPE _hw)
{
	if (initSize < 0) {
		initSize = INIT_VIDEO_QUEUE_SIZE;
	}

	if (maxSize < 0)
	{
		maxSize = INIT_VIDEO_QUEUE_SIZE;
	}

	this->pix_fmt = format;
	this->width = width;
	this->height = height;
	this->alignment = alignment;
	this->hw = _hw;
	this->validSize = 0;
	this->currentSize = initSize;
	this->maxSize = maxSize;

	for (int i = 0; i < initSize; i++)
	{
		VideoFrame* temp = new VideoFrame(format, width, height, alignment, _hw);
		emptyQueue.push(temp);
	}
	pthread_mutex_init(&videoQueueMutex, NULL);
	pthread_mutex_init(&emptyQueueMutex, NULL);
}

VideoFramePool::~VideoFramePool()
{
	pthread_mutex_lock(&videoQueueMutex);
	while (!validQueue.empty())
	{
		VideoFrame* temp = validQueue.front();
		validQueue.pop();
		delete temp;
	}
	pthread_mutex_unlock(&videoQueueMutex);

	pthread_mutex_lock(&emptyQueueMutex);
	while (!emptyQueue.empty())
	{
		VideoFrame* temp = emptyQueue.front();
		emptyQueue.pop();
		delete temp;
	}
	pthread_mutex_unlock(&emptyQueueMutex);

	pthread_mutex_destroy(&videoQueueMutex);
	pthread_mutex_destroy(&emptyQueueMutex);
}

int VideoFramePool::getValid()
{
	return validQueue.size();
}

int VideoFramePool::getCurrent()
{
	return currentSize;
}

int VideoFramePool::gatMax()
{
	return maxSize;
}

void VideoFramePool::pushValid(VideoFrame* frame)
{
	pthread_mutex_lock(&videoQueueMutex);
	validQueue.push(frame);
	pthread_mutex_unlock(&videoQueueMutex);
}

VideoFrame* VideoFramePool::popValid()
{
	VideoFrame* ret = nullptr;
	pthread_mutex_lock(&videoQueueMutex);
	if (validQueue.size() > 0) {
		ret = validQueue.front();
		validQueue.pop();
	}
	pthread_mutex_unlock(&videoQueueMutex);
	return ret;
}

void VideoFramePool::pushEmpty(VideoFrame* frame)
{
	pthread_mutex_lock(&emptyQueueMutex);
	emptyQueue.push(frame);
	pthread_mutex_unlock(&emptyQueueMutex);
}

VideoFrame* VideoFramePool::popEmpty()
{
	VideoFrame* ret = nullptr;
	pthread_mutex_lock(&emptyQueueMutex);
	if (emptyQueue.size() > 0) {
		ret = emptyQueue.front();
		emptyQueue.pop();
	}
	pthread_mutex_unlock(&emptyQueueMutex);

	if (ret == nullptr && currentSize < maxSize)
	{
		currentSize++;
		LOGI("VideoFramePool: new VideoFrame, current size:%d, %p", currentSize, this);
		ret = new VideoFrame(pix_fmt, width, height, alignment, hw);
	}
	return ret;
}

void VideoFramePool::resetVaild()
{
    pthread_mutex_lock(&videoQueueMutex);
    while (validQueue.size() > INIT_VIDEO_QUEUE_SIZE)
    {
        VideoFrame* temp = validQueue.front();
        validQueue.pop();
        delete temp;
    }
    this->currentSize = INIT_VIDEO_QUEUE_SIZE;
    pthread_mutex_unlock(&videoQueueMutex);
}
