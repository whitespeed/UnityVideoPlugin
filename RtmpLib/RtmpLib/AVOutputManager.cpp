#include <stdio.h>
#include <math.h>

#include "AVOutputManager.h"
#define TAG "AVOutputManager"
#include "P2PLog.h"
#include "MessageDelivery.h"
#include "cuda_runtime.h"
#include "libyuv.h"
#include "FPSCounter.h"
#include "BPSCounter.h"


using namespace libyuv;

AVOutputManager::AVOutputManager() {
    LOGI("AVOutputManager");
    //m_pPCMFileter = new PCMFrameFilter();

    mFPSCounter = new FPSCounter();
    mFPSCounter->setInterval(5000);
    m_state = STATUS_IDLE;
    m_isRunning = true;
    cudaStreamCreateWithFlags(&m_cudaStream, cudaStreamNonBlocking);
}

AVOutputManager::~AVOutputManager()
{
    //if (m_pPCMFileter != NULL)
    //{
    //    delete m_pPCMFileter;
    //    m_pPCMFileter = NULL;
    //}
    if (vpool != NULL)
    {
        delete vpool;
        vpool = NULL;
    }
    if (m_encoder_file)
    {
        delete m_encoder_file;
        m_encoder_file = NULL;
    }
    if (m_encoder)
    {
        delete m_encoder;
        m_encoder = NULL;
    }
    if (m_streamer_new_file)
    {
        delete m_streamer_new_file;
        m_streamer_new_file = NULL;
    }
    if (m_streamer_new_rtmp)
    {
        delete m_streamer_new_rtmp;
        m_streamer_new_rtmp = NULL;
    }
    if (m_pVideoFrame)
    {
        delete m_pVideoFrame;
        m_pVideoFrame = NULL;
    }
    if (mFPSCounter)
    {
        delete mFPSCounter;
        mFPSCounter = NULL;
    }
	if (m_cudaStream)
		cudaStreamDestroy(m_cudaStream);
    LOGI("~AVOutputManager\n");
}

int AVOutputManager::stop()
{
    output_start = false;
    m_is_reuse_encoder = false;
    MessageDelivery::sendMessage(EventOutputEnd, 0);
    return 0;
}

int AVOutputManager::wait()
{
    m_isRunning = false;
    pthread_join(outputThread, NULL);
    return 0;
}

void* AVOutputManager::run(void* data) {
    AVOutputManager* encoder = (AVOutputManager*)data;
    encoder->doVideoOutput();
    return 0;
}

void AVOutputManager::doVideoOutput()
{
    int64_t startTime = av_gettime();
    output_start = true;
	m_isRunning = true;
    OnMsg(OT_MANAGER, RT_OK, "doVideoOutput start");
    while (m_isRunning)
    {
        VideoFrame* video = vpool->popValid();
        bool gotFrame = video == nullptr ? false : true;

        if (gotFrame)
        {
            //自适应pix_fmt:
            if (video->pix_fmt == AV_PIX_FMT_ARGB) {
            libyuv:ARGBToI420(video->data[0], video->width * 4, m_pVideoFrame->data[0], video->width, m_pVideoFrame->data[1], video->width / 2, m_pVideoFrame->data[2], video->width / 2, video->width, video->height);
            }
            else {
                //default:
                av_image_copy(m_pVideoFrame->data, m_pVideoFrame->linesize, (const uint8_t **)video->data, video->linesize, video->pix_fmt, video->width, video->height);
            }
            m_pVideoFrame->pts = video->pts;
            vpool->pushEmpty(video);

            if (mFPSCounter->NewFrame()) {
                LOGI("doVideoOutput:video fps %.2f", mFPSCounter->getFps());
            }
            writeVideoFrame(m_pVideoFrame);
        }
        else
        {
            av_usleep(1000);
        }
        av_usleep(1000);
    }
	in_picture_get = false;
    int64_t endTime = av_gettime();
    LOGI("End output manager cost time %lld ", endTime - startTime);
}

int AVOutputManager::onPCMFrame(PCMFrame* packet)
{
    writePCMFrame(packet);
    return 0;
}

int AVOutputManager::onVideoFrame(VideoFrame *packet, VideoFrameSource* src)
{
    uint64_t start = av_gettime();
    LOGV("onVideoFrame");
    if (in_picture_get == false)
    {
        in_picture_get = true;
        if (m_video_width <= 0 || m_video_height <= 0)
        {
            m_video_width = packet->width;
            m_video_height = packet->height;
        }

        //init video output
        m_pVideoFrame = new VideoFrame(AV_PIX_FMT_YUV420P, m_video_width, m_video_height, 1, CUDAHOST);
        vpool = new VideoFramePool(INIT_VIDEO_QUEUE_SIZE, MAX_VIDEO_QUEUE_SIZE, packet->pix_fmt, m_video_width, m_video_height, 1, CUDAHOST);
		LOGI("new vpool %p", vpool);
        int ret = pthread_create(&outputThread, NULL, run, this);
        if (ret != 0)
        {
            in_picture_get = false;
            OnMsg(OT_MANAGER, RT_FAIL, "fail to create thread");
            MessageDelivery::sendMessage(EventOutputFailed, ret);
            return ret;
        }
    }

    if (output_start == false)
    {
        LOGV("onVideoFrame drop");
        return 0;
    }

    //if (first_picture == false)
    //{
    //    first_timestamp = packet->pts;
    //    first_picture = true;
    //}
    VideoFrame* temp = vpool->popEmpty();
    if (temp == nullptr)
    {
        LOGI("AVOutputManager: pop VideoFrame empty!!!");
        MessageDelivery::sendMessage(ErrorOutputVideoQueue, 0);
        return 0;
    }
    if (packet->hw == CUDA)
    {
        //cudaMemcpy(temp->data[0], packet->data[0], packet->size, cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(temp->data[0], packet->data[0], packet->size, cudaMemcpyDeviceToHost, m_cudaStream);
    }
    else
    {
        memcpy(temp->data[0], packet->data[0], packet->size);
    }

    temp->pts = packet->pts/* - first_timestamp*/;
    temp->hw = CUDAHOST;
    temp->pix_fmt = packet->pix_fmt;
    temp->timebase_num = packet->timebase_num;
    //FIX ME:
    temp->timebase_den = 1000;// packet->timebase_den;//timebase是ms单位
    temp->size = packet->size;
    temp->linesize[0] = packet->linesize[0];
    temp->nFrameNum = packet->nFrameNum;
	if (packet->hw == CUDA)
	{
		cudaStreamSynchronize(m_cudaStream);
	}
    vpool->pushValid(temp);
    return 0;
}



void AVOutputManager::stopEncoder(AVEncoder* avEncoder)
{
    //stop encoder
    if (avEncoder)
    {
        avEncoder->wait();
        unregisterVideoFrameSink(avEncoder);
        unregisterPCMFrameSink(avEncoder);
        avEncoder->stop();
        delete avEncoder;
        avEncoder = NULL;
    }
    else
    {
        LOGI("stopEncoder: avEncoder is NULL");
    }
}

void AVOutputManager::registerAndGetMixType()
{
 m_ssg._VideoCapture_->registerVideoFrameSink(this);
    //AudioCapture::GetInstance()->registerPCMFrameSink(this);
    m_ssg._AudioCapture_->registerPCMFrameSink(this);
}

void AVOutputManager::unregisterAndStop()
{
    wait();
    m_ssg._VideoCapture_->unregisterVideoFrameSink(this);

    //AudioCapture::GetInstance()->unregisterPCMFrameSink(this);
    m_ssg._AudioCapture_->unregisterPCMFrameSink(this);
    stop();
    OnMsg(OT_MANAGER, RT_END, "End(called unregisterAndStop)");
}

void AVOutputManager::OnMsg(OutputType nType, ResultType nResult, char *msg)
{
	switch (nType)
	{
	case OT_RTMP:
		LOGI("OnMsg: Rtmp %s", msg);
		if (nResult == RT_OK)
		{
			m_output_state_rtmp = STATUS_START;
		}
		else if (nResult == RT_END)
		{
			doStopOutputRtmp();
			m_output_state_rtmp = STATUS_END;
		}
		else if (nResult == RT_FAIL)
		{
			if (stopOutputRtmp() < 0)
			{
                if (m_is_reuse_encoder)
                {
                    if (AVStreamerNew::instance_ref_count < 1)
                    {
                        stopEncoder(m_encoder);
                        m_encoder = NULL;
                    }
                }
                else
                {
                    stopEncoder(m_encoder);
                    m_encoder = NULL;
                }
			}
			m_output_state_rtmp = STATUS_FAIL;
		}
		else if (nResult == RT_RETRYING)
		{
			m_output_state_rtmp = STATUS_RETRYING;
		}
		break;
	
	case OT_MANAGER:
		LOGI("OnMsg: Output Manager %s", msg);
		if (nResult == RT_OK)
		{
			m_state = STATUS_START;
		}
		else if (nResult == RT_END)
		{
			m_state = STATUS_END;
		}
		else if (nResult == RT_FAIL)
		{
			m_state = STATUS_FAIL;
		}
		break;
	case OT_ENCODER:
		LOGI("OnMsg: Encoder %s", msg);
		if (nResult == RT_FAIL)
		{
			if (stopOutputRtmp() < 0)
			{
                if (m_is_reuse_encoder)
                {
                    if (AVStreamerNew::instance_ref_count < 1)
                    {
                        stopEncoder(m_encoder);
                        m_encoder = NULL;
                    }
                }
                else
                {
                    stopEncoder(m_encoder);
                    m_encoder = NULL;
                }
			}
		}
		break;
	default:
		LOGI("OnMsg: default %s", msg);
		break;
	}

    m_pCallback->OnMsg(nType, nResult, msg);
}


void AVOutputManager::doStopOutputRtmp()
{
    //stop rtmp streamer
    if (m_encoder != NULL)
    {
        m_encoder->unregisterEncodedPacketSink(m_streamer_new_rtmp);
    }
    delete m_streamer_new_rtmp;
    m_streamer_new_rtmp = NULL;
    //stop encoder
    if (m_is_reuse_encoder)
    {
        if (AVStreamerNew::instance_ref_count < 1)
        {
            stopEncoder(m_encoder);
            m_encoder = NULL;
        }
    }
    else
    {
        stopEncoder(m_encoder);
        m_encoder = NULL;
    }
    //stop output manager

            wait();
			//PostProcManager::GetInstance()->unregisterVideoFrameSink(this);
            m_ssg._VideoCapture_->unregisterVideoFrameSink(this);
            stop();
}



int AVOutputManager::stopUdpClient()
{
   return _UDPCLIENT::GetInstance()->stop_client();
}

int AVOutputManager::startUdpClient(char *server_ip, unsigned short port)
{
   return _UDPCLIENT::GetInstance()->start_client(server_ip, port);//server ip and port
}



void AVOutputManager::startOutputRtmp(VrType vrType, const std::string& url, StreamOpt streamOpt, VROutputCallback *pCallback)
{
	printf("startOutputRtmp: state %d, url %s", m_output_state_rtmp, url.c_str());
    if (m_output_state_rtmp > STATUS_IDLE && m_output_state_rtmp < STATUS_STOP) return;
    m_output_state_rtmp = STATUS_PREPARING;
    m_VRType = vrType;
    m_pCallback = pCallback;

    if (m_encoder == NULL) {
        //new rtmp encoder
        m_encoder = new AVEncoder(ST_RTMP);
        m_encoder->initVideoParams(streamOpt.vb);
        m_encoder->initAudioParams(streamOpt.ab);
        m_encoder->setLive(true);
        m_encoder->setAudioDelay(streamOpt.adms);

        registerAndGetMixType();

			registerVideoFrameSink(m_encoder);
        registerPCMFrameSink(m_encoder);
        //start rtmp encoder
        if (int ret = m_encoder->start(this) != 0)
        {
            printf("startOutputRtmp: start encoder fail ret %d", ret);
            return;
        }
    }
    if (m_encoder != NULL)
    {
        //new rtmp streamer
        m_streamer_new_rtmp = new AVStreamerNew(url, m_encoder);
        m_encoder->registerEncodedPacketSink(m_streamer_new_rtmp);
        //start rtmp streamer
        m_output_state_rtmp = STATUS_PREPARED;
        m_streamer_new_rtmp->start(this);
    }
	printf("startOutputRtmp: end state %d", m_output_state_rtmp);
}


//stop rtmp streamer
int AVOutputManager::stopOutputRtmp()
{
    if (m_output_state_rtmp >= STATUS_STOP) return -1;
    m_output_state_rtmp = STATUS_STOP;
    if (m_streamer_new_rtmp != NULL && m_streamer_new_rtmp->stop())
    {
        if (m_streamer_new_rtmp != NULL)
            m_encoder->unregisterEncodedPacketSink(m_streamer_new_rtmp);
        delete m_streamer_new_rtmp;
    }
	return 0;
}


void AVOutputManager::setFrameSourceGroup(StreamSourceGroup & ssg)
{
    m_ssg = ssg;
}