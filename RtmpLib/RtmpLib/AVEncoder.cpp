#include <stdio.h>
#include <math.h>

#include "AVEncoder.h"
#define TAG "AVEncoder"
#include "P2PLog.h"
#include "MessageDelivery.h"
#include "cuda_runtime.h"
#include "libyuv.h"
#include "FPSCounter.h"
#include "BPSCounter.h"
using namespace libyuv;

AVEncoder::AVEncoder(StreamType streamType) {
    LOGI("AVEncoder");
    av_register_all();
    avformat_network_init();


    pthread_mutex_init(&audioQueueMutex, NULL);

    video_stream_index = 0;
    audio_stream_index = 1;

    mFPSCounter = new FPSCounter();
    mFPSCounter->setInterval(1000);

    mBPSCounter = new BPSCounter();
    mBPSCounter->setInterval(1000);

    m_streamType = streamType;
}

AVEncoder::~AVEncoder()
{
    if (fifo)
        av_audio_fifo_free(fifo);

    if (pVOCodecCtx)
        avcodec_close(pVOCodecCtx);

    if (pAOCodecCtx)
        avcodec_close(pAOCodecCtx);

    if (vpool != NULL)
    {
        delete vpool;
    }
    pthread_mutex_destroy(&audioQueueMutex);
    if (mFPSCounter)
    {
        delete mFPSCounter;
    }

    LOGI("~AVEncoder");
}

int AVEncoder::start(VROutputCallback *pCallback)
{
    stopFlag = false;
    m_pCallback = pCallback;
    LOGD("AVEncoder::start !");
    int ret = pthread_create(&encodeThread, NULL, run, this);
    if (ret != 0)
    {
		//MessageDelivery::sendMessage(EventEncodeFailed, ret);
		OnMsg(OT_ENCODER, RT_FAIL, "Failed");
        return ret;
    }
    return 0;
}

int AVEncoder::stop()
{
    //MessageDelivery::sendMessage(EventEncodeEnd, 0);
	OnMsg(OT_ENCODER, RT_END, "End");
    return 0;
}

int AVEncoder::wait()
{
    stopFlag = true;
    pthread_join(encodeThread, NULL);
    return 0;
}

int AVEncoder::initVideoParams(int64_t bit_rate)
{
    this->video_bit_rate = bit_rate;
    this->video_width = 0;
    this->video_height = 0;
    return 0;
}

int AVEncoder::initVideoParams(int64_t bit_rate, int width, int height)
{
    this->video_width = width;
    this->video_height = height;
    this->video_bit_rate = bit_rate;
    return 0;
}

int AVEncoder::initAudioParams(int64_t bit_rate)
{
    this->audio_bit_rate = bit_rate;
    return 0;
}

int AVEncoder::check_sample_fmt(AVCodec *codec, enum AVSampleFormat sample_fmt)
{
    const enum AVSampleFormat *p = codec->sample_fmts;
    while (*p != AV_SAMPLE_FMT_NONE)
    {
        if (*p == sample_fmt)
            return 1;
        else
            LOGD("AVSampleFormat is %d ", *p);
        p++;
    }
    return 0;
}

int AVEncoder::select_sample_rate(AVCodec *codec)
{
    const int *p;
    int best_samplerate = 0;

    if (!codec->supported_samplerates)
        return OUTPUT_AUDIO_SAMPLE_RATE;

    p = codec->supported_samplerates;
    while (*p)
    {
        best_samplerate = FFMAX(*p, best_samplerate);
        p++;
    }
    return best_samplerate;
}

int64_t AVEncoder::select_channel_layout(AVCodec *codec)
{
    const uint64_t *p;
    uint64_t best_ch_layout = 0;
    int best_nb_channels = 0;

    if (!codec->channel_layouts)
        return AV_CH_LAYOUT_STEREO;

    p = codec->channel_layouts;
    while (*p)
    {
        int nb_channels = av_get_channel_layout_nb_channels(*p);

        if (nb_channels > best_nb_channels)
        {
            best_ch_layout = *p;
            best_nb_channels = nb_channels;
        }
        p++;
    }
    return best_ch_layout;
}

void* AVEncoder::run(void* data) {
    LOGD("AVEncoder::run !");
    AVEncoder* encoder = (AVEncoder*)data;
    encoder->doEncode();
    LOGD("AVEncoder::run out!");
    return 0;
}

void AVEncoder::doEncode()
{
    bool audioEof = false;
    bool videoEOf = false;
    int ret = 0;
    //MessageDelivery::sendMessage(EventEncodePrepare, 0);
    while (!videoEncodeInit || !audioEncodeInit)
    {
        if (stopFlag)
            return;
        av_usleep(10000);
    }
    int64_t startTime = av_gettime();

    LOGI("doEncode start, video_bit_rate %d, audio_bit_rate %d", video_bit_rate, audio_bit_rate);
    encode_start = true;

    //FIX ME:
	//MessageDelivery::sendMessage(EventEncodeStart, ret);
	OnMsg(OT_ENCODER, RT_OK, "doEncode start");
    while (!stopFlag)
    {
        if (audioEof && videoEOf)
            break;

        ///LOGV("audio_next_pts %d video_next_pts %d", audio_next_pts, video_next_pts);
        if (av_compare_ts(audio_next_pts, pAOCodecCtx->time_base, video_next_pts, pVOCodecCtx->time_base) <= 0 && audioEof == false)
        {
            //audio timestamp is before video
            bool audio_get = false;
            pthread_mutex_lock(&audioQueueMutex);
            if (av_audio_fifo_size(fifo) > audio_frame->nb_samples) {
                av_audio_fifo_read(fifo, (void **)audio_frame->data, audio_frame->nb_samples);
                audio_next_pts += audio_frame->nb_samples;
                audio_frame->pts = audio_next_pts;
                audio_get = true;
            }
            pthread_mutex_unlock(&audioQueueMutex);

            if (audio_get)
            {
                ret = encAudio(audio_frame);
                if (ret < 0)
                {
                    break;
                }
            }
            else
            {
                av_usleep(1000);
                if (liveMode == false)
                {
                    audioEof = true;
                }
            }
        }
        else
        {
            VideoFrame* video = vpool->popValid();
            bool gotFrame = video == nullptr ? false : true;

            if (gotFrame)
            {
                av_frame_make_writable(video_frame);
                //自适应pix_fmt:
                //if (video->pix_fmt == AV_PIX_FMT_ARGB) {
                //libyuv:ARGBToI420(video->data[0], video->width * 4, video_frame->data[0], video->width, video_frame->data[1], video->width / 2, video_frame->data[2], video->width / 2, video->width, video->height);
                //}
                //else {
                //    //default:
                //    av_image_copy(video_frame->data, video_frame->linesize, (const uint8_t **)video->data, video->linesize, video->pix_fmt, video->width, video->height);
                //}
				av_image_copy(video_frame->data, video_frame->linesize, (const uint8_t **)video->data, video->linesize, video->pix_fmt, video->width, video->height);
                v_timestamp = video->pts;
                if (ptsFromPacket == mPtsStratage)
                {
                    //获取后处理过来pts ，timebase = 1/1000，需要转换成timebase= 2/50的情况
                    video_next_pts = (v_timestamp * pVOCodecCtx->time_base.den / video->timebase_den);
                }
                else if (ptsAutoGen == mPtsStratage)
                {
                    video_next_pts++;
                }
                vpool->pushEmpty(video);
                video_frame->pts = video_next_pts;
                if (packet_pre_timestamp >= video_next_pts)
                {
                    LOGD("Error packet_pre_timestamp, continue!!!");
                    continue;
                }
                packet_pre_timestamp = video_next_pts;
                //LOGI("v_timestamp = %lld,first_timestamp =%lld ,video_next_pts = %d", v_timestamp, first_timestamp, video_next_pts);
                ret = encVideo(video_frame);
                if (ret < 0)
                {
                    LOGE("encode videoframe error...");
                    break;
                }
            }
            else
            {
                av_usleep(1000);
                if (liveMode == false)
                {
                    videoEOf = true;
                }
            }
        }
        av_usleep(1000);
    }

    int64_t endTime = av_gettime();
    LOGI("End encoding cost time %lld ,ret = %d", endTime - startTime,ret);
}

/**
* Convert an error code into a text message.
* @param error Error code to be converted
* @return Corresponding error text (not thread-safe)
*/
std::string AVEncoder::get_error_text(const int error)
{
    char error_buffer[255];
    av_strerror(error, error_buffer, sizeof(error_buffer));
    return std::string(error_buffer);
}

int AVEncoder::encAudio(AVFrame * frame) {

    AVPacket output_packet;
    int ret = 0;

    av_init_packet(&output_packet);
    output_packet.data = NULL;
    output_packet.size = 0;

    ret = avcodec_send_frame(pAOCodecCtx, frame);
    if (ret < 0)
    {
        LOGE("audio avcodec_send_frame %s", get_error_text(ret).c_str());
        MessageDelivery::sendMessage(ErrorEncoderAudioEncode, 0);
        return ret;
    }

    while (1)
    {
        ret = avcodec_receive_packet(pAOCodecCtx, &output_packet);
        if (ret == AVERROR(EAGAIN))
        {
            ret = 0;
            break;
        }

        if (ret < 0)
        {
            LOGE("audio avcodec_receive_packet %s", get_error_text(ret).c_str());
            MessageDelivery::sendMessage(ErrorEncoderAudioEncode, 1);
            break;
        }

        /** Write one audio frame from the temporary packet to the output file. */
        if (ret == 0) {
            EncodedPacket vPacket;
            vPacket.stream_index = audio_stream_index;
            vPacket.time_base = pAOCodecCtx->time_base;
            av_copy_packet(&vPacket.pkt, &output_packet);
            writeEncodedPacket(&vPacket);
            av_packet_unref(&output_packet);
        }
    }

    return ret;
}

int AVEncoder::encVideo(AVFrame * frame) {

    AVPacket output_packet;
    int ret;

    av_init_packet(&output_packet);
    output_packet.data = NULL;
    output_packet.size = 0;

    /*ffmpeg 3.1.5*/
    ret = avcodec_send_frame(pVOCodecCtx, frame);
    if (ret < 0)
    {
        LOGE("video avcodec_send_frame %s", get_error_text(ret).c_str());
        MessageDelivery::sendMessage(ErrorEncoderVideoEncode, 0);
        return ret;
    }

    while (1)
    {
        ret = avcodec_receive_packet(pVOCodecCtx, &output_packet);
        if (ret == AVERROR(EAGAIN))
        {
            ret = 0;
            break;
        }

        if (ret < 0)
        {
            LOGE("video avcodec_receive_packet %s", get_error_text(ret).c_str());
            MessageDelivery::sendMessage(ErrorEncoderVideoEncode, 1);
            break;
        }

        /** Write one video frame from the temporary packet to the output file. */
        if (ret == 0)
        {
            if (mBPSCounter->NewPacket(output_packet.size)) {
                //LOGE("AVEncoder:encode video kbps %d", mBPSCounter->getkbps());
                if (m_streamType == ST_FILE)
                {
                    MessageDelivery::sendMessage(EventEncodeFileBPS, mBPSCounter->getkbps());
                }
                else if (m_streamType == ST_RTMP)
                {
                    MessageDelivery::sendMessage(EventEncodeRtmpBPS, mBPSCounter->getkbps());
                }
            }
            if (mFPSCounter->NewFrame()) {
                //LOGE("AVEncoder:video fps %.2f", mFPSCounter->getFps());
                if (m_streamType == ST_FILE)
                {
                    MessageDelivery::sendMessage(EventEncodeFileFPS, mFPSCounter->getFps() * 100);
                }
                else if (m_streamType == ST_RTMP)
                {
                    MessageDelivery::sendMessage(EventEncodeRtmpFPS, mFPSCounter->getFps() * 100);
                }
            }
            EncodedPacket vPacket;
            vPacket.stream_index = video_stream_index;
            vPacket.time_base = pVOCodecCtx->time_base;
            av_copy_packet(&vPacket.pkt, &output_packet);
            writeEncodedPacket(&vPacket);
            av_packet_unref(&output_packet);
        }
    }

    return ret;
}

int AVEncoder::onPCMFrame(PCMFrame* packet)
{
	//open audio encoder:
	if (in_audio_get == false)
	{
		in_audio_get = true;
		//int  in_audio_sample_rate = packet->nb_samples*av_get_bytes_per_sample(packet->sample_fmt);
		AVSampleFormat in_audio_sample_fmt = (AVSampleFormat)packet->sample_fmt;
		int64_t in_audio_channel_layout = av_get_default_channel_layout(packet->channels);

		//open audio encoder:
		if (openAudioEncoder() < 0)
			return -1;

		//重采样准备参数：
		if (m_channel_layout != in_audio_channel_layout || m_sample_fmt != in_audio_sample_fmt || m_sample_rate != packet->sample_rate)
		{
			LOGI("audio fomat incompatible income channel_layout %d sample_fmt %d sample_rate %d", in_audio_channel_layout, in_audio_sample_fmt, packet->sample_rate);
			LOGI("audio fomat incompatible out going channel_layout %d sample_fmt %d sample_rate %d", m_channel_layout, m_sample_fmt, m_sample_rate);
			m_pSwr_ctx = swr_alloc_set_opts(NULL, m_channel_layout, m_sample_fmt, m_sample_rate, in_audio_channel_layout, in_audio_sample_fmt, packet->sample_rate, 0, NULL);
			if (m_pSwr_ctx == NULL)
			{
				MessageDelivery::sendMessage(ErrorEncoderAudio, 0);
				OnMsg(OT_ENCODER, RT_FAIL, "swr_alloc_set_opts fail");
				return -1;
			}

			int ret = swr_init(m_pSwr_ctx);
			if (ret < 0)
			{
				MessageDelivery::sendMessage(ErrorEncoderAudio, 1);
				OnMsg(OT_ENCODER, RT_FAIL, "swr_init fail");
				return ret;
			}

			ret = av_samples_alloc_array_and_samples(&m_resample_buf, &m_resample_buf_size, 2, OUTPUT_AUDIO_SAMPLE_RATE, m_sample_fmt, 1);
			if (ret < 0)
			{
				LOGE("av_samples_alloc_array_and_samples resample_buf_size %d", m_resample_buf_size);
				MessageDelivery::sendMessage(ErrorEncoderAudio, 2);
				OnMsg(OT_ENCODER, RT_FAIL, " av_samples_alloc_array_and_samples fail");
				return ret;
			}
		}
	}
	if (encode_start == false)
	{
		LOGV("onPCMFrame drop");
		return 0;
	}
	if (videoDelay > 0)
	{
		videoDelay -= packet->nb_samples * 1000 / packet->sample_rate;
		return 0;
	}
    if (av_audio_fifo_size(fifo) > ENCODER_MAX_AUDIO_FIFO_SAMPLES) {
        LOGE("fifo size exceeds ENCODER_MAX_AUDIO_FIFO_SAMPLES(5S)");
        return 0;
    }
	if (packet->size)
	{
		//统一做处理：
		if (packet != NULL)
		{
			if (m_pSwr_ctx)
			{
				/* convert to destination format */
				int len = swr_convert(m_pSwr_ctx, m_resample_buf, m_resample_buf_size, (const uint8_t **)packet->data, packet->nb_samples);
				//printf("convert audio resample_buf_size %d ,len %d,dstpack->samplecount %d", m_resample_buf_size, len, packet->nb_samples);
				av_audio_fifo_write(fifo, (void**)m_resample_buf, len);
			}
			else
			{
				//printf("copy audio %d", dstpack->nb_samples);
				av_audio_fifo_write(fifo, (void**)packet->data, packet->nb_samples);
			}
		}
	}
	return 0;
}

int AVEncoder::onVideoFrame(VideoFrame *packet, VideoFrameSource* src)
{
    if (in_picture_get == false)
    {
        in_picture_get = true;
        in_picture_pix_fmt = AV_PIX_FMT_YUV420P;//AV_PIX_FMT_ARGB
        in_picture_width = packet->width;
        in_picture_height = packet->height;

        if (video_width <= 0 || video_height <= 0)
        {
            video_width = in_picture_width;
            video_height = in_picture_height;
        }

		if (openVideoEncoder() < 0)
			return -1;

        vpool = new VideoFramePool(INIT_VIDEO_QUEUE_SIZE, MAX_VIDEO_QUEUE_SIZE, in_picture_pix_fmt, in_picture_width, in_picture_height);
		LOGI("new vpool %p", vpool);
    }

    if (encode_start == false)
    {
        LOGV("onVideoFrame drop");
        return 0;
    }

    //for audio delay:
    if (audioDelay > 0)
    {
        audioDelay -= 1000 / DEST_FPS;
        return 0;
    }
    if (first_picture == false)
    {
        first_timestamp = packet->pts;
        first_picture = true;
    }
    VideoFrame* temp = vpool->popEmpty();
    if (temp == nullptr)
    {
        LOGI("AVEncoder: pop VideoFrame empty!!!");
        MessageDelivery::sendMessage(ErrorEncoderVideoQueue, 0);
        return 0;
    }
    if (packet->hw == CUDA)
    {
		cudaMemcpy(temp->data[0], packet->data[0], packet->size, cudaMemcpyDeviceToHost);
    }
    else
    {
		memcpy(temp->data[0], packet->data[0], packet->size);
    }
    //LOGI("debug: packet pts %lld in_picture_width = %d,in_picture_height = %d",packet->pts, in_picture_width, in_picture_height);
    temp->pts = packet->pts - first_timestamp;
    temp->hw = CPU;
    temp->pix_fmt = packet->pix_fmt;
    temp->timebase_num = packet->timebase_num;
    //FIX ME:
    temp->timebase_den = 1000;// packet->timebase_den;//timebase是ms单位
    temp->size = packet->size;
    temp->linesize[0] = packet->linesize[0];
    temp->nFrameNum = packet->nFrameNum;
    vpool->pushValid(temp);
    return 0;
}

int AVEncoder::openAudioEncoder()
{
	pAOCodec = avcodec_find_encoder(AV_CODEC_ID_AAC);
    if (!pAOCodec)
    {
        MessageDelivery::sendMessage(ErrorEncoderAudioInit, 0);
		OnMsg(OT_ENCODER, RT_FAIL, "can not found aac encoder");
        return -1;
    }

	pAOCodecCtx = avcodec_alloc_context3(pAOCodec);
    if (!pAOCodecCtx)
    {
        MessageDelivery::sendMessage(ErrorEncoderAudioInit, 1);
		OnMsg(OT_ENCODER, RT_FAIL, "can not allocate audio codec context");
        return -1;
    }

	m_sample_rate = OUTPUT_AUDIO_SAMPLE_RATE;
	m_sample_fmt = OUTPUT_AUDIO_SAMLE_FMT;
	m_channel_layout = OUTPUT_AUDIO_CHANNEL_LAYOUT;
	m_channels = av_get_channel_layout_nb_channels(m_channel_layout);

    pAOCodecCtx->sample_rate = m_sample_rate;
    pAOCodecCtx->sample_fmt = m_sample_fmt;
    pAOCodecCtx->channel_layout = m_channel_layout;
    pAOCodecCtx->channels = m_channels;
    pAOCodecCtx->bit_rate = audio_bit_rate;

    AVRational time_base = { 1, pAOCodecCtx->sample_rate };
    pAOCodecCtx->time_base = time_base;

    if (!check_sample_fmt(pAOCodec, pAOCodecCtx->sample_fmt))
    {
        LOGI("encoder does not support sample format %s ", av_get_sample_fmt_name(pAOCodecCtx->sample_fmt));
		OnMsg(OT_ENCODER, RT_FAIL, "encoder does not support sample format");
        return -1;
    }
    pAOCodecCtx->flags |= CODEC_FLAG_GLOBAL_HEADER;
    if (avcodec_open2(pAOCodecCtx, pAOCodec, 0) < 0)
    {
        MessageDelivery::sendMessage(ErrorEncoderAudioInit, 2);
		OnMsg(OT_ENCODER, RT_FAIL, "can not open audio codec");
        return -1;
    }

    //allocate a fifo for 100 seconds
    fifo = av_audio_fifo_alloc(pAOCodecCtx->sample_fmt, pAOCodecCtx->channels, pAOCodecCtx->sample_rate * ENCODER_MAX_AUDIO_FIFO_SECONDS);

    //allocate a frame to store audio data be encoded
    audio_frame = av_frame_alloc();
    audio_frame->nb_samples = pAOCodecCtx->frame_size;
    audio_frame->channel_layout = pAOCodecCtx->channel_layout;
    audio_frame->format = pAOCodecCtx->sample_fmt;
    audio_frame->sample_rate = pAOCodecCtx->sample_rate;
    if (av_frame_get_buffer(audio_frame, 1) < 0) {
        av_frame_free(&audio_frame);
		OnMsg(OT_ENCODER, RT_FAIL, "can not allocate audio AVFrame");
		return -1;
    }

    audioEncodeInit = true;
	LOGI("openAudioEncoder");
    return 0;
}

int AVEncoder::openVideoEncoder()
{
    int ret = 0;
    pVOCodec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!pVOCodec)
    {
        MessageDelivery::sendMessage(ErrorEncoderVideoInit, 0);
		OnMsg(OT_ENCODER, RT_FAIL, "can not find h264 encoder");
        return -1;
    }

    pVOCodecCtx = avcodec_alloc_context3(pVOCodec);
    if (!pVOCodecCtx)
    {
        MessageDelivery::sendMessage(ErrorEncoderVideoInit, 1);
		OnMsg(OT_ENCODER, RT_FAIL, "can not allocate video codec context");
        return -1;
    }
    pVOCodecCtx->codec_id = AV_CODEC_ID_H264;
    pVOCodecCtx->pix_fmt = AV_PIX_FMT_YUV420P;
    pVOCodecCtx->width = video_width;
    pVOCodecCtx->height = video_height;
    pVOCodecCtx->time_base.num = 1;
    pVOCodecCtx->time_base.den = 2 * DEST_FPS;
    pVOCodecCtx->ticks_per_frame = 2;//每个frame的timestamp的ticks数，time_base.num * ticks_per_frame/time_base.den = 1/framerate
    //control bitrate
    pVOCodecCtx->bit_rate = video_bit_rate;
    pVOCodecCtx->rc_max_rate = video_bit_rate*2;
    pVOCodecCtx->bit_rate_tolerance = 0; // 允许的误差，缺省值为1
    pVOCodecCtx->rc_buffer_size = video_bit_rate;

    //control player prepare speed
    pVOCodecCtx->rc_initial_buffer_occupancy = video_bit_rate * 0.5;
    pVOCodecCtx->gop_size = DEST_FPS;//12;

    pVOCodecCtx->qmin = 10;
    pVOCodecCtx->qmax = 51;
    pVOCodecCtx->max_b_frames = 2;
    pVOCodecCtx->thread_count = 12; // 8; //4 to 8 for encode speed

    AVDictionary *param = 0;
    av_dict_set(&param, "preset", "superfast", 0); //veryfast to superfast
    av_dict_set(&param, "tune", "zerolatency", 0);

    pVOCodecCtx->flags |= CODEC_FLAG_GLOBAL_HEADER;

    if (avcodec_open2(pVOCodecCtx, pVOCodec, &param) < 0)
    {
        MessageDelivery::sendMessage(ErrorEncoderVideoInit, 2);
		OnMsg(OT_ENCODER, RT_FAIL, "failed to open video encoder");
        return -1;
    }

    video_frame = av_frame_alloc();
    video_frame->format = pVOCodecCtx->pix_fmt;
    video_frame->width = pVOCodecCtx->width;
    video_frame->height = pVOCodecCtx->height;

    if (av_frame_get_buffer(video_frame, 32) < 0) {
        av_frame_free(&video_frame);
		OnMsg(OT_ENCODER, RT_FAIL, "can not allocate video AVFrame");
		return -1;
    }
    videoEncodeInit = true;
	LOGI("openVideoEncoder");
    return 0;
}

void AVEncoder::setLive(bool mode)
{
    liveMode = mode;
}

void AVEncoder::setAudioDelay(int millisecond)
{
    if (millisecond > 0)
        this->audioDelay = millisecond;
    else
        this->videoDelay = abs(millisecond);
}

void AVEncoder::setPtsStratage(PtsStratage stratage)
{
    mPtsStratage = stratage;
}

PtsStratage AVEncoder::getPtsStratage()
{
    return mPtsStratage;
}

void AVEncoder::OnMsg(OutputType nType, ResultType nResult, char *msg)
{
	if (m_pCallback) {
		m_pCallback->OnMsg(nType, nResult, msg);
	}
}