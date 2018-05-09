#include "FFMpegDecoder.h"
#include "Logger.h"
#include "H264Queue.h"
#include "libavutil\imgutils.h"

using namespace std::placeholders;

FFMpegDecoder::FFMpegDecoder()
{
	mAVFormatContext = NULL;
	mVideoCodec = NULL;
	mAudioCodec = NULL;
	mVideoCodecContext = NULL;
	mAudioCodecContext = NULL;
	//mSspClient = NULL;
	//mThreadLooper = NULL;
	mSwsContext = NULL;
	av_init_packet(&mPacket);
	mSwrContext = NULL;
	//mVideoBuffMax = 12;
	//mAudioBuffMax = 24;
	//mQueueMaxSize = 25;
	memset(&mVideoInfo, 0, sizeof(VideoInfo));
	memset(&mAudioInfo, 0, sizeof(AudioInfo));
	//memset(&mVideoMeta, 0, sizeof(imf::SspVideoMeta));
	//memset(&mAudioMeta, 0, sizeof(imf::SspAudioMeta));
	//memset(&mSSpMeta, 0, sizeof(imf::SspAudioMeta));
	//mIsConnected = false;
	mIsInitialized = false;
	mIsAudioAllChEnabled = false;
	//mUseTCP = false;
	mIsSeekToAny = false;
}


AVFrame* FFMpegDecoder::convertToYUV420P(AVFrame* src)
{
	if (NULL == src)
		return NULL;
	AVFrame* dst = av_frame_alloc();
	int numBytes = avpicture_get_size(AV_PIX_FMT_YUV420P, src->width, src->height);
	uint8_t* buffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));
	avpicture_fill((AVPicture *)dst, buffer, AV_PIX_FMT_YUV420P, src->width, src->height);
	dst->format = AV_PIX_FMT_YUV420P;
	dst->width = src->width;
	dst->height = src->height;
	if (NULL == mSwsContext)
	{
		mSwsContext = sws_getContext(src->width, src->height, (AVPixelFormat)src->format, 
			dst->width, dst->height, AV_PIX_FMT_YUV420P, SWS_BICUBIC, NULL, NULL, NULL);
	}
	int result = sws_scale(mSwsContext, (const uint8_t * const*)src->data, src->linesize, 0, src->height, dst->data, dst->linesize);
	if (result < 0)
	{
		LOG("Convert frame format to YUV420P failed.\n");
		av_frame_free(&dst);
		dst = NULL;
	}
	return dst;
}


bool FFMpegDecoder::Decode(std::list<AVFrame*> &mVideoFrames, H264Queue &mH264Queue, const std::function<void(FFMpegDecoder::VideoInfo, std::list<AVFrame*> &mVideoFrames)> &decodeCallback,
	const bool &queueReady, const bool buffBlocked)
{
	if (!mIsInitialized)
	{
		LOG("Not Initialized. \n");
		return false;
	}

	if (queueReady && buffBlocked)
	{
		H264Data* h264Data = mH264Queue.dequeue();
		if (NULL == h264Data)
		{
			LOG("H264 queue is empty or used up, maybe network is unstable.");
			return true;
		}
		auto start = clock();
		av_init_packet(&mPacket);
		mPacket.data = h264Data->data;
		mPacket.size = h264Data->len;
		int errorCode = avcodec_send_packet(mVideoCodecContext, &mPacket);
		if (errorCode != 0)
		{
			printErrorMsg(errorCode);
		}

		int frameCount = 0;
		while(1)
		{
			AVFrame* frameDecoded = av_frame_alloc();
			errorCode = avcodec_receive_frame(mVideoCodecContext, frameDecoded);
			if (errorCode != 0)
			{
				av_frame_free(&frameDecoded);
				break;
			}
			else
			{
				if (frameDecoded->format != AV_PIX_FMT_YUV420P)
				{
					AVFrame *frameYUV = convertToYUV420P(frameDecoded);
					if (NULL != frameYUV)
					{
						av_frame_free(&frameDecoded);
						frameYUV->pkt_dts = ++mDtsIndex;
						pushVideoFrame(frameYUV, mVideoFrames);
					}
				}
				else
				{
					frameDecoded->pkt_dts = ++mDtsIndex;
					pushVideoFrame(frameDecoded, mVideoFrames);
				}
				if (frameCount > 0)
				{
					LOG("Decoder output more than 1 frame in 1 packet.\n");
				}
				frameCount++;
			}
		}
		release_h264_data(h264Data);
		auto delta = (double)(clock() - start) / CLOCKS_PER_SEC * 1000;
		LOG("Decoder cost time per frame: %f\n", delta);
	}
	if (buffBlocked)
	{
		//LOG("Video frames buffer is full.\n");
	}
	decodeCallback(mVideoInfo, mVideoFrames);
	return true;
}

bool FFMpegDecoder::Init(IStreamInput *IOInput)
{
	if (mIsInitialized) {
		LOG("Decoder has been init. \n");
		return true;
	}

	if (NULL == IOInput) {
		LOG("Stream Input is NULL. \n");
		return false;
	}

	av_register_all();
	av_log_set_level(AV_LOG_DEBUG);

	if (NULL == mVideoCodec) {
		mVideoCodec = avcodec_find_decoder_by_name("h264_cuvid");
	}

	if (mVideoCodec == NULL) {
		mVideoCodec = avcodec_find_decoder(AV_CODEC_ID_H264);
	}

	if (mVideoCodec == NULL) {
		LOG("Could not find any video h264 codecs. \n");
		return false;
	}

	mVideoCodecContext = avcodec_alloc_context3(mVideoCodec);
	avcodec_get_context_defaults3(mVideoCodecContext, mVideoCodec);
	int errorCode = avcodec_open2(mVideoCodecContext, mVideoCodec, NULL);

	if (errorCode < 0) {
		printErrorMsg(errorCode);
	}
	else
	{
		LOG("Init video codec: %s\n", mVideoCodec->long_name);
		LOG("Codec pixel-format: %s, color-space: %s, color-range: %s. \n", 
			av_get_pix_fmt_name(mVideoCodecContext->pix_fmt),
			av_get_colorspace_name(mVideoCodecContext->colorspace),
			av_color_range_name(mVideoCodecContext->color_range));
	}
	mVideoInfo.isEnabled = true;
	mAudioInfo.isEnabled = false;
	mDtsIndex = 0;
	mIsInitialized = true;
}

FFMpegDecoder::VideoInfo FFMpegDecoder::GetVideoInfo()
{
	return mVideoInfo;
}

FFMpegDecoder::AudioInfo FFMpegDecoder::GetAudioInfo()
{
	return mAudioInfo;
}

void FFMpegDecoder::pushVideoFrame(AVFrame* frameDecoded, std::list<AVFrame*> &mVideoFrames)
{
	std::lock_guard<std::mutex> lock(mVideoMutex);
	mVideoFrames.push_back(frameDecoded);
	//LOG("Push video frame: %d %f", mVideoFrames.size(), (double)frameDecoded->pts / (double)mVideoMeta.timescale * (double)mVideoMeta.unit);
}

