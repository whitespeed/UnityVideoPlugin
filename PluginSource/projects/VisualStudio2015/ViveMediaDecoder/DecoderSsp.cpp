#include "DecoderSsp.h"
#include "Logger.h"

using namespace std::placeholders;

DecoderSsp::DecoderSsp()
{
	mAVFormatContext = NULL;
	mVideoCodec = NULL;
	mAudioCodec = NULL;
	mVideoCodecContext = NULL;
	mAudioCodecContext = NULL;
	mSspClient = NULL;
	mThreadLooper = NULL;

	av_init_packet(&mPacket);
	mSwrContext = NULL;
	mVideoBuffMax = 12;
	mAudioBuffMax = 24;
	mQueueMaxSize = 25;
	memset(&mVideoInfo, 0, sizeof(VideoInfo));
	memset(&mAudioInfo, 0, sizeof(AudioInfo));
	memset(&mVideoMeta, 0, sizeof(imf::SspVideoMeta));
	memset(&mAudioMeta, 0, sizeof(imf::SspAudioMeta));
	memset(&mSSpMeta, 0, sizeof(imf::SspAudioMeta));
	mIsConnected = false;
	mIsInitialized = false;
	mIsAudioAllChEnabled = false;
	mUseTCP = false;
	mIsSeekToAny = false;
}

DecoderSsp::~DecoderSsp()
{
	destroy();
}

int DecoderSsp::initSwrContext()
{
	if (mAudioCodecContext == NULL) {
		LOG("Audio context is null. \n");
		return -1;
	}

	int errorCode = 0;
	int64_t inChannelLayout = av_get_default_channel_layout(mAudioCodecContext->channels);
	uint64_t outChannelLayout = mIsAudioAllChEnabled ? inChannelLayout : AV_CH_LAYOUT_STEREO;
	AVSampleFormat inSampleFormat = mAudioCodecContext->sample_fmt;
	AVSampleFormat outSampleFormat = AV_SAMPLE_FMT_FLT;
	int inSampleRate = mAudioCodecContext->sample_rate;
	int outSampleRate = inSampleRate;

	if (mSwrContext != NULL) {
		swr_close(mSwrContext);
		swr_free(&mSwrContext);
		mSwrContext = NULL;
	}

	mSwrContext = swr_alloc_set_opts(NULL,
		outChannelLayout, outSampleFormat, outSampleRate,
		inChannelLayout, inSampleFormat, inSampleRate,
		0, NULL);


	if (swr_is_initialized(mSwrContext) == 0) {
		errorCode = swr_init(mSwrContext);
	}

	//	Save the output audio format
	mAudioInfo.channels = av_get_channel_layout_nb_channels(outChannelLayout);
	mAudioInfo.sampleRate = outSampleRate;
	//mAudioInfo.totalTime = mAudioStream->duration <= 0 ? (double)(mAVFormatContext->duration) / AV_TIME_BASE : mAudioStream->duration * av_q2d(mAudioStream->time_base);

	return errorCode;
}

void DecoderSsp::updateBufferState()
{
	if (mVideoInfo.isEnabled) {
		if (mVideoFrames.size() >= mVideoBuffMax) {
			mVideoInfo.bufferState = BufferState::FULL;
		}
		else if (mVideoFrames.size() == 0) {
			mVideoInfo.bufferState = BufferState::EMPTY;
		}
		else {
			mVideoInfo.bufferState = BufferState::NORMAL;
		}
	}

	if (mAudioInfo.isEnabled) {
		if (mAudioFrames.size() >= mAudioBuffMax) {
			mAudioInfo.bufferState = BufferState::FULL;
		}
		else if (mAudioFrames.size() == 0) {
			mAudioInfo.bufferState = BufferState::EMPTY;
		}
		else {
			mAudioInfo.bufferState = BufferState::NORMAL;
		}
	}
}

bool DecoderSsp::isBuffBlocked()
{
	bool ret = false;
	if (mVideoInfo.isEnabled && mVideoFrames.size() >= mVideoBuffMax) {
		ret = true;
	}

	if (mAudioInfo.isEnabled && mAudioFrames.size() >= mAudioBuffMax) {
		ret = true;
	}

	return ret;
}

void DecoderSsp::freeFrontFrame(std::list<AVFrame*>* frameBuff, std::mutex * mutex)
{
	std::lock_guard<std::mutex> lock(*mutex);
	if (!mIsInitialized || frameBuff->size() == 0) {
		LOG("Not initialized or buffer empty. \n");
		return;
	}

	AVFrame* frame = frameBuff->front();
	av_frame_free(&frame);
	frameBuff->pop_front();
	updateBufferState();
}

void DecoderSsp::flushBuffer(std::list<AVFrame*>* frameBuff, std::mutex * mutex)
{
	std::lock_guard<std::mutex> lock(*mutex);
	while (!frameBuff->empty()) {
		av_frame_free(&(frameBuff->front()));
		frameBuff->pop_front();
	}
}

int DecoderSsp::loadConfig()
{
	return 0;
}

void DecoderSsp::printErrorMsg(int errorCode)
{
	char msg[500];
	av_strerror(errorCode, msg, sizeof(msg));
	LOG("Decoder Ssp Error: %s \n", msg);
}

void DecoderSsp::setup(imf::Loop *loop, const char* url)
{
	std::string ip(url);
	mSspClient = new imf::SspClient(ip, loop, 0x400000);
	mSspClient->init();
	mSspClient->setOnH264DataCallback(std::bind(&DecoderSsp::on_264, this, _1, _2, _3, _4, _5));
	mSspClient->setOnMetaCallback(std::bind(&DecoderSsp::on_meta, this, _1, _2, _3));
	mSspClient->setOnDisconnectedCallback(std::bind(&DecoderSsp::on_disconnect, this));
	mSspClient->start();
}

bool DecoderSsp::init(const char* filePath)
{
	if (mIsInitialized) {
		LOG("Decoder has been init. \n");
		return true;
	}

	if (filePath == NULL) {
		LOG("File path is NULL. \n");
		return false;
	}

	av_register_all();
	av_log_set_level(AV_LOG_DEBUG);

	if (NULL == mAVFormatContext) {
		mAVFormatContext = avformat_alloc_context();
	}

	if (mVideoCodec == NULL) {
		mVideoCodec = avcodec_find_decoder(AV_CODEC_ID_H264);
	}
	if (NULL == mVideoCodec) {
		mVideoCodec = avcodec_find_decoder_by_name("h264_qsv");
	}
	if (NULL == mVideoCodec) {
		mVideoCodec = avcodec_find_decoder_by_name("h264_cuvid");
	}
	if (mVideoCodec == NULL) {
		LOG("Could not open video h264 codec. \n");
		return false;
	}

	mVideoCodecContext = avcodec_alloc_context3(mVideoCodec);
	avcodec_get_context_defaults3(mVideoCodecContext, mVideoCodec);
	int errorCode = avcodec_open2(mVideoCodecContext, mVideoCodec, NULL);
	if (errorCode < 0) {
		LOG("Could not open  h264 codec.");
	}
	mThreadLooper = new imf::ThreadLoop(std::bind(&DecoderSsp::setup, this, _1, filePath));
	mThreadLooper->start();
	mVideoInfo.isEnabled = true;
	mAudioInfo.isEnabled = false;
	mIsInitialized = true;
	return true;
}

bool DecoderSsp::decode()
{
	if (!mIsInitialized) {
		LOG("Not initialized. \n");
		return false;
	}
	if (isH264QueueReady() && !isBuffBlocked())
	{
		H264Data* h264Data = mH264Queue.dequeue();
		if (NULL == h264Data)
		{
			LOG("h264 queue is empty or used up, maybe network is unstable.");
			return true;
		}
		av_init_packet(&mPacket);
		mPacket.data = h264Data->data;
		mPacket.pts = h264Data->frm_no;
		mPacket.size = h264Data->len;
		int errorCode = avcodec_send_packet(mVideoCodecContext, &mPacket);
		if (errorCode != 0)
		{
			printErrorMsg(errorCode);
		}
		while (true)
		{
			AVFrame* frameDecoded = av_frame_alloc();
			errorCode = avcodec_receive_frame(mVideoCodecContext, frameDecoded);
			if (errorCode != 0 && frameDecoded->format != AV_PIX_FMT_NV12)
			{
				av_frame_free(&frameDecoded);
				break;
			}
			else
			{
				//pgm_save(frameDecoded,frameDecoded->width, frameDecoded->height);
				pushVideoFrame(frameDecoded);
			}
		}
		release_h264_data(h264Data);
	}
	if (isBuffBlocked())
	{
		LOG("Video frame buffer is full, checkout the performence of player");
	}
	updateBufferState();
	return true;
}

void DecoderSsp::pushVideoFrame(AVFrame* frameDecoded)
{
	std::lock_guard<std::mutex> lock(mVideoMutex);
	mVideoFrames.push_back(frameDecoded);
	//LOG("Push video frame: %d %f", mVideoFrames.size(), (double)frameDecoded->pts / (double)mVideoMeta.timescale * (double)mVideoMeta.unit);
}

bool DecoderSsp::isH264QueueReady()
{
	return mH264Queue.size() > 0;
}

void DecoderSsp::seek(double time)
{

}

void DecoderSsp::destroy()
{
	if (mSspClient != NULL)
	{
		mSspClient->stop();
		mSspClient->setOnH264DataCallback(NULL);
		mSspClient->setOnDisconnectedCallback(NULL);
		mSspClient->setOnMetaCallback(NULL);
		mSspClient->setOnRecvBufferFullCallback(NULL);
		mSspClient->setOnAudioDataCallback(NULL);
		mSspClient->setOnExceptionCallback(NULL);
	}
	if (mThreadLooper != NULL)
	{
		mThreadLooper->stop();
		delete mThreadLooper;
		mThreadLooper = NULL;
	}
	if (mSspClient != NULL)
	{
		delete mSspClient;
		mSspClient = NULL;
	}
	if (mVideoCodecContext != NULL) {
		avcodec_close(mVideoCodecContext);
		mVideoCodecContext = NULL;
	}
	if (mAudioCodecContext != NULL) {
		avcodec_close(mAudioCodecContext);
		mAudioCodecContext = NULL;
	}
	if (mAVFormatContext != NULL) {
		avformat_close_input(&mAVFormatContext);
		avformat_free_context(mAVFormatContext);
		mAVFormatContext = NULL;
	}
	if (mSwrContext != NULL) {
		swr_close(mSwrContext);
		swr_free(&mSwrContext);
		mSwrContext = NULL;
	}
	flushBuffer(&mVideoFrames, &mVideoMutex);
	flushBuffer(&mAudioFrames, &mAudioMutex);
	mH264Queue.release();
}

IDecoder::VideoInfo DecoderSsp::getVideoInfo()
{
	return mVideoInfo;
}

IDecoder::AudioInfo DecoderSsp::getAudioInfo()
{
	return mAudioInfo;
}

void DecoderSsp::setVideoEnable(bool isEnable)
{
	mVideoInfo.isEnabled = isEnable;
}

void DecoderSsp::setAudioEnable(bool isEnable)
{
	mAudioInfo.isEnabled = isEnable;
}

void DecoderSsp::setAudioAllChDataEnable(bool isEnable)
{
	mIsAudioAllChEnabled = isEnable;
	initSwrContext();
}

double DecoderSsp::getVideoFrame(unsigned char** outputY, unsigned char** outputU, unsigned char** outputV)
{
	std::lock_guard<std::mutex> lock(mVideoMutex);

	if (!mIsInitialized || mVideoFrames.size() == 0) {
		LOG("Video frame not available. ");
		*outputY = *outputU = *outputV = NULL;
		return -1;
	}
	AVFrame* frame = mVideoFrames.front();
	*outputY = frame->data[0];
	*outputU = frame->data[1];
	*outputV = frame->data[2];
	mVideoInfo.lastTime = (double)frame->pts / (double)mVideoMeta.timescale * (double)mVideoMeta.unit;
	//LOG("Play video frame: %d %f", mVideoFrames.size(), (double)frame->pts / (double)mVideoMeta.timescale * (double)mVideoMeta.unit);
	return  mVideoInfo.lastTime;
}

double DecoderSsp::getAudioFrame(unsigned char** outputFrame, int& frameSize)
{
	std::lock_guard<std::mutex> lock(mAudioMutex);
	if (!mIsInitialized || mAudioFrames.size() == 0) {
		LOG("Audio frame not available. ");
		*outputFrame = NULL;
		return -1;
	}

	AVFrame* frame = mAudioFrames.front();
	*outputFrame = frame->data[0];
	frameSize = frame->nb_samples;
	return 0;
}

void DecoderSsp::freeVideoFrame()
{
	freeFrontFrame(&mVideoFrames, &mVideoMutex);
	//LOG("Free oldest video frame and remain %d", mVideoFrames.size());
}

void DecoderSsp::freeAudioFrame()
{
	freeFrontFrame(&mAudioFrames, &mAudioMutex);
}

int DecoderSsp::getMetaData(char**& key, char**& value)
{
	if (!mIsInitialized || key != NULL || value != NULL) {
		return 0;
	}

	AVDictionaryEntry *tag = NULL;
	int metaCount = av_dict_count(mAVFormatContext->metadata);

	key = (char**)malloc(sizeof(char*) * metaCount);
	value = (char**)malloc(sizeof(char*) * metaCount);

	for (int i = 0; i < metaCount; i++) {
		tag = av_dict_get(mAVFormatContext->metadata, "", tag, AV_DICT_IGNORE_SUFFIX);
		key[i] = tag->key;
		value[i] = tag->value;
	}

	return metaCount;
}

void DecoderSsp::on_264(uint8_t * data, size_t len, uint64_t pts, uint32_t frm_no, uint32_t type)
{
	//LOG("On receive h264 data, cur queue size %d.", mH264Queue.size());
	if (mQueueMaxSize >mH264Queue.size())
	{
		H264Data* h264 = pack_h264_data(data, len, pts, frm_no, type);
		mH264Queue.queue(h264);
	}
	else
	{
		LOG("Drop frame, the decoder is too slow. Check out performence");
	}
}

void DecoderSsp::on_meta(struct imf::SspVideoMeta *v, struct imf::SspAudioMeta *a, struct imf::SspMeta *s)
{

	memcpy(&mVideoMeta, v, sizeof(imf::SspVideoMeta));
	mVideoInfo.width = mVideoMeta.width;
	mVideoInfo.height = mVideoMeta.height;
	mVideoInfo.bufferState = IDecoder::EMPTY;
	memcpy(&mAudioMeta, a, sizeof(imf::SspAudioMeta));
	mAudioInfo.channels = mAudioMeta.channel;
	mAudioInfo.sampleRate = mAudioMeta.sample_rate;
	mAudioInfo.bufferState = IDecoder::EMPTY;
	memcpy(&mSSpMeta, s, sizeof(imf::SspMeta));
	mVideoCodecContext->gop_size = mVideoMeta.gop;

}

void DecoderSsp::on_disconnect()
{
	LOG("on disconnet");
	//TODO: reconnect ssp server or push flush packet
}

