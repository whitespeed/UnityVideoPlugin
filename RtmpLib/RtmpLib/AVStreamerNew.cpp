#include "AVStreamerNew.h"
#define TAG "AVStreamerNew"
#include "P2PLog.h"
#include "MessageDelivery.h"
//#include "FPSCounter.h"

int AVStreamerNew::instance_ref_count = 0;

AVStreamerNew::AVStreamerNew(const std::string& url, AVEncoder *encoder)
{
	LOGI("AVStreamerNew");
	this->url = url;
	this->m_encoder = encoder;
	av_register_all();
	avformat_network_init();
	//m_output_type = encoder->m_streamType;
	if (url.compare(0, 7, "rtmp://") == 0) {
		m_output_type = OT_RTMP;
		if (avformat_alloc_output_context2(&ofmt_ctx, NULL, "flv", url.c_str()) < 0)
			LOGI("AVStreamerNew: avformat_alloc_output_context2 failed ");
	}

	pthread_mutex_init(&packetQueueMutex, NULL);
	AVStreamerNew::instance_ref_count++;

	//mFPSCounter = new FPSCounter();
	//mFPSCounter->setInterval(5000);
}

AVStreamerNew::~AVStreamerNew()
{
	if (ofmt_ctx && !(ofmt_ctx->oformat->flags & AVFMT_NOFILE))
		avio_closep(&ofmt_ctx->pb);
	avformat_free_context(ofmt_ctx);

	if (packetPool != NULL)
	{
		delete packetPool;
		packetPool = NULL;
	}
	//if (mFPSCounter)
	//{
	//	delete mFPSCounter;
	//}
	AVStreamerNew::instance_ref_count--;
	
	if (AVStreamerNew::instance_ref_count < 1)
	{
		m_encoder = NULL;
		AVStreamerNew::instance_ref_count = 0;
		LOGI("~AVStreamerNew The Last One");
	}
	else
	{
		LOGI("~AVStreamerNew");
	}
}

int AVStreamerNew::start(/*int mix_type, */VROutputCallback *pCallback)
{
	stopFlag = false;
	//g_mix_type = mix_type;
	m_pCallback = pCallback;
	int ret = pthread_create(&newStreamThread, NULL, run, this);
	if (ret != 0)
	{
		MessageDelivery::sendMessage(EventStreamFailed, 0);
		OnMsg(m_output_type, RT_FAIL, "Failed");
		return ret;
	}
	return 0;
}

int AVStreamerNew::stop()
{
	if (stopFlag)
	{
		if (!realstopFlag)
		{
			MessageDelivery::sendMessage(ErrorStreamIOStop, m_output_type);
			return 0;
		}
	}
	stopFlag = true;
	if (!realstopFlag)
	{
		return 0;
	}
	return 1;
}

//int AVStreamerNew::wait()
//{
//	stopFlag = true;
//	pthread_join(newStreamThread, NULL);
//	return 0;
//}

void* AVStreamerNew::run(void* data) {
	AVStreamerNew* avstream = (AVStreamerNew*)data;
	while (avstream->doStream() < 0 && avstream->url.compare(0, 7, "rtmp://") == 0)
	{
		//PingReply reply;
		//string strPing;
		////if (avstream->url.compare(0, 7, "rtmp://") == 0)
		////{
		//	if (!avstream->getDomain(avstream->url.c_str(), strPing))
		//	{
		//		LOGD("AVStreamerNew::run: get domain error!");
		//		return NULL;
		//	}
		////}
		//while (!avstream->ItemPing2.Ping(strPing.c_str(), &reply))
		//{
		//	if (avstream->stopFlag)
		//		break;
		//	Sleep(100);
		//}

		if (avstream->stopFlag)
			break;
		avstream->OnMsg(avstream->m_output_type, RT_RETRYING, "avio_open2 timeout(40000ns), try again!");
		//free out avformat context
		if (avstream->ofmt_ctx && !(avstream->ofmt_ctx->oformat->flags & AVFMT_NOFILE))
		{
			avio_closep(&avstream->ofmt_ctx->pb);
		}
		avformat_free_context(avstream->ofmt_ctx);

		//recreate out avformat context
		if (avstream->url.compare(0, 7, "rtmp://") == 0) {
			if (avformat_alloc_output_context2(&avstream->ofmt_ctx, NULL, "flv", avstream->url.c_str()) < 0)
				LOGD("AVStreamerNew: avformat_alloc_output_context2 failed ");
		}
		avstream->videoStreamInit = false;
		avstream->audioStreamInit = false;
		avstream->stream_start = false;
		avstream->in_packet_get = false;
	}

	//real stop
	if (avstream->stopFlag)
	{
		avstream->realstopFlag = true;
		avstream->OnMsg(avstream->m_output_type, RT_END, "End");
	}
	else
	{
		avstream->stopFlag = true;
	}
	return 0;
}

void log_packet2(const AVFormatContext *fmt_ctx, const AVPacket *pkt)
{
	AVRational *time_base = &fmt_ctx->streams[pkt->stream_index]->time_base;
	LOGV("pts:%ld dts:%ld duration:%ld time_base %d/%d size %d stream_index:%d",
		pkt->pts, pkt->dts, pkt->duration, time_base->num, time_base->den, pkt->size,
		pkt->stream_index);
}

static int write_frame(AVFormatContext *fmt_ctx, const AVRational *time_base, AVStream *st, AVPacket *pkt)
{
	/* rescale output packet timestamp values from codec to stream timebase */
	av_packet_rescale_ts(pkt, *time_base, st->time_base);
	pkt->stream_index = st->index;

	/* Write the compressed frame to the media file. */
	//log_packet2(fmt_ctx, pkt);
	return av_interleaved_write_frame(fmt_ctx, pkt);
}

int AVStreamerNew::doStream()
{
	bool audioEof = false;
	bool videoEOf = false;
	int ret = 0;
	MessageDelivery::sendMessage(EventStreamPrepare, 0);
	while (!videoStreamInit || !audioStreamInit)
	{
		if (stopFlag)
			return 0;
		av_usleep(10000);
	}
	int64_t startTime = av_gettime();
	AVDictionary *d = NULL;
	av_dict_set(&d, "rw_timeout", "1000000", 0);//设置read/write frame的超时时长4000ms
	if ((ret = avio_open2(&ofmt_ctx->pb, url.c_str(), AVIO_FLAG_READ_WRITE, NULL, &d)) < 0)
	{
		//if (m_time_retry == 0)
		//{
		//	m_time_retry = av_gettime();
		//}
		//else {
		//	if (av_gettime() - m_time_retry > 40000)
		//	{
		//		LOGE("AVStreamerNew: Failed to open output file %s!", url.c_str());
		//		MessageDelivery::sendMessage(ErrorStreamIOOpen, 0);
		//		OnMsg(m_output_type, RT_FAIL, "AVStreamerNew avio_open2 retry timeout(40000ns)");
		//		return 1;
		//	}
		//}
		MessageDelivery::sendMessage(ErrorStreamIOOpen, 0);
		LOGE("Failed to open output file %s!", url.c_str());
		return -1;
	}

	av_dump_format(ofmt_ctx, 0, url.c_str(), 1);
	stream_start = true;
	avformat_write_header(ofmt_ctx, NULL);
	
	MessageDelivery::sendMessage(EventStreamStart, ret);
	OnMsg(m_output_type, RT_OK, "doStream start");
	while (!stopFlag)
	{
		EncodedPacket outpacket;
		EncodedPacket* pPacket = packetPool->popValid();
		if (pPacket) {
			pthread_mutex_lock(&packetQueueMutex);
			outpacket.stream_index = pPacket->stream_index;
			outpacket.time_base = pPacket->time_base;
			av_copy_packet(&outpacket.pkt, &pPacket->pkt);
			pthread_mutex_unlock(&packetQueueMutex);

			//if (mFPSCounter->NewFrame()) {
			//	LOGE("doStream:video fps %.2f", mFPSCounter->getFps());
			//}
			if ((ret = write_frame(ofmt_ctx, &outpacket.time_base, ofmt_ctx->streams[outpacket.stream_index], &outpacket.pkt)) < 0) {
				LOGE("doStream:Could not write video packet (error '%s'), erro no %d", get_error_text(ret).c_str(), ret);
				MessageDelivery::sendMessage(ErrorStreamIOSend, ret);
				packetPool->pushEmpty(pPacket);
				av_write_trailer(ofmt_ctx);
				return -1;
			}
			packetPool->pushEmpty(pPacket);
		}
		else {
			av_usleep(10000);
		}
	}

	if (ret == 0)
	{
		MessageDelivery::sendMessage(EventStreamEnd, ret);
	}

	av_write_trailer(ofmt_ctx);
	int64_t endTime = av_gettime();
	LOGI("End stream cost time %lld ,ret = %d", endTime - startTime, ret);
	return 0;
}

/**
* Convert an error code into a text message.
* @param error Error code to be converted
* @return Corresponding error text (not thread-safe)
*/
std::string AVStreamerNew::get_error_text(const int error)
{
	char error_buffer[255];
	av_strerror(error, error_buffer, sizeof(error_buffer));
	return std::string(error_buffer);
}

int AVStreamerNew::onEncodedPacket(EncodedPacket *packet, EncodedPacketSource* src)
{
	uint64_t start = av_gettime();
	//LOGV("onEncodedPacket");
	if (in_packet_get == false)
	{
		in_packet_get = true;
		if (openVideoStream() < 0 || openAudioStream() < 0)
			return 0;
		LOGI("onEncodedPacket: open stream success");
		if (packetPool == NULL)
		{
			packetPool = new EncodedPacketPool(INIT_PACKET_QUEUE_SIZE, MAX_PACKET_QUEUE_SIZE);
		}
	}

	if (stream_start == false)
	{
		LOGV("onEncodedPacket drop");
		return 0;
	}

	//
	EncodedPacket* temp = packetPool->popEmpty();
	if (temp == nullptr)
	{
		LOGI("onEncodedPacket: pop video packet empty!!!");
		MessageDelivery::sendMessage(ErrorStreamVideoQueue, 0);
		return 0;
	}
	temp->stream_index = packet->stream_index;
	temp->time_base = packet->time_base;
	av_copy_packet(&temp->pkt, &packet->pkt);
	packetPool->pushValid(temp);
	return 0;
}

int AVStreamerNew::openAudioStream()
{
	if (m_encoder == NULL || m_encoder->pAOCodec == NULL || m_encoder->pAOCodecCtx == NULL) {
		LOGI("encoder or pAOCodec or pAOCodecCtx is null when open audio stream!!");
		return -1;
	}
	//open audio stream
	audio_st = avformat_new_stream(ofmt_ctx, m_encoder->pAOCodec);
	if (audio_st == NULL)
	{
		MessageDelivery::sendMessage(ErrorStreamAudioInit, 0);
		OnMsg(m_output_type, RT_FAIL, "failed to open audio stream");
		return -1;
	}
	audio_st->id = audio_st->index;
	LOGI("audio_st->index %d", audio_st->index);
	LOGI("ofmt_ctx  nb_streams %d", ofmt_ctx->nb_streams);

	//
	/*ffmpeg 3.1.5*/
	int ret = avcodec_parameters_from_context(audio_st->codecpar, m_encoder->pAOCodecCtx);
	if (ret < 0)
	{
		MessageDelivery::sendMessage(ErrorStreamAudioInit, 1);
		OnMsg(m_output_type, RT_FAIL, "audio codec failed to avcodec_parameters_to_context");
		return -1;
	}
	/*ffmpeg 3.1.5*/

	audioStreamInit = true;
	return 0;
}

int AVStreamerNew::openVideoStream()
{
	if (m_encoder == NULL || m_encoder->pVOCodec == NULL || m_encoder->pVOCodecCtx == NULL) {
		LOGI("encoder or pVOCodec or pVOCodecCtx is null when open video stream!!");
		return -1;
	}
	//open video stream
	video_st = avformat_new_stream(ofmt_ctx, m_encoder->pVOCodec);
	if (video_st == NULL)
	{
		MessageDelivery::sendMessage(ErrorStreamVideoInit, 0);
		OnMsg(m_output_type, RT_FAIL, "failed to open video stream");
		return -1;
	}
	video_st->id = video_st->index;
	LOGI("video_st->index %d ", video_st->index);
	LOGI("ofmt_ctx  nb_streams %d ", ofmt_ctx->nb_streams);

	int ret = avcodec_parameters_from_context(video_st->codecpar, m_encoder->pVOCodecCtx);
	if (ret < 0)
	{
		MessageDelivery::sendMessage(ErrorStreamVideoInit, 1);
		OnMsg(m_output_type, RT_FAIL, "video codec failed to avcodec_parameters_to_context");
		return -1;
	}
	videoStreamInit = true;

	return 0;
}

void AVStreamerNew::OnMsg(OutputType nType, ResultType nResult, char *msg)
{
	if (m_pCallback) {
		m_pCallback->OnMsg(nType, nResult, msg);
	}
}

//bool AVStreamerNew::getDomain(const char* url, string& domain)
//{
//	const char *tmpstr;
//	const char *curstr;
//	int len;
//	int i;
//	int userpass_flag;
//	int bracket_flag;
//
//	curstr = url;
//
//	/*
//	* <scheme>:<scheme-specific-part>
//	* <scheme> := [a-z\+\-\.]+
//	*             upper case = lower case for resiliency
//	*/
//	/* Read scheme */
//	tmpstr = strchr(curstr, ':');
//	if (NULL == tmpstr) {
//		/* Not found the character */
//		return false;
//	}
//
//	/* Skip ':' */
//	tmpstr++;
//	curstr = tmpstr;
//
//	/*
//	* //<user>:<password>@<host>:<port>/<url-path>
//	* Any ":", "@" and "/" must be encoded.
//	*/
//	/* Eat "//" */
//	for (i = 0; i < 2; i++) {
//		if ('/' != *curstr) {
//			return false;
//		}
//		curstr++;
//	}
//
//	/* Check if the user (and password) are specified. */
//	userpass_flag = 0;
//	tmpstr = curstr;
//	while ('\0' != *tmpstr) {
//		if ('@' == *tmpstr) {
//			/* Username and password are specified */
//			userpass_flag = 1;
//			break;
//		}
//		else if ('/' == *tmpstr) {
//			/* End of <host>:<port> specification */
//			userpass_flag = 0;
//			break;
//		}
//		tmpstr++;
//	}
//
//	/* User and password specification */
//	tmpstr = curstr;
//	if (userpass_flag) {
//		/* Read username */
//		while ('\0' != *tmpstr && ':' != *tmpstr && '@' != *tmpstr) {
//			tmpstr++;
//		}
//		/* Proceed current pointer */
//		curstr = tmpstr;
//		if (':' == *curstr) {
//			/* Skip ':' */
//			curstr++;
//			/* Read password */
//			tmpstr = curstr;
//			while ('\0' != *tmpstr && '@' != *tmpstr) {
//				tmpstr++;
//			}
//			curstr = tmpstr;
//		}
//		/* Skip '@' */
//		if ('@' != *curstr) {
//			return false;
//		}
//		curstr++;
//	}
//
//	if ('[' == *curstr) {
//		bracket_flag = 1;
//	}
//	else {
//		bracket_flag = 0;
//	}
//	/* Proceed on by delimiters with reading host */
//	tmpstr = curstr;
//	while ('\0' != *tmpstr) {
//		if (bracket_flag && ']' == *tmpstr) {
//			/* End of IPv6 address. */
//			tmpstr++;
//			break;
//		}
//		else if (!bracket_flag && (':' == *tmpstr || '/' == *tmpstr)) {
//			/* Port number is specified. */
//			break;
//		}
//		tmpstr++;
//	}
//	len = tmpstr - curstr;
//	domain.assign(curstr, len);
//	return true;
//}
