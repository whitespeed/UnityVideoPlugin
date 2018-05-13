#include "PushOutput.h"
#include <stdio.h>
#include "Logger.h"

#ifndef SAFE_DELETE
#define SAFE_DELETE(p) {if(p){delete(p); (p)=NULL;}}
#endif

PushOutput::PushOutput()
{
	mBufferSize = 4096;
}

PushOutput::~PushOutput()
{
	avformat_free_context(mFormatContext);
	av_free(mIOContext);
	av_free(mFormatContext);
	av_free(outFormat);
	self = NULL;
}

int PushOutput::InitAVFormatContext(char * stream)
{
	av_register_all();
	av_log_set_level(AV_LOG_DEBUG);
	mUrl.assign(stream);

	if (NULL == mFormatContext) {
		mFormatContext = avformat_alloc_context();
	}
	avformat_network_init();

	avformat_alloc_output_context2(&mFormatContext, NULL, NULL, stream);

	if (!mFormatContext) {
		LOG("Could not create output context\n");
		ret = AVERROR_UNKNOWN;
		return 0;
	}
	outFormat = mFormatContext->oformat;

	return 1;
}

int  PushOutput::DoOutput(AVFormatContext *ifmt_ctx)
{
	ret = avformat_write_header(mFormatContext, NULL);
	if (ret < 0) {
		LOG("Error occurred when opening output URL\n");
		return 0;
	}

	start_time = av_gettime();

	for (i = 0; i<ifmt_ctx->nb_streams; i++)
		if (ifmt_ctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO) {
			videoIndex = i;
			break;
		}

	av_dump_format(ifmt_ctx, 0, mUrl.data(), 0);

	while (1) {

		ret = av_read_frame(ifmt_ctx, &pkt);
		if (ret < 0)
		{
			LOG("Can't get the frame work pixel");
			break;
		}

		if (pkt.pts == AV_NOPTS_VALUE) {
			AVRational time_base1 = ifmt_ctx->streams[videoIndex]->time_base;
			//Duration between 2 frames (us)  
			int64_t calc_duration = (double)AV_TIME_BASE / av_q2d(ifmt_ctx->streams[videoIndex]->r_frame_rate);
			//Parameters  
			pkt.pts = (double)(frame_index*calc_duration) / (double)(av_q2d(time_base1)*AV_TIME_BASE);
			pkt.dts = pkt.pts;
			pkt.duration = (double)calc_duration / (double)(av_q2d(time_base1)*AV_TIME_BASE);
		}

		if (pkt.stream_index == videoIndex) {
			AVRational time_base = ifmt_ctx->streams[videoIndex]->time_base;
			AVRational time_base_q = { 1,AV_TIME_BASE };
			int64_t pts_time = av_rescale_q(pkt.dts, time_base, time_base_q);
			int64_t now_time = av_gettime() - start_time;
			if (pts_time > now_time)
				av_usleep(pts_time - now_time);
		}

		pkt.pos = -1;

		if (pkt.stream_index == videoIndex) {
			printf("Send %8d video frames to output URL\n", frame_index);
			frame_index++;
		}

		ret = av_interleaved_write_frame(mFormatContext, &pkt);

		if (ret < 0) {
			LOG("Error muxing packet\n");
			break;
		}

		av_free_packet(&pkt);
	}

	av_write_trailer(mFormatContext);
end:
	avformat_close_input(&ifmt_ctx);
	if (ret < 0 && ret != AVERROR_EOF) {
		printf("Error occurred.\n");
		return -1;
	}
	return 0;
}