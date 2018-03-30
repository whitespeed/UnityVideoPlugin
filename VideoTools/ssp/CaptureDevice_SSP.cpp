#include "CaptureDevice_SSP.h"

#define TAG "CaptureDevice_SSP"

char *dup_wchar_to_utf8(const wchar_t *w)
{
	char *s = NULL;
	int l = WideCharToMultiByte(CP_UTF8, 0, w, -1, 0, 0, 0, 0);
	s = (char *)av_malloc(l);
	if (s)
		WideCharToMultiByte(CP_UTF8, 0, w, -1, s, l, 0, 0);
	return s;
}

CaptureDevice_SSP::CaptureDevice_SSP()
{
	m_StreamVideoIndex = -1;
	m_StreamAudioIndex = -1;
	m_brun = false;
	m_pCodecContext = NULL;
	m_pFormatCtx = NULL;
	m_Eventhandle = ::CreateEvent(NULL, FALSE, FALSE, NULL);
	m_lastTick = 0;
	m_pCodecParam = 0;
	m_pCodec = NULL;
	m_pARGB32VideoData = NULL;
	m_fps = 0;
}
void logFF(void *, int level, const char *fmt, va_list ap)
{
	printf(fmt, ap);
}

CaptureDevice_SSP::~CaptureDevice_SSP()
{
	m_StreamVideoIndex = -1;
	m_StreamAudioIndex = -1;
	m_brun = false;
	if (m_pFormatCtx)
	{
		avformat_free_context(m_pFormatCtx);
		m_pFormatCtx = NULL;
	}
	if (m_pCodecContext)
	{
		avcodec_close(m_pCodecContext);
		avcodec_free_context(&m_pCodecContext);
	}
	if (m_pCodecParam)
	{
		avcodec_parameters_free(&m_pCodecParam);
		m_pCodecContext = NULL;
	}
	m_pCodec = NULL;
	m_pCodecContext = NULL;
	m_pFormatCtx = NULL;
	CloseHandle(m_Eventhandle);
}
bool CaptureDevice_SSP::SetSync(bool bSync)
{
	m_bSync = bSync;
	return true;
}
CaptureDeviceType CaptureDevice_SSP::GetDeviceType()
{
	return VDEVICE_NETWORK;
}
char * CaptureDevice_SSP::GetDeviceName()
{
	return (char *)m_devicename.c_str();
}
void CaptureDevice_SSP::SetParam(void *lp)
{
	if (m_bAudio)
	{
		CaptureAudioParam *pcap = (CaptureAudioParam *)lp;
		m_cap.nchannel = pcap->nchannel;
		m_cap.sample_fmt = pcap->sample_fmt;
		m_cap.sample_rate = pcap->sample_rate;
	}
	else
	{
		CaptureVideoParam *pcvp = (CaptureVideoParam *)lp;
		m_cvp.fps = pcvp->fps;
		m_cvp.height = pcvp->height;
		m_cvp.width = pcvp->width;
		m_cvp.pix_fmt = pcvp->pix_fmt;
		m_cvp.p_videobuffer = pcvp->p_videobuffer;
		m_cvp.buffer_len = pcvp->buffer_len;
	}
}
bool CaptureDevice_SSP::StartCapture()
{
	m_pCodec = NULL;
	if (m_pFormatCtx == NULL)
	{
		m_pFormatCtx = avformat_alloc_context();
		m_pCodec = avcodec_find_decoder_by_name("h264_cuvid");
		if (m_pCodec == NULL)
		{
			m_pCodec = avcodec_find_decoder_by_name("h264_qsv");
		}
		if (m_pCodec == NULL)
		{
			m_pCodec = avcodec_find_decoder(AV_CODEC_ID_H264);
		}

		m_pCodecParam = avcodec_parameters_alloc();
		if (m_pCodec == NULL) {
			LOGE("Codec not found.\n");
		}
		m_pFrame = av_frame_alloc();
		m_pOutFrame = av_frame_alloc();
	}
	m_brun = false;
	m_pLoop = new imf::ThreadLoop(std::bind(&CaptureDevice_SSP::CaptureThreadFun, this, _1));
	m_pLoop->start();
	return true;
}
bool CaptureDevice_SSP::StopCapture()
{
	m_brun = false;
	m_pClient->stop();
	Sleep(100);
	delete m_pLoop;
	m_pLoop = NULL;
	WaitForSingleObject(m_DecodeHandle, 10000);
	//avformat_free_context(m_pFormatCtx);
	//avcodec_parameters_free(&m_pCodecParam);
	//avcodec_close(m_pCodecContext);
	//avcodec_free_context(&m_pCodecContext);
	delete this;
	return true;
}
void CaptureDevice_SSP::SetAudioCaptureCallBack(AudioCapturePacketCallback *pCC)
{
    m_pACC = pCC;
}
void CaptureDevice_SSP::SetVideoCaptureCallBack(VideoCapturePacketCallback *pCC)
{
    m_pVCC = pCC;
}
bool CaptureDevice_SSP::SetCaptureDeviceName(wchar_t *pDeviceName, bool bAudio)
{
	m_bAudio = bAudio;
	wstring DeviceName = wstring(pDeviceName);
	m_devicename = string(dup_wchar_to_utf8(DeviceName.c_str()));
	return true;
}
void CaptureDevice_SSP::SetDeivceIndex(int nIndex)
{
	m_nDeviceIndex = nIndex;
}
void CaptureDevice_SSP::SetStreamIndex(int nIndex)
{
	m_nStreamIndex = nIndex;
}
void CaptureDevice_SSP::SetBufferGroup(BufferGroup *pBG)
{
	m_pBufferGroup = pBG;
}
void CaptureDevice_SSP::on_264(uint8_t * data, size_t len, uint64_t pts, uint32_t frm_no, uint32_t type)
{
	if (!m_brun)
	{
		return;
	}
	//if (GetTickCount() - m_lastTick >1000)
	//{
	//	printf("回调的帧率为 %d %d \n", m_nIndex, m_fps);
	//	m_lastTick = GetTickCount();
	//	m_fps = 0;
	//}
	//m_fps++;
	//if (m_nIndex == 0)
	//{
	//	printf("回调的帧率为 %d %d %lld\n", m_nIndex, GetTickCount() - m_lastTick, pts);
	//	m_lastTick = GetTickCount();
	//}
	if (m_listRecycleData.size() > 0)
	{
		h264_data * ph264 = m_listRecycleData.front();
		m_listRecycleData.pop();
		memcpy(ph264->pData, data, len);
		ph264->DataLen = (unsigned int)len;
		ph264->frm_no = frm_no;
		ph264->pts = pts;
		ph264->bnew = true;
		ph264->nType = type;
		m_listData.push(ph264);
	}
	else
	{
		LOGE("丢帧了 %d %d \n", m_nStreamIndex, frm_no);
	}
	//printf("当前帧号 %d %d\n", m_nIndex, frm_no);
	//SetEvent(m_Eventhandle);

	return;
}
void CaptureDevice_SSP::on_meta(SspVideoMeta*v, SspAudioMeta*a)
{
	if (!m_brun)
	{
		int ret;
		if (m_pCodecContext == NULL)
		{
			m_pCodecContext = avcodec_alloc_context3(m_pCodec);
			if (!m_bAudio)
			{
				m_pCodecContext->gop_size = v->gop;
				avcodec_get_context_defaults3(m_pCodecContext, m_pCodec);
				if (ret = avcodec_open2(m_pCodecContext, m_pCodec, NULL) < 0) {
					LOGE("Could not open codec. %s\n", get_error_text(ret).c_str());
				}
			}
		}
		m_brun = true;
		m_DecodeHandle = CreateThread(NULL, NULL, CaptureThread, this, NULL, NULL);
	}
}
void CaptureDevice_SSP::on_audio(uint8_t * data, size_t len, uint64_t pts)
{
}
void CaptureDevice_SSP::ondisconnect()
{
	LOGE("disconnect---------------------------- %d\n", m_nStreamIndex);
	if (m_brun)
	{
		m_brun = false;
		WaitForSingleObject(m_DecodeHandle, 10000);
		LOGI("the client attempts to reconnect continuously\n");
		m_pClient->start();
	}
}
void CaptureDevice_SSP::onbufferfull()
{
	LOGE("onbufferfull---------------------------- %d\n", m_nStreamIndex);
}

#define MAX_AUDIO_FRAME_SIZE 192000 // 1 second of 48khz 32bit audio  

void CaptureDevice_SSP::CaptureThreadFun(imf::Loop *loop)
{
	std::string ip = "192.168.1.102";//"172.16.152.33";
	ip = m_devicename.substr(6, m_devicename.length() - 6);
	m_pClient = new imf::SspClient(ip, loop, 0x400000);
	m_pClient->init();
	for (int i = 0; i < 10; i++)
	{
		h264_data *ph264 = new h264_data();
		m_listRecycleData.push(ph264);
	}
	m_pClient->setOnH264DataCallback(std::bind(&CaptureDevice_SSP::on_264, this, _1, _2, _3, _4, _5));
	m_pClient->setOnMetaCallback(std::bind(&CaptureDevice_SSP::on_meta, this, _1, _2));
	m_pClient->setOnAudioDataCallback(std::bind(&CaptureDevice_SSP::on_audio, this, _1, _2, _3));
	m_pClient->setOnRecvBufferFullCallback(std::bind(&CaptureDevice_SSP::onbufferfull, this));
	m_pClient->setOnDisconnectedCallback(std::bind(&CaptureDevice_SSP::ondisconnect, this));
	m_pClient->start();
}
DWORD WINAPI CaptureDevice_SSP::CaptureThread(LPVOID lp)
{
	CaptureDevice_SSP *pthis = (CaptureDevice_SSP *)lp;
	pthis->DecodeThread();
	return 0;
}
void CaptureDevice_SSP::DecodeThread()
{
	while (m_listData.size() < 10 && m_brun)
	{
		Sleep(1);
	}
	m_StartTime = 0;
	while (m_brun)
	{
		if (m_listData.size() < 1)
		{
			m_StartTime = 0;
			continue;
		}
		h264_data *ph264 = m_listData.front();

		int ret = 0;
		AVPacket	pkt;
		av_init_packet(&pkt);
		pkt.data = ph264->pData;
		pkt.size = ph264->DataLen;
		pkt.pts = ph264->pts;
		m_curdts = ph264->pts;
		m_listData.pop();
		m_listRecycleData.push(ph264);
		ret = avcodec_send_packet(m_pCodecContext, &pkt);
		if (ret < 0)
		{
			//printf("video avcodec_send_frame error %s \n", get_error_text(ret).c_str());
			return;
		}
		ph264->bnew = false;
		while (1)
		{
			ret = avcodec_receive_frame(m_pCodecContext, m_pFrame);
			if (ret == AVERROR(EAGAIN))
			{
				//printf("1video avcodec_receive_frame %s \n", get_error_text(ret).c_str());
				ret = 0;
				break;
			}
			if (ret < 0)
			{
				//printf("2video avcodec_receive_frame %s \n", get_error_text(ret).c_str());
				break;
			}
			if (ret == 0)
			{
				if (m_pCodecParam->format != m_pFrame->format)
				{
					//sws_freeContext(m_video_convert_ctx);
					//m_video_convert_ctx = sws_getContext(m_pFrame->width, m_pFrame->height, (AVPixelFormat)m_pFrame->format,
					//	m_cvp.width, m_cvp.height, (AVPixelFormat)m_pFrame->format, SWS_BICUBIC, NULL, NULL, NULL);
					//m_pCodecParam->format = m_pFrame->format;
					m_outsize = av_image_get_buffer_size(AV_PIX_FMT_ARGB, m_pFrame->width, m_pFrame->height, 1); //av_image_get_buffer_size((AVPixelFormat)m_pFrame->format, m_pFrame->width, m_pFrame->height, 1);
					if (m_pARGB32VideoData == NULL)
					{
						m_pARGB32VideoData = (char *)av_malloc(m_outsize);
						//cudaMallocHost(&m_pARGB32VideoData, m_outsize);
					}
					//av_image_fill_arrays(m_pOutFrame->data, m_pOutFrame->linesize, (uint8_t *)m_pARGB32VideoData, m_cvp.pix_fmt, m_pFrame->width, m_pFrame->height, 1);
				}
				libyuv::NV12ToARGB(m_pFrame->data[0], m_pFrame->linesize[0], m_pFrame->data[1], m_pFrame->linesize[1], (uint8_t *)m_pARGB32VideoData, m_pFrame->width * 4, m_pFrame->width, m_pFrame->height);
				//libyuv::I420ToABGR(m_pFrame->data[0], m_pFrame->linesize[0], m_pFrame->data[1], m_pFrame->linesize[1], m_pFrame->data[2], m_pFrame->linesize[2], (uint8_t *)m_pARGB32VideoData, m_pFrame->width* 4, m_pFrame->width, m_pFrame->height);
				//av_image_copy_to_buffer(m_cvp.p_videobuffer + m_nIndex*m_outsize, m_outsize, m_pFrame->data, m_pFrame->linesize, (AVPixelFormat)m_pFrame->format, m_pFrame->width, m_pFrame->height, 1);
				//av_frame_copy(m_pOutFrame, m_pFrame);
				if (m_bSync)
				{
					int ret = VideoSyncUtil::GetInstance()->SyncFrameTime(m_nStreamIndex, m_pFrame->pts);
					if (ret > 0)
					{
						VideoSyncUtil::GetInstance()->Wait();
					}
					else if (ret == -1)
					{
						continue;
					}
					else if (ret == 0)
					{
						//同步完成
					}
				}
				CUdeviceptr ptr = m_pBufferGroup->GetPtrByIdx(m_nStreamIndex);
				cudaError_t st = cudaMemcpy((void*)ptr, m_pARGB32VideoData, m_outsize, cudaMemcpyHostToDevice);

				VideoCapturePacket vcp;
				vcp.pts = m_pFrame->pts / m_cvp.fps;
				vcp.nIndex = m_nStreamIndex;
				vcp.pData = NULL;
				vcp.data_len = m_outsize;
				vcp.width = m_pFrame->width;
				vcp.height = m_pFrame->height;
				vcp.pDevData = 0;
				vcp.bCopy = false;
				vcp.pix_fmt = AV_PIX_FMT_ARGB;// (AVPixelFormat)m_pFrame->format;
				//printf("video%d w %d h %d \n",m_nIndex, m_pFrame->width, m_pFrame->height);
                if(m_pVCC)
                    m_pVCC->onHandleVideoPackets(&vcp, this);
				m_pBufferGroup->UpdateDataByIndex(m_nStreamIndex, m_outsize, m_pFrame->pts);

				if (m_nStreamIndex == 0)
				{
					LOGI("视频帧的间隔为 %d %d \n", m_nStreamIndex, GetTickCount() - m_lastTick);
					m_lastTick = GetTickCount();
				}

				if (m_StartTime == 0)
				{
					m_StartTime = vcp.pts;
					m_StartCount = GetTickCount();
				}
				else
				{
					long offset = vcp.pts - m_StartTime + m_StartCount - GetTickCount();
					if (offset > 0 && offset < 100)
					{
						WaitForSingleObject(m_Eventhandle, offset);
					}
				}
			}
		}
	}
}