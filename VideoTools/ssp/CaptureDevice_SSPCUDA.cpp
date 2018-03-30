#include "CaptureDevice_SSPCUDA.h"

//#define SSP 1
#define TAG "CaptureDevice_SSPCUDA"
char *dup_wchar_to_utf8(const wchar_t *w)
{
    char *s = NULL;
    int l = WideCharToMultiByte(CP_UTF8, 0, w, -1, 0, 0, 0, 0);
    s = (char *)av_malloc(l);
    if (s)
        WideCharToMultiByte(CP_UTF8, 0, w, -1, s, l, 0, 0);
    return s;
}

CaptureDevice_SSPCUDA::CaptureDevice_SSPCUDA()
{
    m_StreamVideoIndex = -1;
    m_StreamAudioIndex = -1;
    m_brun = false;
    m_Eventhandle = ::CreateEvent(NULL, FALSE, FALSE, NULL);
    m_lastTick = 0;
    m_fps = 0;
    m_bSync = true;
    m_pLoop = NULL;
    m_video_h = 0;
    m_video_w = 0;
    m_videoParser = NULL;
    m_videoDec = NULL;
    m_CtxLock = NULL;
    m_oContext = NULL;
    m_ptcp = new tcp_connect();
    m_ptcp->Init();
    m_httpSetting = new CHttpClient();
	m_bResetRuning = false;
}

CaptureDevice_SSPCUDA::~CaptureDevice_SSPCUDA()
{
    m_StreamVideoIndex = -1;
    m_StreamAudioIndex = -1;
    m_brun = false;
    CloseHandle(m_Eventhandle);
    delete m_ptcp;
    m_ptcp = NULL;
}
CaptureDeviceType CaptureDevice_SSPCUDA::GetDeviceType()
{
    return VDEVICE_NETWORK;
}
char * CaptureDevice_SSPCUDA::GetDeviceName()
{
    return (char *)m_devicename.c_str();
}
void CaptureDevice_SSPCUDA::SetParam(void *lp)
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
        m_pBufferGroup = pcvp->pBufferGroup;
    }
}
bool CaptureDevice_SSPCUDA::StartCapture()
{
    // Create video decoder
    if (m_brun)
    {
        return true;
    }
    m_brun = false;
    std::string ip = "192.168.1.102";//"172.16.152.33";
    ip = m_devicename.substr(6, m_devicename.length() - 6);

    if (m_httpSetting->isNeedSet(ip))
    {
        LOGI("设置相机%s的参数\n", ip.c_str());
        if (m_httpSetting->setData(ip) == 0)
        {
            LOGI("设置相机%s的参数成功\n", ip.c_str());
        }
        else
        {
            LOGE("设置相机%s的参数失败\n", ip.c_str());
            return false;
        }
    }
    m_brun = true;
#ifdef SSP
    m_pLoop = new imf::ThreadLoop(std::bind(&CaptureDevice_SSPCUDA::CaptureThreadFun, this, _1));
    m_pLoop->start();
#else
    if (m_ptcp->Connect(ip.c_str()) == 0)
    {
        LOGI("连接成功%s\n", ip.c_str());
    }
    else
    {
        return false;
    }

    m_RecvHandle = CreateThread(NULL, NULL, RecvThread, this, NULL, NULL);
#endif

    m_DecodeHandle = CreateThread(NULL, NULL, CaptureThread, this, NULL, NULL);
    return true;
}
bool CaptureDevice_SSPCUDA::StopCapture()
{
    m_brun = false;
#ifdef SSP
    m_pClient->stop();
	printf("stop 1\n");
    Sleep(100);
    delete m_pLoop;
    m_pLoop = NULL;
#else
    m_ptcp->DisConnect();
    WaitForSingleObject(m_RecvHandle, 10000);
#endif

    WaitForSingleObject(m_DecodeHandle, 10000);
    if (m_videoDec)
        cuvidDestroyDecoder(m_videoDec);
    if (m_videoParser)
        cuvidDestroyVideoParser(m_videoParser);
    if (m_CtxLock)
        cuvidCtxLockDestroy(m_CtxLock);
    if (m_oContext)
        cuCtxDestroy(m_oContext);
    m_videoParser = NULL;
    m_videoDec = NULL;
    m_CtxLock = NULL;
    m_oContext = NULL;
	delete this;
    return true;
}
bool CaptureDevice_SSPCUDA::ReStartCapture()
{
    std::string ip = "192.168.1.102";//"172.16.152.33";
    ip = m_devicename.substr(6, m_devicename.length() - 6);

    if (m_httpSetting->isNeedSet(ip))
    {
        ReStopCapture();
        LOGI("设置相机%s的参数\n", ip.c_str());
        if (m_httpSetting->setData(ip) == 0)
        {
            LOGI("设置相机%s的参数成功\n", ip.c_str());
        }
        else
        {
            LOGE("设置相机%s的参数失败\n", ip.c_str());
            return false;
        }
#ifdef SSP
		m_pClient->start();
		m_bStartOver = false;
#else
        if (m_ptcp->Connect(ip.c_str()) == 0)
        {
            LOGI("连接成功%s\n", ip.c_str());

        }
        else
        {
            return false;
        }
#endif
    }
    m_ptcp->SetCanConnect(true);
    return true;
}
bool CaptureDevice_SSPCUDA::ReStopCapture()
{
#ifdef SSP
    m_pClient->stop();
	printf("stop 2\n");
    Sleep(100);
#else
    m_ptcp->DisConnect();
#endif
    return true;
}

void CaptureDevice_SSPCUDA::SetAudioCaptureCallBack(AudioCapturePacketCallback *pCC)
{
    m_pACC = pCC;
}
void CaptureDevice_SSPCUDA::SetVideoCaptureCallBack(VideoCapturePacketCallback *pCC)
{
    m_pVCC = pCC;
}
bool CaptureDevice_SSPCUDA::SetCaptureDeviceName(wchar_t *pDeviceName, bool bAudio)
{
    m_bAudio = bAudio;
    wstring DeviceName = wstring(pDeviceName);
    m_devicename = string(dup_wchar_to_utf8(DeviceName.c_str()));
    return true;
}
void CaptureDevice_SSPCUDA::SetDeivceIndex(int nIndex)
{
    m_nDeviceIndex = nIndex;
}
void CaptureDevice_SSPCUDA::SetStreamIndex(int nIndex)
{
    m_nStreamIndex = nIndex;
}
int CUDAAPI CaptureDevice_SSPCUDA::HandlePictureSequence(void * pUserData, CUVIDEOFORMAT * pPicFormat)
{
    CaptureDevice_SSPCUDA *pthis = (CaptureDevice_SSPCUDA *)pUserData;

    //printf("HandlePictureSequence %d %d  %d %d \n", pPicFormat->codec, pPicFormat->coded_height, pPicFormat->coded_width, pPicFormat->chroma_format);
    if (pthis->m_videoDec == NULL)
    {
        CUVIDDECODECREATEINFO	info;
        memset(&info, 0, sizeof(CUVIDDECODECREATEINFO));
        info.CodecType = pPicFormat->codec;
        info.ulWidth = pPicFormat->display_area.right - pPicFormat->display_area.left;
        info.ulHeight = pPicFormat->display_area.bottom - pPicFormat->display_area.top;
        info.ulTargetWidth = info.ulWidth;
        info.ulTargetHeight = info.ulHeight;
        pthis->m_video_h = info.ulHeight;
        pthis->m_video_w = info.ulWidth;
        info.ulNumDecodeSurfaces = 8;
        info.ChromaFormat = pPicFormat->chroma_format; //输入
        info.OutputFormat = cudaVideoSurfaceFormat_NV12;//输出
        info.DeinterlaceMode = cudaVideoDeinterlaceMode_Adaptive;
        info.ulCreationFlags = cudaVideoCreate_PreferCUVID;// cudaVideoCreate_PreferCUDA cudaVideoCreate_PreferDXVA
        if ((info.CodecType == cudaVideoCodec_H264) ||
            (info.CodecType == cudaVideoCodec_H264_SVC) ||
            (info.CodecType == cudaVideoCodec_H264_MVC))
        {
            // assume worst-case of 20 decode surfaces for H264
            info.ulNumDecodeSurfaces = 20;
        }
        if (info.CodecType == cudaVideoCodec_HEVC)
        {
            // ref HEVC spec: A.4.1 General tier and level limits
            int MaxLumaPS = 35651584; // currently assuming level 6.2, 8Kx4K
            int MaxDpbPicBuf = 6;
            int PicSizeInSamplesY = info.ulWidth * info.ulHeight;
            int MaxDpbSize;
            if (PicSizeInSamplesY <= (MaxLumaPS >> 2))
                MaxDpbSize = MaxDpbPicBuf * 4;
            else if (PicSizeInSamplesY <= (MaxLumaPS >> 1))
                MaxDpbSize = MaxDpbPicBuf * 2;
            else if (PicSizeInSamplesY <= ((3 * MaxLumaPS) >> 2))
                MaxDpbSize = (MaxDpbPicBuf * 4) / 3;
            else
                MaxDpbSize = MaxDpbPicBuf;
            MaxDpbSize = MaxDpbSize < 16 ? MaxDpbSize : 16;
            info.ulNumDecodeSurfaces = MaxDpbSize + 4;
        }
#define MAX_FRAME_COUNT 10
        // No scaling
        //info.display_area.left = 0;
        //info.display_area.right = info.ulTargetWidth;
        //info.display_area.top = 0;
        //info.display_area.bottom = info.ulTargetHeight;

        info.ulNumOutputSurfaces = MAX_FRAME_COUNT;  // We won't simultaneously map more than 8 surfaces
        info.vidLock = pthis->m_CtxLock;

        CUresult result = cuvidCreateDecoder(&pthis->m_videoDec, &info);
        if (CUDA_SUCCESS != result)
        {
            LOGE("创建解码器失败\n");
            return 0;
        }
        else
        {
            LOGI("创建解码器成功\n");
        }
        result = cuStreamCreate(&pthis->m_StreamID, 0);
        if (CUDA_SUCCESS != result)
        {
            LOGE("创建流失败\n");
            return 0;
        }
        MessageDelivery::sendMessage(EventDeviceConnectSuc, 0);
    }
    if (pthis->m_video_w != pPicFormat->display_area.right - pPicFormat->display_area.left)
    {
        if (pthis->m_videoDec)
            cuvidDestroyDecoder(pthis->m_videoDec);
        pthis->m_videoDec = NULL;
    }
    return 1; //必须为1
}
int CUDAAPI CaptureDevice_SSPCUDA::HandlePictureDecode(void * pUserData, CUVIDPICPARAMS * pPicParams)
{
    //printf("准备解码 %d  %d \n", pPicParams->FrameHeightInMbs, pPicParams->PicWidthInMbs);
    CaptureDevice_SSPCUDA *pthis = (CaptureDevice_SSPCUDA *)pUserData;
    CUresult ret = cuvidDecodePicture(pthis->m_videoDec, pPicParams);
    if (CUDA_SUCCESS != ret)
    {
        LOGE("解码失败 %d\n", ret);
    }
    return 1;
}
void CaptureDevice_SSPCUDA::SetBufferGroup(BufferGroup *pBG)
{
    m_pBufferGroup = pBG;
}
int CUDAAPI CaptureDevice_SSPCUDA::HandlePictureDisplay(void *pUserData, CUVIDPARSERDISPINFO * pPicParams)
{
    CaptureDevice_SSPCUDA *pthis = (CaptureDevice_SSPCUDA *)pUserData;
    if (pthis->m_fifo.size() > pthis->m_fifo.get_max_size()-10)
    {
        //防止数据堆积，解码后抛弃，避免花屏
        return 0;
    }
    //pthis->m_pFrameQueue->enqueue(pPicParams);
    //printf("。。。。。。。。。。。。。。。。%d %d准备显示 %lld\n", pthis->m_nStreamIndex, pthis->m_cvp.width, pPicParams->timestamp);
    CUdeviceptr pDecodedFrame = 0;
    CUdeviceptr InteropFrame = pthis->m_pBufferGroup->GetPtrByIdx(pthis->m_nStreamIndex);//pthis->m_InteropFrame;// pthis->m_pBufferGroup->getPtrByRouteId(pthis->m_nIndex, pthis->m_cvp.width, pthis->m_cvp.height);
    unsigned int nDecodedPitch = 0;
    unsigned int pFramePitch = pthis->m_video_w * 4;
    CUVIDPROCPARAMS oVideoProcessingParameters;
    memset(&oVideoProcessingParameters, 0, sizeof(CUVIDPROCPARAMS));
    oVideoProcessingParameters.progressive_frame = pPicParams->progressive_frame;
    oVideoProcessingParameters.second_field = 0;
    oVideoProcessingParameters.top_field_first = pPicParams->top_field_first;
    oVideoProcessingParameters.unpaired_field = 1;
    CUresult oResult = cuvidMapVideoFrame(pthis->m_videoDec, pPicParams->picture_index, &pDecodedFrame, &nDecodedPitch, &oVideoProcessingParameters);

    //printf("。。。。。。。。。。。。。。。。%d准备显示 单行数据长度%d\n", pthis->m_nIndex, nDecodedPitch);

    float hueColorSpaceMat[9];
    setColorSpaceMatrix(ITU601, hueColorSpaceMat, 0.0f);
    updateConstantMemory_drvapi(pthis->cuda_oModule, hueColorSpaceMat, pthis->m_StreamID);

    oResult = cudaLaunchNV12toARGBDrv(pDecodedFrame, nDecodedPitch, InteropFrame, pFramePitch, pthis->m_video_w, pthis->m_video_h, pthis->NV12ToARGBCudaFunc, pthis->m_StreamID);

    if (pthis->m_pVCC != NULL&&pthis->m_bResetRuning ==false) {
        VideoCapturePacket	vcp;
        vcp.width = pthis->m_video_w;
        vcp.height = pthis->m_video_h;
        vcp.pData = NULL;
        vcp.nIndex = pthis->m_nStreamIndex;
        vcp.data_len = pFramePitch*vcp.height;
        vcp.pix_fmt = AV_PIX_FMT_ARGB;
        vcp.bCopy = false;
        vcp.pDevData = InteropFrame;
        vcp.pts = pPicParams->timestamp / 25;
        pthis->m_pVCC->onHandleVideoPackets(&vcp, pthis);
        pthis->m_pBufferGroup->UpdateDataByIndex(pthis->m_nStreamIndex, vcp.data_len, vcp.pts);
        pthis->m_curdts = vcp.pts;
        //if (pthis->m_bSync) //先不做同步
        //{
        //	int ret = VideoSyncUtil::GetInstance()->SyncFrameTime(pthis->m_nStreamIndex, vcp.pts);
        //	if (ret > 0)
        //	{
        //		VideoSyncUtil::GetInstance()->Wait();
        //	}
        //	else if (ret == 0)
        //	{
        //		//同步完成
        //	}
        //}

        if (pthis->m_StartTime == 0)
        {
            pthis->m_StartTime = vcp.pts;
            pthis->m_StartCount = GetTickCount();
        }
        else
        {
            __int64 offset = vcp.pts - pthis->m_StartTime + pthis->m_StartCount - GetTickCount();
            if (offset > 0)
            {
                if (offset > 100)
                {
                    pthis->m_StartTime = 0;
                    offset = 10;
                }
                WaitForSingleObject(pthis->m_Eventhandle, offset);
            }
        }
    }
    oResult = cuvidUnmapVideoFrame(pthis->m_videoDec, pDecodedFrame);

    return 0;
}

void CaptureDevice_SSPCUDA::on_264(uint8_t * data, size_t len, uint64_t pts, uint32_t frm_no, uint32_t type)
{
	
    if (!m_brun)
    {
        return;
    }
    if (m_lastTick == 0)
    {
        m_lastTick = GetTickCount();
        m_fps = 0;
    }

    if (GetTickCount() - m_lastTick > 5000)
    {
        m_lastTick = 0;
        LOGI(" %d 接收到帧率%d \n", m_nStreamIndex, m_fps / 5);
    }
    m_fps++;
    h264_data * ph264 = m_fifo.put();
    if (ph264)
    {
        memcpy(ph264->pData, data, len);
        ph264->DataLen = (unsigned int)len;
        ph264->frm_no = frm_no;
        ph264->pts = pts;
        ph264->bnew = true;
        ph264->nType = type;
        m_fifo.put_over();
        //printf("recv time pts %lld %d\n", ph264->pts, m_fifo.size());
    }
    else
    {
        LOGE("丢帧了 %d %d  当前列表中 %zd\n", m_nStreamIndex, frm_no, m_fifo.size());
    }
	if (!m_bStartOver)
	{
		if (m_fifo.size() > 5)
		{
			m_bStartOver = true;
		}
	}

    return;
}
void CaptureDevice_SSPCUDA::on_meta(SspVideoMeta*v, SspAudioMeta*a)
{
    if (!m_brun)
    {
        m_brun = true;
        m_DecodeHandle = CreateThread(NULL, NULL, CaptureThread, this, NULL, NULL);
    }
}
void CaptureDevice_SSPCUDA::on_audio(uint8_t * data, size_t len, uint64_t pts)
{
    //if (m_devicename == "ssp://172.29.4.22")
    //{
    //    LOGI("音频  %d %lld", len, pts);
    //}
}
void CaptureDevice_SSPCUDA::ondisconnect()
{
	printf("disconnect %d %s \n", m_nStreamIndex, m_devicename.c_str());
    LOGE("disconnect---------------------------- %d\n", m_nStreamIndex);
	MessageDelivery::sendMessage(EventDeviceDisConnect, 0);
    if (m_brun) //需要重连
    {
		MessageDelivery::sendMessage(EventDeviceConnect, 0);
		if (!m_bResetRuning)
		{
			Sleep(1000);
			m_pClient->start();
			m_bStartOver = false;
		}
    }

}
void CaptureDevice_SSPCUDA::onbufferfull()
{
    LOGE("onbufferfull---------------------------- %d\n", m_nStreamIndex);
	if (m_nSendFlag)
	{
		MessageDelivery::sendMessage(EventCacheError, 0);
		m_nSendFlag--;
	}
}
void CaptureDevice_SSPCUDA::onexception(int code, const char* description)
{
	m_bStartOver = true;
    printf("on exception %d %s \n", code, description);
	
	MessageDelivery::sendMessage(EventDeviceConnectFailed, code);
    if (code == ERROR_SSP_CONNECTION_FAILED)
    {
        if (m_brun)
        {
			m_pClient->stop();
        }
    }

}
#define MAX_AUDIO_FRAME_SIZE 192000 // 1 second of 48khz 32bit audio  
DWORD WINAPI CaptureDevice_SSPCUDA::ResetThread(LPVOID lp)
{
    CaptureDevice_SSPCUDA *pthis = (CaptureDevice_SSPCUDA *)lp;
    while (pthis->m_bResetRuning&&pthis->m_brun)
    {
        if (pthis->ReStartCapture())
        {
            LOGI("重置成功");
            break;
        }

        Sleep(500);
    }
    Sleep(500);
    pthis->m_bResetRuning = false;
    return 0;
}
void CaptureDevice_SSPCUDA::CaptureThreadFun(imf::Loop *loop)
{
    std::string ip = "192.168.1.102";//"172.16.152.33";
    ip = m_devicename.substr(6, m_devicename.length() - 6);
    m_pClient = new imf::SspClient(ip, loop, 0xFF0000 * 2);
    m_pClient->init();
    m_pClient->setOnH264DataCallback(std::bind(&CaptureDevice_SSPCUDA::on_264, this, _1, _2, _3, _4, _5));
    m_pClient->setOnMetaCallback(std::bind(&CaptureDevice_SSPCUDA::on_meta, this, _1, _2));
    m_pClient->setOnAudioDataCallback(std::bind(&CaptureDevice_SSPCUDA::on_audio, this, _1, _2, _3));
    m_pClient->setOnRecvBufferFullCallback(std::bind(&CaptureDevice_SSPCUDA::onbufferfull, this));
    m_pClient->setOnDisconnectedCallback(std::bind(&CaptureDevice_SSPCUDA::ondisconnect, this));
    m_pClient->setOnExceptionCallback(std::bind(&CaptureDevice_SSPCUDA::onexception, this, _1, _2));
    m_pClient->start();
	m_bStartOver = false;
}
DWORD WINAPI CaptureDevice_SSPCUDA::RecvThread(LPVOID lp)
{
    CaptureDevice_SSPCUDA *pthis = (CaptureDevice_SSPCUDA *)lp;
    pthis->RectDataThread();
    return 0;
}
void CaptureDevice_SSPCUDA::RectDataThread()
{
    uint64_t frm_no = 0;
    int recv_len = 0;
    int packetlen = 0;
    while (m_brun)
    {
        m_ptcp->RectData(frm_no%100==0);
        int packet_len = 0;
        do {
            h264_data * ph264 = m_fifo.put();
            if (ph264)
            {
                packet_len = m_ptcp->GetPacket((char *)ph264->pData, MAX_CODEC_SIZE);
                if (packet_len)
                {
                    ph264->pData = ph264->pData;
                    ph264->DataLen = (unsigned int)packet_len;
                    ph264->frm_no = frm_no;
                    ph264->pts = frm_no * 1000;
                    ph264->bnew = true;
                    ph264->nType = 1;
                    frm_no++;
                    m_fifo.put_over();
                    //LOGI("recv m_fifo Size is %d\n", m_fifo.size());
                }
                else
                {
                    break;
                }
            }
            else
            {
                Sleep(10);
            }
        } while (packet_len != 0 && m_brun);
    }
}
DWORD WINAPI CaptureDevice_SSPCUDA::CaptureThread(LPVOID lp)
{
    CaptureDevice_SSPCUDA *pthis = (CaptureDevice_SSPCUDA *)lp;
    pthis->DecodeThread();
    return 0;
}
void CaptureDevice_SSPCUDA::DecodeThread()
{
    int waitTime = 40; //固定30帧
    long StartF = 0;
    long StartT = 0;
    cuCtxCreate(&m_oContext, CU_CTX_BLOCKING_SYNC, CaptureDeviceUtil::GetInstance()->cuda_oDevice);
    CUresult result = cuvidCtxLockCreate(&m_CtxLock, m_oContext);
    if (CUDA_SUCCESS != result)
    {
        LOGE("cuvidCtxLockCreate failed \n");
        return;
    }
    CUVIDPARSERPARAMS oVideoParserParameters;
    memset(&oVideoParserParameters, 0, sizeof(CUVIDPARSERPARAMS));
    oVideoParserParameters.CodecType = cudaVideoCodec_H264;
    oVideoParserParameters.ulMaxNumDecodeSurfaces = 8;
    oVideoParserParameters.ulMaxDisplayDelay = 4; // this flag is needed so the parser will push frames out to the decoder as quickly as it can//
    oVideoParserParameters.pUserData = this;   //传递使用者自定义的结构//
    oVideoParserParameters.pfnSequenceCallback = HandlePictureSequence;    // Called before decoding frames and/or whenever there is a format change//
    oVideoParserParameters.pfnDecodePicture = HandlePictureDecode;    // Called when a picture is ready to be decoded (decode order)//
    oVideoParserParameters.pfnDisplayPicture = HandlePictureDisplay;   // Called whenever a picture is ready to be displayed (display order)//
    result = cuvidCreateVideoParser(&m_videoParser, &oVideoParserParameters);
    if (CUDA_SUCCESS != result)
    {
        LOGE("cuvidCreateVideoParser failed \n");
        return;
    }

    //int file_size = 0;
    //string module_path = "NV12ToARGB_drvapi64.ptx";

    // in this branch we use compilation with parameters
    const unsigned int jitNumOptions = 3;
    CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
    void **jitOptVals = new void *[jitNumOptions];

    // set up size of compilation log buffer
    jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
    int jitLogBufferSize = 1024;
    jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

    // set up pointer to the compilation log buffer
    jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
    char *jitLogBuffer = new char[jitLogBufferSize];
    jitOptVals[1] = jitLogBuffer;

    // set up pointer to set the Maximum # of registers for a particular kernel
    jitOptions[2] = CU_JIT_MAX_REGISTERS;
    int jitRegCount = 32;
    jitOptVals[2] = (void *)(size_t)jitRegCount;
    CUresult cuStatus;
    cuStatus = cuModuleLoadDataEx(&cuda_oModule, CaptureDeviceUtil::GetInstance()->ptx_source.c_str(), jitNumOptions, jitOptions, (void **)jitOptVals);
    if (cuStatus != CUDA_SUCCESS)
    {
        LOGE("cuModuleLoadDataEx error!\n");
        //return false;
    }
    delete[] jitOptions;
    delete[] jitOptVals;
    delete[] jitLogBuffer;

    cuStatus = cuModuleGetFunction(&(NV12ToARGBCudaFunc), cuda_oModule, "NV12ToARGB_drvapi");
    if (cuStatus != CUDA_SUCCESS)
    {
        LOGE("cuModuleGetFunction error!\n");
        return;
    }
    while (m_fifo.size() < 10 && m_brun)
    {
        Sleep(1);
    }
    m_StartCount = 0;
    m_StartTime = 0;
    m_curdts = 0;
    int ZeroNum = 0;
    StartT = GetTickCount();
    bool bBlackScreen = false;
    while (m_brun)
    {
        h264_data *ph264 = m_fifo.get();
        //LOGI("read m_fifo Size is %d\n", m_fifo.size());
        if (m_fifo.size() < 40)
        {
            long ShouldTime = StartT + StartF*waitTime;
            long offset = ShouldTime - GetTickCount();
            if (offset > 0)
            {
                WaitForSingleObject(m_Eventhandle, offset);
            }
            StartF++;
            if (ph264 == NULL)
            {
                StartF = 0;
                m_StartCount = 0;
                m_StartTime = 0;
                StartT = GetTickCount();
                WaitForSingleObject(m_Eventhandle, 1000 / m_cvp.fps);
                ZeroNum++;
                if (ZeroNum > m_cvp.fps) //正在重连不算时间，防止重连慢造成联系重连
                {
                    ZeroNum = 0;
                    //m_pBufferGroup->ResetData();
                    //m_curdts += 1000 / m_cvp.fps;
                    VideoCapturePacket	vcp;
                    vcp.width = m_video_w;
                    vcp.height = m_video_h;
                    vcp.pData = NULL;
                    vcp.nIndex = m_nStreamIndex;
                    vcp.data_len = vcp.width *vcp.height * 4;
                    vcp.pix_fmt = AV_PIX_FMT_ARGB;
                    vcp.bCopy = false;
                    vcp.pDevData = NULL;
                    vcp.pts = m_curdts;
                    if(m_pVCC)
                        m_pVCC->onHandleVideoPackets(&vcp, this);
                    bBlackScreen = true;
					//printf("Disconnect the camera at %d", GetTickCount());
                    //m_pBufferGroup->UpdateDataByIndex(m_nStreamIndex, vcp.data_len, m_curdts);
                    if (CaptureDeviceUtil::GetInstance()->GetResetParamOnReconnect())
                    {
                        if (!m_bResetRuning&&m_bStartOver)
                        {
                            LOGI("重置设备连接");
							m_bResetRuning = true;
                            CreateThread(NULL, NULL, ResetThread, this, NULL, NULL);
                        }
                    }
					else
					{
#ifdef SSP
						if (m_bStartOver)
						{
							m_pClient->stop();
							printf("stop 3\n");
						}
#endif
					}
                }
                continue;
            }
            else
            {
                if (bBlackScreen)
                {
                    if (m_fifo.size() < 10)
                    {
                        continue;
                    }
                }
            }
        }
        else
        {
            StartF = 0;
            m_StartCount = 0;
            m_StartTime = 0;
            StartT = GetTickCount();
        }
        if (ph264)
        {
            if (!bBlackScreen)
            {
                ZeroNum = 0;
                CUVIDSOURCEDATAPACKET pkt;
                pkt.flags = CUVID_PKT_TIMESTAMP;
                pkt.payload_size = ph264->DataLen;
                pkt.payload = ph264->pData;
                pkt.timestamp = ph264->pts;
                m_curdts = ph264->pts;
                long st = GetTickCount();
                //printf("read time pts %lld %d\n", ph264->pts, m_fifo.size());
                CUresult Curet = cuvidParseVideoData(m_videoParser, &pkt);
                m_fifo.get_over();
            }
            else
            {
                m_fifo.get_over();
                string ip = m_devicename.substr(6, m_devicename.length() - 6);
                if (!m_bResetRuning&&m_httpSetting->isNeedSet(ip))
                {
                    LOGI("相机参数需要重设\n");
                    MessageDelivery::sendMessage(EventInputDeivceParamError, 0);
                }
                else
                {
                    bBlackScreen = false;
                }
            }
        }
    }
}

bool CaptureDevice_SSPCUDA::SetSync(bool bSync)
{
    m_bSync = bSync;
    return true;
}