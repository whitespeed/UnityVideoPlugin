#pragma once

#ifdef	__cplusplus
extern "C"
{
#endif
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"

#ifdef __cplusplus
};
#endif
#include "tcp_connect.h"

#include<thread>
#include <imf/net/loop.h>
#include <imf/net/threadloop.h>
#include <imf/ssp/sspclient.h>
#include<nvcuvid.h>
#include<cuviddec.h>
#include "CaptureDevice.h"
#include "BufferGroup.h"

#undef _WINDOWS_
#include "HttpClient.h"

using namespace std::placeholders;
using namespace imf;
using namespace std;

class CaptureDevice_SSPCUDA :
    public CaptureDevice
{
public:
    CaptureDevice_SSPCUDA();
    ~CaptureDevice_SSPCUDA();
public:
    virtual CaptureDeviceType GetDeviceType();
    virtual char * GetDeviceName();
    virtual void SetParam(void *);
    virtual bool StartCapture();
    virtual bool StopCapture();
    virtual bool SetSync(bool bSync);
    virtual void SetAudioCaptureCallBack(AudioCapturePacketCallback *pCC);
    virtual void SetVideoCaptureCallBack(VideoCapturePacketCallback *pCC);
private:
    bool ReStartCapture();
    bool ReStopCapture();
    static int CUDAAPI HandlePictureSequence(void * pUserData, CUVIDEOFORMAT * pPicFormat);
    static int CUDAAPI HandlePictureDecode(void * pUserData, CUVIDPICPARAMS * pPicParams);
    static int CUDAAPI HandlePictureDisplay(void *pUserData, CUVIDPARSERDISPINFO * pPicParams);
private:
    AudioCapturePacketCallback	*m_pACC;
    VideoCapturePacketCallback	*m_pVCC;
    imf::SspClient *			m_pClient;
    imf::ThreadLoop *			m_pLoop;
    CUvideodecoder				m_videoDec;
    CUvideoparser				m_videoParser;
    CUcontext					m_oContext; //需要放到线程中创建，切隶属于这个线程
    CUvideoctxlock				m_CtxLock;
    CUstream					m_StreamID;
    CUmodule					cuda_oModule;
    CUfunction					NV12ToARGBCudaFunc;
    BufferGroup					*m_pBufferGroup;
    fifo_h264                   m_fifo;
    string						m_devicename;
    bool						m_brun;
    HANDLE						m_threadHandle;
    bool						m_bAudio;
    int							m_video_w;
    int							m_video_h;
    CaptureAudioParam	m_cap;
    CaptureVideoParam	m_cvp;
    char				*m_pRGB24VideoData;
    int					m_StreamVideoIndex;
    int					m_StreamAudioIndex;
    AVFrame				*m_pFrame;
    AVFrame				*m_pOutFrame;
    int					m_outsize;
    HANDLE				m_Eventhandle;
    HANDLE				m_DecodeHandle;
    HANDLE				m_RecvHandle;
    long				m_StartCount;
    __int64				m_StartTime;

    int					m_fps;
    long				m_lastTick;
    bool				m_bSync;

    tcp_connect			*m_ptcp;
	int					m_nSendFlag = 5;
    CHttpClient         *m_httpSetting;
	bool				m_bStartOver = false;
public:
    int64_t				m_curdts;
    uint64_t			m_frmno;
    int					m_nDeviceIndex;
    int					m_nStreamIndex;

    void on_264(uint8_t * data, size_t len, uint64_t pts, uint32_t frm_no, uint32_t type);
    void on_meta(struct SspVideoMeta*, struct SspAudioMeta*);
    void on_audio(uint8_t * data, size_t len, uint64_t pts);
    void ondisconnect();
    void onbufferfull();
    void onexception(int code, const char* description);

    bool SetCaptureDeviceName(wchar_t *pVirtualDeviceName /*url or file path*/, bool bAudio);
    void SetDeivceIndex(int nIndex);
    void SetStreamIndex(int nIndex);
    void SetBufferGroup(BufferGroup *pBG);
    void CaptureThreadFun(imf::Loop *loop);
    void DecodeThread();
    static DWORD WINAPI CaptureThread(LPVOID lp);
    void RectDataThread();
    static DWORD WINAPI RecvThread(LPVOID lp);
    bool m_bResetRuning;
    static DWORD WINAPI ResetThread(LPVOID lp);
};
