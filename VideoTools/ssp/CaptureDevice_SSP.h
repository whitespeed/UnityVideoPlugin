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

#include<thread>
#include <imf/net/loop.h>
#include <imf/net/threadloop.h>
#include <imf/ssp/sspclient.h>
#include<nvcuvid.h>
//#include<cudaGL.h>
#include<cuviddec.h>
#include "CaptureDevice.h"

using namespace std::placeholders;
using namespace imf;
using namespace std;

class CaptureDevice_SSP :
	public CaptureDevice
{
public:
	CaptureDevice_SSP();
	~CaptureDevice_SSP();
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
    AudioCapturePacketCallback	*m_pACC;
    VideoCapturePacketCallback	*m_pVCC;
	imf::SspClient *			m_pClient;
	imf::ThreadLoop *			m_pLoop;
	queue<h264_data *>			m_listData;
	queue<h264_data *>			m_listRecycleData;
	string				m_devicename;
	AVFormatContext		*m_pFormatCtx;
	AVCodecParameters	*m_pCodecParam;
	AVCodecContext		*m_pCodecContext;
	AVCodec				*m_pCodec;
	SwsContext			*m_video_convert_ctx;
	SwrContext			*m_audio_context_ctx;
	bool				m_brun;
	HANDLE				m_threadHandle;
	bool				m_bAudio;
	
	CaptureAudioParam	m_cap;
	CaptureVideoParam	m_cvp;
	char				*m_pARGB32VideoData;
	int					m_StreamVideoIndex;
	int					m_StreamAudioIndex;
	AVFrame				*m_pFrame;
	AVFrame				*m_pOutFrame;
	int					m_outsize;
	HANDLE				m_Eventhandle;
	HANDLE				m_DecodeHandle;
	BufferGroup			*m_pBufferGroup;
	int					m_fps;
	long				m_lastTick;
	long				m_StartTime;
	long				m_StartCount;
	bool				m_bSync;
public:
	int64_t				m_curdts;
	uint64_t			m_frmno;
	int					m_nDeviceIndex;
	int					m_nStreamIndex;
	void on_264(uint8_t * data, size_t len, uint64_t pts, uint32_t frm_no, uint32_t type);
	void on_meta(struct SspVideoMeta* , struct SspAudioMeta*);
	void on_audio(uint8_t * data, size_t len, uint64_t pts);
	void ondisconnect();
	void onbufferfull();

	bool SetCaptureDeviceName(wchar_t *pVirtualDeviceName /*url or file path*/, bool bAudio);
	void SetDeivceIndex(int nIndex);
	void SetStreamIndex(int nIndex);
	void CaptureThreadFun(imf::Loop *loop);
	void SetBufferGroup(BufferGroup *pBG);
	void DecodeThread();
	static DWORD WINAPI CaptureThread(LPVOID lp);

};

