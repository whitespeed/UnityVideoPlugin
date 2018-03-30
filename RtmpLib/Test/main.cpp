#include <string>
#include "../RtmpLib/VideoFrameFilter.h"
#include "../RtmpLib/PCMFrameFilter.h"
#include "../RtmpLib/AVOutputManager.h"


#pragma comment (lib,"../x64/Debug/RtmpLib.lib")
#pragma comment (lib,"../libs/avutil.lib")
#pragma comment (lib,"../libs/pthreadVC2.lib")
#pragma comment (lib, "cudart.lib")
#pragma comment (lib,"../libs/libyuv.lib")
#pragma comment (lib,"../libs/avformat.lib")

#pragma comment (lib,"../libs/swresample.lib")
#pragma comment(lib, "ws2_32.lib")
class TestSource : public PCMFrameSource , public VideoFrameSource, public VROutputCallback {
public:
    TestSource() {

    }


    void registerVideoFrameSink(VideoFrameSink* sink) {

    }
    void unregisterVideoFrameSink(VideoFrameSink* sink) {

    }
    void writeVideoFrame(VideoFrame *packet) {

    }

    
};

void main() {
    AVOutputManager* m_pOutputManger = new AVOutputManager();
    StreamSourceGroup  m_streamSourceGroup ;
    TestSource* testsource = new TestSource();
    m_streamSourceGroup._AudioCapture_ = testsource;
    m_streamSourceGroup._VideoCapture_ = testsource;
    m_pOutputManger->setFrameSourceGroup(m_streamSourceGroup);
    std::string  url = "rtmp://vrlivegs.stream.moguv.com/live/20180328";
    StreamOpt m_streamOpt;
    /*
    set some value for opt
    */
    m_streamOpt.w = 1920;
    m_streamOpt.h = 960;
    m_streamOpt.vb = 3145728;
    m_streamOpt.vc = 0;
    m_streamOpt.ab = 65536;
    m_streamOpt.ac = 0;
    m_streamOpt.adms = 0;

    printf("start to push rtmp stream\n");
    m_pOutputManger->startOutputRtmp((VrType)0, url, m_streamOpt, testsource);

    VideoFrame* videoframe = new VideoFrame(AV_PIX_FMT_YUV420P, 1920, 960);
    //TODO 野割video data

    PCMFrame * pcmframe = new PCMFrame(AV_SAMPLE_FMT_S16, 2, 44100);
    pcmframe->sample_rate = 44100;
    //TODO 野割audio data
    while (true) {
        Sleep(40);
        m_pOutputManger->onVideoFrame(videoframe, NULL);
        //m_pOutputManger->onPCMFrame(pcmframe);
    }
    m_pOutputManger->stopOutputRtmp();
    
}