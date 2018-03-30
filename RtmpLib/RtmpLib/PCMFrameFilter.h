#pragma once

#include <stdint.h>
#include <list>
#include <queue>
extern "C"
{
#include "libavutil/avutil.h"
#include "libswresample/swresample.h"
#include "pthread.h"
#include "libavutil/audio_fifo.h"
}
//for audio mixing:
#define SIZE_AUDIO_FRAME 2
#define MIC_LINEIN_PCK_SIZE 88200
#define MAX_HDMI_AUDIO_CACHE 200*4096  //分配的用于存放audio buffersize的临时内存
typedef enum MixType {
	SINGLE = 0,    //单路
	MIXTYPE = 0x01,// 双路
	UNSUPPORTED = 0x02, //三路以上，不支持
}MixType;

//旧的mixtype:
/*
typedef enum MixType {
	SINGLE = 0,    //单路
	MIXTYPE0 = 0x01,// 双路
	MIXTYPE1 = 0x02,// HDMI & MIC
	MIXTYPE2 = 0x04,// HDMI & LINEIN
    MIXTYPE3 = 0x08,// MIC & LINEIN
}MixType;
*/
class PCMFrame;
class PCMFrameSink;

typedef enum AudioType
{
    NONE = 0,
    HDMI0 = 0x01,
    HDMI1 = 0x02,
    HDMI2 = 0x04,
    HDMI3 = 0x08,
    MICROPHONE = 0x10,
    LINEIN = 0x20,
}AudioType;

class PCMFrame
{
public:
    PCMFrame(enum AVSampleFormat sample_fmt, int nb_channels, int nb_samples, int align = 1);
    ~PCMFrame();
	//audio dump:
	PCMFrame* dump();
    uint8_t *data[8]; // audio data
    int linesize[8];
    enum AVSampleFormat sample_fmt;
    int size;

    /**
    * number of audio samples (per channel) described by this frame
    */
    int nb_samples;
    /**
    * Sample rate of the audio data.
    */
    int sample_rate;

    /**
    * number of audio channels, only used for audio.
    */
    int channels;

    /**
    * Channel layout of the audio data.
    */
    uint64_t channel_layout;

    int64_t pts;
    int timebase_num;
    int timebase_den;
    /*
    * the audio is from which device
    */
	AudioType type;
    int AudioIndex;
};

class PCMFrameFilter
{
public:
    PCMFrameFilter();
    ~PCMFrameFilter();
	//for audio mix:
	void AudioMixMethod(char sourseFile[10][SIZE_AUDIO_FRAME], int number, char *objectFile);
	MixType GetAudioMixType(int mix_type);
	PCMFrame *DoAudioMix(MixType dest_audiomix, PCMFrame *apck, int first_mix_audio_type);
	//std::queue<PCMFrame *> firstqa;
	//std::queue<PCMFrame *> secondqa;
	//pthread_mutex_t fqaQueueMutex;
	//pthread_mutex_t sqaQueueMutex;
    AVAudioFifo *fifo1 = NULL;
    AVAudioFifo *fifo2 = NULL;
    pthread_mutex_t audioQueueMutex;
    int m_nb_samples_1 = 0;
    AVSampleFormat m_sample_fmt_1 = AV_SAMPLE_FMT_S16;
    int m_channels_1 = 2;
    int m_sample_rate_1 = 44100;
    int m_nb_samples_2 = 0;
    AVSampleFormat m_sample_fmt_2 = AV_SAMPLE_FMT_S16;
    int m_channels_2 = 2;
    int m_sample_rate_2 = 44100;
};

class PCMFrameSource
{
public:
    PCMFrameSource();
    ~PCMFrameSource();
    void registerPCMFrameSink(PCMFrameSink* sink);
    void unregisterPCMFrameSink(PCMFrameSink* sink);
    void writePCMFrame(PCMFrame *packet);

private:
    pthread_mutex_t handlers_mutex;
    std::list<PCMFrameSink*> handlers;
};

class PCMFrameSink
{
public:
    virtual int onPCMFrame(PCMFrame *packet) = 0;
};

