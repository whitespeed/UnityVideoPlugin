#include "PCMFrameFilter.h"
extern "C"
{
#include "libavutil/samplefmt.h"
}
#include "P2PLog.h"

int	offset_hdmi_a = 0;
int	offset_hdmi_a_temp = 0;
char sourseFile[10][2];
unsigned char* hdmi_pAudioData = new uint8_t[MAX_HDMI_AUDIO_CACHE];
#define PCMFILTER_MAX_AUDIO_FIFO_SECONDS 100

PCMFrameFilter::PCMFrameFilter()
{
	//hdmi_pAudioData = new uint8_t[MAX_HDMI_AUDIO_CACHE];
	//pthread_mutex_init(&fqaQueueMutex, NULL);
	//pthread_mutex_init(&sqaQueueMutex, NULL);
    pthread_mutex_init(&audioQueueMutex, NULL);
}

PCMFrameFilter::~PCMFrameFilter()
{
    if (fifo1)
        av_audio_fifo_free(fifo1);
    if (fifo2)
        av_audio_fifo_free(fifo2);
	//pthread_mutex_destroy(&fqaQueueMutex);
	//pthread_mutex_destroy(&sqaQueueMutex);
    pthread_mutex_destroy(&audioQueueMutex);
}
/*旧的音频输入区分方式
MixType PCMFrameFilter::GetAudioMixType(AudioType mixtype)

{
	设置是否属于混音的flag，最多只允许选两路的情况下：
	
	if (mixtype == HDMI0 || mixtype == HDMI1 || mixtype == HDMI2 || mixtype == HDMI3 || mixtype == MICROPHONE || mixtype == LINEIN)
	{
		audio_mix = SINGLE;  //单路
	}
	else if (mixtype == (HDMI0 | HDMI1) || mixtype == (HDMI1 | HDMI2) || mixtype == (HDMI2 | HDMI3) || mixtype == (HDMI3 | HDMI0) || mixtype == (HDMI3 | HDMI1) || mixtype == (HDMI3 | HDMI2))
	{
		audio_mix = MIXTYPE0; //两个HDMI mixing
	}
	else if (mixtype == (HDMI0 | MICROPHONE) || mixtype == (HDMI1 | MICROPHONE) || mixtype == (HDMI2 | MICROPHONE) || mixtype == (HDMI3 | MICROPHONE))
	{
		audio_mix = MIXTYPE1; //HDMI与mic mixing,mic的pcksize和linein是一样的88200
		//LOGD("MIXTYPE1 %d,apck->size =  %d \n", mixtype, apck->size);
	}
	else if (mixtype == (HDMI0 | LINEIN) || mixtype == (HDMI1 | LINEIN) || mixtype == (HDMI2 | LINEIN) || mixtype == (HDMI3 | LINEIN))
	{
		audio_mix = MIXTYPE2; // HDMI与LINEIN mixing
		//LOGD("MIXTYPE2 %d,apck->size =  %d \n", mixtype, apck->size);
	}
	else if (mixtype == (LINEIN | MICROPHONE))
	{
		audio_mix = MIXTYPE3;//mic与LINEIN mixing
		//LOGD("MIXTYPE3 == %d,apck->size =  %d \n", mixtype, apck->size);
	}
	return audio_mix;
}
*/
//新的采集方案中音频输入统一了输入的packet大小：
MixType PCMFrameFilter::GetAudioMixType(int mix_type)
{
	MixType audio_mix;
	if (mix_type == 1)
	{
		audio_mix = SINGLE;
	}
	else if (mix_type == 2)
	{
		audio_mix = MIXTYPE;
	}
	else {
		audio_mix = UNSUPPORTED;
	}
	return audio_mix;
}
void PCMFrameFilter::AudioMixMethod(char sourseFile[10][SIZE_AUDIO_FRAME], int number, char *objectFile)
{
	//归一化混音  
	const int MAX = 32767;
	const int MIN = -32768;

	double f = 1;
	int output;
	int i = 0, j = 0;
	for (i = 0; i<SIZE_AUDIO_FRAME / 2; i++)
	{
		int temp = 0;
		for (j = 0; j<number; j++)
		{
			temp += *(short*)(sourseFile[j] + i * 2);
		}
		output = (int)(temp*f);
		if (output>MAX)
		{
			f = (double)MAX / (double)(output);
			output = MAX;
		}
		if (output<MIN)
		{
			f = (double)MIN / (double)(output);
			output = MIN;
		}
		if (f<1)
		{
			f += ((double)1 - f) / (double)32;
		}
		*(short*)(objectFile + i * 2) = (short)output;
	}
}
/*verison2.2版本混音处理说明：*/
//目前只针对输入是16 bit的音频格式进行处理，两路直接进行混音，然后在重采样成fltp 32 bit类型的，因为目前的混音算法只支持16bit的
//后续计划是先将输入统一重采样成16bit，之后进行混音，然后再将混音结果重采样成fltp
//或者是找到fltp类型的混音算法，然后先重采样成fltp，之后直接混音就可以输出了

PCMFrame *PCMFrameFilter::DoAudioMix(MixType dest_audiomix, PCMFrame *apck, int first_mix_audio_type)
{
	PCMFrame *dstpack = NULL;
	short data0, data1, data_mix = 0;
	//if (dest_audiomix == MIXTYPE0 || dest_audiomix == MIXTYPE3) //HDMI + HDMI or mic + linein
	if (dest_audiomix == MIXTYPE)
	{
		if (apck->AudioIndex == first_mix_audio_type)
		{
			//PCMFrame *temp = new PCMFrame(apck->sample_fmt, apck->channels, apck->nb_samples, 1);
			//av_samples_copy(temp->data, apck->data, 0, 0, apck->nb_samples, apck->channels, apck->sample_fmt);
   //         temp->sample_rate = apck->sample_rate;
			//pthread_mutex_lock(&fqaQueueMutex);
			//firstqa.push(temp);
			//pthread_mutex_unlock(&fqaQueueMutex);
            if (fifo1 == NULL)
            {
                m_nb_samples_1 = apck->nb_samples;
                m_sample_fmt_1 = apck->sample_fmt;
                m_channels_1 = apck->channels;
                m_sample_rate_1 = apck->sample_rate;
                fifo1 = av_audio_fifo_alloc(m_sample_fmt_1, m_channels_1, m_sample_rate_1 * PCMFILTER_MAX_AUDIO_FIFO_SECONDS);
                LOGI("DoAudioMix: m_nb_samples_1 %d , m_sample_fmt_1 %d, m_channels_1 %d, m_sample_rate_1 %d", m_nb_samples_1, m_sample_fmt_1, m_channels_1, m_sample_rate_1);
            }
            av_audio_fifo_write(fifo1, (void**)apck->data, apck->nb_samples);
		}
		else
		{
			//PCMFrame *temp = new PCMFrame(apck->sample_fmt, apck->channels, apck->nb_samples, 1);
			//av_samples_copy(temp->data, apck->data, 0, 0, apck->nb_samples, apck->channels, apck->sample_fmt);
   //         temp->sample_rate = apck->sample_rate;
			//pthread_mutex_lock(&sqaQueueMutex);
			//secondqa.push(temp);
			//pthread_mutex_unlock(&sqaQueueMutex);
            if (fifo2 == NULL)
            {
                m_nb_samples_2 = apck->nb_samples;
                m_sample_fmt_2 = apck->sample_fmt;
                m_channels_2 = apck->channels;
                m_sample_rate_2 = apck->sample_rate;
                fifo2 = av_audio_fifo_alloc(m_sample_fmt_2, m_channels_2, m_sample_rate_2 * PCMFILTER_MAX_AUDIO_FIFO_SECONDS);
                LOGI("DoAudioMix: m_nb_samples_2 %d , m_sample_fmt_2 %d, m_channels_2 %d, m_sample_rate_2 %d", m_nb_samples_2, m_sample_fmt_2, m_channels_2, m_sample_rate_2);
            }
            av_audio_fifo_write(fifo2, (void**)apck->data, apck->nb_samples);
		}
	}
	else
	{
		return NULL;
	}
	/*旧的音频输入方式对于mic和hdmi的packet大小不一致，混音时需要做如下处理：
	else if (dest_audiomix == MIXTYPE1 || dest_audiomix == MIXTYPE2)
	{
		if (apck->type == HDMI0 || apck->type == HDMI1 || apck->type == HDMI2 || apck->type == HDMI3)
		{
			//通过每次存够88200bytes的HDMI audiopacket再push到fqa,然后才和mic的进行混音
			if (offset_hdmi_a + apck->size > MAX_HDMI_AUDIO_CACHE)
			{
				memmove(hdmi_pAudioData, hdmi_pAudioData + offset_hdmi_a_temp, offset_hdmi_a - offset_hdmi_a_temp);
				offset_hdmi_a = offset_hdmi_a - offset_hdmi_a_temp;
				offset_hdmi_a_temp = 0;
			}
			memcpy(hdmi_pAudioData + offset_hdmi_a, apck->data[0], apck->size);
			offset_hdmi_a += apck->size;
			while ((offset_hdmi_a - offset_hdmi_a_temp) >= MIC_LINEIN_PCK_SIZE) {  //88200是mic的每个packet size
				PCMFrame *temp = new PCMFrame(apck->sample_fmt, apck->channels, (MIC_LINEIN_PCK_SIZE * apck->nb_samples )/ apck->size , 1);
				//av_samples_copy(temp->data, &hdmi_pAudioData, 0, 0, apck->nb_samples * MIC_LINEIN_PCK_SIZE / apck->size, apck->channels, apck->sample_fmt);
				memcpy(temp->data[0], hdmi_pAudioData + offset_hdmi_a_temp, MIC_LINEIN_PCK_SIZE);
				offset_hdmi_a_temp += MIC_LINEIN_PCK_SIZE;
				pthread_mutex_lock(&fqaQueueMutex);
				firstqa.push(temp);
				pthread_mutex_unlock(&fqaQueueMutex);
			}
		}
		else
		{
			PCMFrame *temp = new PCMFrame(apck->sample_fmt, apck->channels, apck->nb_samples, 1);
			av_samples_copy(temp->data, apck->data, 0, 0, apck->nb_samples, apck->channels, apck->sample_fmt);
			//dest_nb_samples = apck->nb_samples;
			pthread_mutex_lock(&sqaQueueMutex);
			secondqa.push(temp);
			pthread_mutex_unlock(&sqaQueueMutex);
		}
	}
	*/
 //  if (!firstqa.empty() && !secondqa.empty())
 //  {
	//pthread_mutex_lock(&fqaQueueMutex);
	//PCMFrame *fqapack = firstqa.front();
	//firstqa.pop();
	//pthread_mutex_unlock(&fqaQueueMutex);
	//pthread_mutex_lock(&sqaQueueMutex);
	//PCMFrame *sqapack = secondqa.front();
	//secondqa.pop();
	//pthread_mutex_unlock(&sqaQueueMutex);
	//dstpack = new PCMFrame(sqapack->sample_fmt, sqapack->channels, sqapack->nb_samples, 1);//16 bits 2 channels
 //   dstpack->sample_rate = sqapack->sample_rate;
	//for (int i = 0; i < (dstpack->size) / 2; i++)
	//{
	//	memcpy(&data0, fqapack->data[0] + i * 2, 2);
	//	memcpy(&data1, sqapack->data[0] + i * 2, 2);
	//	//fwrite(&data0, 2, 1, fp1);//存pcm文件供参考，这个hdmi0的pcm声音是正常的
	//	//fwrite(&data1, 2, 1, fp2);//存pcm文件供参考，这个hdmi1的声音也是正常的 
	//	*(short*)sourseFile[0] = data0;
	//	*(short*)sourseFile[1] = data1;
	//	//归一化算法:
	//	AudioMixMethod(sourseFile, 2, (char *)&data_mix);//目前效果相对较好的
	//	//fwrite(&data_mix, 2, 1, fpm_data);
	//	memcpy(dstpack->data[0] + i * 2, &data_mix, 2);
	//	//fwrite(dstpack->data + i * 2, 2, 1, fpm);
	//}
	////fwrite(dstpack->data, dstpack->size, 1, fpm);
	//delete fqapack;
	//delete sqapack;
 //  }

    if (fifo1 == NULL || fifo2 == NULL)
    {
        return NULL;
    }
    int audio_frame_nb_samples = m_nb_samples_1;
    AVSampleFormat sample_fmt = m_sample_fmt_1;
    int channels = m_channels_1;
    int sample_rate = m_sample_rate_1;
    if (m_nb_samples_1 > m_nb_samples_2)
    {
        audio_frame_nb_samples = m_nb_samples_2;
        sample_fmt = m_sample_fmt_2;
        channels = m_channels_2;
        sample_rate = m_sample_rate_2;
    }
    int fifo1_size = av_audio_fifo_size(fifo1);
    int fifo2_size = av_audio_fifo_size(fifo2);
    LOGD("DoAudioMix: fifo1_size %d , fifo2_size %d", fifo1_size, fifo2_size);
    PCMFrame *dst_pack = NULL;
    PCMFrame *first_pack = NULL;
    PCMFrame *second_pack = NULL;
    pthread_mutex_lock(&audioQueueMutex);
    if (fifo1_size > audio_frame_nb_samples && fifo2_size > audio_frame_nb_samples) {
        //
        first_pack = new PCMFrame(sample_fmt, channels, audio_frame_nb_samples);
        second_pack = new PCMFrame(sample_fmt, channels, audio_frame_nb_samples);
        dstpack = new PCMFrame(sample_fmt, channels, audio_frame_nb_samples);
        dstpack->sample_rate = sample_rate;

        av_audio_fifo_read(fifo1, (void **)first_pack->data, audio_frame_nb_samples);
        av_audio_fifo_read(fifo2, (void **)second_pack->data, audio_frame_nb_samples);
        for (int i = 0; i < (dstpack->size) / 2; i++)
        {
            memcpy(&data0, first_pack->data[0] + i * 2, 2);
            memcpy(&data1, second_pack->data[0] + i * 2, 2);
            *(short*)sourseFile[0] = data0;
            *(short*)sourseFile[1] = data1;
            //归一化算法:
            AudioMixMethod(sourseFile, 2, (char *)&data_mix);
            memcpy(dstpack->data[0] + i * 2, &data_mix, 2);
        }
        delete first_pack;
        delete second_pack;
    }
    pthread_mutex_unlock(&audioQueueMutex);
    return dstpack;
}
PCMFrame::PCMFrame(enum AVSampleFormat sample_fmt, int nb_channels, int nb_samples, int align)
{
    this->sample_fmt = sample_fmt;
    this->channels = nb_channels;
    this->nb_samples = nb_samples;
    size = av_samples_alloc(data, linesize, nb_channels, nb_samples, sample_fmt, 1);

}

PCMFrame* PCMFrame::dump()
{
	PCMFrame* ret = NULL;
	ret = new PCMFrame(sample_fmt, channels, nb_samples,1);
	memcpy(ret->data[0], data[0], size);
	ret->AudioIndex = AudioIndex;
	ret->channels = channels;
	ret->channel_layout = channel_layout;
	ret->nb_samples = nb_samples;
	ret->pts = pts;
	ret->sample_fmt = sample_fmt;
	ret->sample_rate = sample_rate;
	ret->size = size;
	ret->linesize[0] = linesize[0];
	ret->timebase_den = timebase_den;
	ret->timebase_num = timebase_num;
	ret->type = type;
	return ret;
}

PCMFrame::~PCMFrame()
{
    if(size > 0)
        av_freep(&data[0]);
}

PCMFrameSource::PCMFrameSource()
{
    pthread_mutex_init(&handlers_mutex, NULL);
}

PCMFrameSource::~PCMFrameSource()
{
    pthread_mutex_destroy(&handlers_mutex);
}

void PCMFrameSource::registerPCMFrameSink(PCMFrameSink* sink)
{
    bool already_in = false;
    pthread_mutex_lock(&handlers_mutex);
    handlers.push_back(sink);
    handlers.unique();
    pthread_mutex_unlock(&handlers_mutex);
}

void PCMFrameSource::unregisterPCMFrameSink(PCMFrameSink* sink)
{
    pthread_mutex_lock(&handlers_mutex);
    handlers.remove(sink);
    pthread_mutex_unlock(&handlers_mutex);
}

void PCMFrameSource::writePCMFrame(PCMFrame *packet)
{
    pthread_mutex_lock(&handlers_mutex);
    std::list<PCMFrameSink*>::iterator iter = handlers.begin();
    for (; iter != handlers.end(); iter++) {
        (*iter)->onPCMFrame(packet);
    }
    pthread_mutex_unlock(&handlers_mutex);
}