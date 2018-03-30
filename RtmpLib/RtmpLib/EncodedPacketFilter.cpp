#include "EncodedPacketFilter.h"
//#ifdef CUDA_ACCEL
//#include <cuda_runtime.h>
//#endif
//extern "C"
//{
//#include "libavutil/Imgutils.h"
//}
#define TAG "EncodedPacket"
#include "P2PLog.h"

EncodedPacketFilter::EncodedPacketFilter()
{
}

EncodedPacketFilter::~EncodedPacketFilter()
{
}

EncodedPacketSource::EncodedPacketSource()
{
	pthread_mutex_init(&handlers_mutex, NULL);
}

EncodedPacketSource::~EncodedPacketSource()
{
	pthread_mutex_destroy(&handlers_mutex);
}

void EncodedPacketSource::registerEncodedPacketSink(EncodedPacketSink* sink)
{
	bool already_in = false;
	pthread_mutex_lock(&handlers_mutex);
	handlers.push_back(sink);
	handlers.unique();
	pthread_mutex_unlock(&handlers_mutex);
}

void EncodedPacketSource::unregisterEncodedPacketSink(EncodedPacketSink* sink)
{
	pthread_mutex_lock(&handlers_mutex);
	handlers.remove(sink);
	pthread_mutex_unlock(&handlers_mutex);
}

void EncodedPacketSource::writeEncodedPacket(EncodedPacket *packet)
{
	pthread_mutex_lock(&handlers_mutex);
	std::list<EncodedPacketSink*>::iterator iter = handlers.begin();
	for (; iter != handlers.end(); iter++) {
		if ((*iter)->onEncodedPacket(packet, this) < 0)
		{
			iter = handlers.erase(iter);
			if (iter == handlers.end())
			{
				break;
			}
		}
	}
	pthread_mutex_unlock(&handlers_mutex);
}

EncodedPacket::EncodedPacket()
{
	av_init_packet(&this->pkt);
	this->pkt.data = NULL;
	this->pkt.size = 0;
}

EncodedPacket::~EncodedPacket()
{
	av_packet_unref(&this->pkt);
}

EncodedPacketPool::EncodedPacketPool(int initSize, int maxSize)
{
	if (initSize < 0) {
		initSize = INIT_PACKET_QUEUE_SIZE;
	}

	if (maxSize < 0)
	{
		maxSize = MAX_PACKET_QUEUE_SIZE;
	}

	this->validSize = 0;
	this->currentSize = initSize;
	this->maxSize = maxSize;

	for (int i = 0; i < initSize; i++)
	{
		EncodedPacket* temp = new EncodedPacket();
		emptyQueue.push(temp);
	}
	pthread_mutex_init(&videoQueueMutex, NULL);
	pthread_mutex_init(&emptyQueueMutex, NULL);
}

EncodedPacketPool::~EncodedPacketPool()
{
	pthread_mutex_lock(&videoQueueMutex);
	while (!validQueue.empty())
	{
		EncodedPacket* temp = validQueue.front();
		validQueue.pop();
		delete temp;
	}
	pthread_mutex_unlock(&videoQueueMutex);

	pthread_mutex_lock(&emptyQueueMutex);
	while (!emptyQueue.empty())
	{
		EncodedPacket* temp = emptyQueue.front();
		emptyQueue.pop();
		delete temp;
	}
	pthread_mutex_unlock(&emptyQueueMutex);

	pthread_mutex_destroy(&videoQueueMutex);
	pthread_mutex_destroy(&emptyQueueMutex);
}

int EncodedPacketPool::getValidSize()
{
	return validQueue.size();
}

int EncodedPacketPool::getEmptySize()
{
	return emptyQueue.size();
}

int EncodedPacketPool::getCurrent()
{
	return currentSize;
}

int EncodedPacketPool::gatMax()
{
	return maxSize;
}

void EncodedPacketPool::pushValid(EncodedPacket* packet)
{
	pthread_mutex_lock(&videoQueueMutex);
	validQueue.push(packet);
	pthread_mutex_unlock(&videoQueueMutex);
}

EncodedPacket* EncodedPacketPool::popValid()
{
	EncodedPacket* ret = nullptr;
	pthread_mutex_lock(&videoQueueMutex);
	if (validQueue.size() > 0) {
		ret = validQueue.front();
		validQueue.pop();
	}
	pthread_mutex_unlock(&videoQueueMutex);
	return ret;
}

void EncodedPacketPool::pushEmpty(EncodedPacket* packet)
{
	pthread_mutex_lock(&emptyQueueMutex);
	av_packet_unref(&packet->pkt);
	packet->stream_index = 0;
	packet->time_base.den = 0;
	packet->time_base.num = 0;
	emptyQueue.push(packet);
	pthread_mutex_unlock(&emptyQueueMutex);
}

EncodedPacket* EncodedPacketPool::popEmpty()
{
	EncodedPacket* ret = nullptr;

	pthread_mutex_lock(&emptyQueueMutex);
	if (emptyQueue.size() > 0) {
		ret = emptyQueue.front();
		emptyQueue.pop();
	}
	pthread_mutex_unlock(&emptyQueueMutex);

	if (ret == nullptr && currentSize < maxSize)
	{
		currentSize++;
		LOGI("EncodedPacketPool: new EncodedPacket, current size:%d", currentSize);
		ret = new EncodedPacket();
	}
	return ret;
}