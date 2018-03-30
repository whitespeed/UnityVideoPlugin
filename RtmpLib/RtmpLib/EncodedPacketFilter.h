#pragma once

#include <stdint.h>
#include <list>
#include <queue>
#include <cuda.h>
extern "C"
{
#include "libavutil/avutil.h"
#include "libavutil/pixdesc.h"
#include "libavcodec/avcodec.h"
#include "pthread.h"
}

#define INIT_PACKET_QUEUE_SIZE 10
#define MAX_PACKET_QUEUE_SIZE 200
#define RESERVED_PACKET_QUEUE_SIZE 10

class EncodedPacket;
class EncodedPacketSink;

class EncodedPacketFilter
{
public:
	EncodedPacketFilter();
	~EncodedPacketFilter();
};

class EncodedPacketSource
{
public:
	EncodedPacketSource();
	~EncodedPacketSource();
	virtual void registerEncodedPacketSink(EncodedPacketSink* sink);
	virtual void unregisterEncodedPacketSink(EncodedPacketSink* sink);
	virtual void writeEncodedPacket(EncodedPacket *packet);

private:
	pthread_mutex_t handlers_mutex;
	std::list<EncodedPacketSink*> handlers;
};

class EncodedPacketSink
{
public:
	virtual int onEncodedPacket(EncodedPacket *packet, EncodedPacketSource *nSource) = 0;
};

class EncodedPacket
{
public:
	EncodedPacket();
	~EncodedPacket();
public:
	AVPacket pkt;
	AVRational time_base;
	int stream_index;
};

class EncodedPacketPool
{
public:
	EncodedPacketPool(int initSize, int maxSize);
	~EncodedPacketPool();
	int getValidSize();
	int getEmptySize();
	int getCurrent();
	int gatMax();

	void pushValid(EncodedPacket* packet);
	EncodedPacket* popValid();

	void pushEmpty(EncodedPacket* packet);
	EncodedPacket* popEmpty();
private:
	std::queue<EncodedPacket *> validQueue;
	std::queue<EncodedPacket *> emptyQueue;
	pthread_mutex_t videoQueueMutex;
	pthread_mutex_t emptyQueueMutex;
	volatile int validSize = 0;
	volatile int currentSize = 0;
	int maxSize = MAX_PACKET_QUEUE_SIZE;
};
