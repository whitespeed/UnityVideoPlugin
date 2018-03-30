#pragma once

#include <stdint.h>
#include <list>
#include <queue>
extern "C"
{
#include "pthread.h"
}
#include "Constants.h"

#define INIT_INFO_QUEUE_SIZE 2
#define MAX_INFO_QUEUE_SIZE 10

class InteractivePIs;
class InteractivePIsSink;

class InteractivePIsFilter
{
public:
	InteractivePIsFilter();
	~InteractivePIsFilter();
};

class InteractivePIsSource
{
public:
	InteractivePIsSource();
	~InteractivePIsSource();
	virtual void registerInteractivePIsSink(InteractivePIsSink* sink);
	virtual void unregisterInteractivePIsSink(InteractivePIsSink* sink);
	virtual void writeInteractivePIs(InteractivePIs *infos);

private:
	pthread_mutex_t handlers_mutex;
	std::list<InteractivePIsSink*> handlers;
};

class InteractivePIsSink
{
public:
	virtual int onInteractivePIs(InteractivePIs *infos, InteractivePIsSource *nSource) = 0;
};

class InteractivePIs
{
public:
	InteractivePIs();
	~InteractivePIs();
public:
	int type;
	short px;
	short py;
	short width;
	short height;
	int pitch;
	int block_count;
	int blocks[PIS_BLOCKS_MAX_SIZE];
};

//class InteractivePIsPool
//{
//public:
//	InteractivePIsPool(int initSize, int maxSize);
//	~InteractivePIsPool();
//	int getValidSize();
//	int getEmptySize();
//	int getCurrent();
//	int gatMax();
//
//	void pushValid(InteractivePIs* infos);
//	InteractivePIs* popValid();
//
//	void pushEmpty(InteractivePIs* infos);
//	InteractivePIs* popEmpty();
//private:
//	std::queue<InteractivePIs *> validQueue;
//	std::queue<InteractivePIs *> emptyQueue;
//	pthread_mutex_t validQueueMutex;
//	pthread_mutex_t emptyQueueMutex;
//	volatile int validSize = 0;
//	volatile int currentSize = 0;
//	int maxSize = MAX_INFO_QUEUE_SIZE;
//};
