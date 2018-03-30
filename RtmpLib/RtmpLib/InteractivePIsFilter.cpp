#include "InteractivePIsFilter.h"
#define TAG "InteractivePIs"
#include "P2PLog.h"

InteractivePIsFilter::InteractivePIsFilter()
{
}

InteractivePIsFilter::~InteractivePIsFilter()
{
}

InteractivePIsSource::InteractivePIsSource()
{
	pthread_mutex_init(&handlers_mutex, NULL);
}

InteractivePIsSource::~InteractivePIsSource()
{
	pthread_mutex_destroy(&handlers_mutex);
}

void InteractivePIsSource::registerInteractivePIsSink(InteractivePIsSink* sink)
{
	bool already_in = false;
	pthread_mutex_lock(&handlers_mutex);
	handlers.push_back(sink);
	handlers.unique();
	pthread_mutex_unlock(&handlers_mutex);
}

void InteractivePIsSource::unregisterInteractivePIsSink(InteractivePIsSink* sink)
{
	pthread_mutex_lock(&handlers_mutex);
	handlers.remove(sink);
	pthread_mutex_unlock(&handlers_mutex);
}

void InteractivePIsSource::writeInteractivePIs(InteractivePIs *infos)
{
	pthread_mutex_lock(&handlers_mutex);
	std::list<InteractivePIsSink*>::iterator iter = handlers.begin();
	for (; iter != handlers.end(); iter++) {
		if ((*iter)->onInteractivePIs(infos, this) < 0)
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

InteractivePIs::InteractivePIs()
{
	type = MSG_TYPE_NONE;
	px = 0;
	py = 0;
	width = 0;
	height = 0;
	pitch = 0;
	block_count = 0;
	memset(blocks, 0, PIS_BLOCKS_MAX_SIZE);
	////htf_test
	//valid = true;
	//px = 2140;
	//py = 50;
	//width = 50;
	//height = 80;
	//pitch = 10;
	//block_count = 50;
	//memset(blocks, 0, PIS_BLOCKS_MAX_SIZE);
	//for (int i = 0; i < block_count; i++)
	//{
	//	blocks[i] = 0 + rand() % 0xffffff;
	//}
}

InteractivePIs::~InteractivePIs()
{

}

//InteractivePIsPool::InteractivePIsPool(int initSize, int maxSize)
//{
//	if (initSize < 0) {
//		initSize = INIT_INFO_QUEUE_SIZE;
//	}
//
//	if (maxSize < 0)
//	{
//		maxSize = MAX_INFO_QUEUE_SIZE;
//	}
//
//	this->validSize = 0;
//	this->currentSize = initSize;
//	this->maxSize = maxSize;
//
//	for (int i = 0; i < initSize; i++)
//	{
//		InteractivePIs* temp = new InteractivePIs();
//		emptyQueue.push(temp);
//	}
//	pthread_mutex_init(&validQueueMutex, NULL);
//	pthread_mutex_init(&emptyQueueMutex, NULL);
//}
//
//InteractivePIsPool::~InteractivePIsPool()
//{
//	pthread_mutex_lock(&validQueueMutex);
//	while (!validQueue.empty())
//	{
//		InteractivePIs* temp = validQueue.front();
//		validQueue.pop();
//		delete temp;
//	}
//	pthread_mutex_unlock(&validQueueMutex);
//
//	pthread_mutex_lock(&emptyQueueMutex);
//	while (!emptyQueue.empty())
//	{
//		InteractivePIs* temp = emptyQueue.front();
//		emptyQueue.pop();
//		delete temp;
//	}
//	pthread_mutex_unlock(&emptyQueueMutex);
//
//	pthread_mutex_destroy(&validQueueMutex);
//	pthread_mutex_destroy(&emptyQueueMutex);
//}
//
//int InteractivePIsPool::getValidSize()
//{
//	return validQueue.size();
//}
//
//int InteractivePIsPool::getEmptySize()
//{
//	return emptyQueue.size();
//}
//
//int InteractivePIsPool::getCurrent()
//{
//	return currentSize;
//}
//
//int InteractivePIsPool::gatMax()
//{
//	return maxSize;
//}
//
//void InteractivePIsPool::pushValid(InteractivePIs* infos)
//{
//	pthread_mutex_lock(&validQueueMutex);
//	validQueue.push(infos);
//	pthread_mutex_unlock(&validQueueMutex);
//}
//
//InteractivePIs* InteractivePIsPool::popValid()
//{
//	InteractivePIs* ret = nullptr;
//	pthread_mutex_lock(&validQueueMutex);
//	if (validQueue.size() > 0) {
//		ret = validQueue.front();
//		validQueue.pop();
//	}
//	pthread_mutex_unlock(&validQueueMutex);
//	return ret;
//}
//
//void InteractivePIsPool::pushEmpty(InteractivePIs* infos)
//{
//	pthread_mutex_lock(&emptyQueueMutex);
//	infos->m_x = 0;
//	infos->m_y = 0;
//	infos->m_w = 0;
//	infos->m_h = 0;
//	infos->m_pitch = 0;
//	memset(infos->m_blocks, 0, PIS_BLOCKS_MAX_SIZE);
//	emptyQueue.push(infos);
//	pthread_mutex_unlock(&emptyQueueMutex);
//}
//
//InteractivePIs* InteractivePIsPool::popEmpty()
//{
//	InteractivePIs* ret = nullptr;
//
//	pthread_mutex_lock(&emptyQueueMutex);
//	if (emptyQueue.size() > 0) {
//		ret = emptyQueue.front();
//		emptyQueue.pop();
//	}
//	pthread_mutex_unlock(&emptyQueueMutex);
//
//	if (ret == nullptr && currentSize < maxSize)
//	{
//		currentSize++;
//		LOGI("InteractivePIsPool: new InteractivePIs, current size:%d\n", currentSize);
//		ret = new InteractivePIs();
//	}
//	return ret;
//}