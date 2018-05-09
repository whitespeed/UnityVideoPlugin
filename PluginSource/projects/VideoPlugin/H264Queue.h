#include <list>
#include <mutex>
#include <string>
#include <iostream>
#include <stdio.h>


struct H264Data
{
	uint8_t* data;
	size_t len;
	uint32_t type;
	uint32_t pts;
	uint32_t frm_no;
	virtual ~H264Data() {
		len = 0;
		frm_no = -1;
		type = -1;
		pts = -1;
		if (data != NULL)
			delete[] data;
		data = NULL;
	}
}; H264Data;

class H264Queue
{
public:
	H264Queue()
	{

	}
	~H264Queue()
	{
		release();
	}
	void queue(H264Data* data)
	{
		std::lock_guard<std::mutex> lock(mOpMutex);
		mDataList.push_back(data);
	}
	H264Data* dequeue()
	{
		std::lock_guard<std::mutex> lock(mOpMutex);
		if (mDataList.size() <= 0)
			return NULL;

		H264Data* data = mDataList.front();
		mDataList.pop_front();
		return data;
	}
	size_t size()
	{
		std::lock_guard<std::mutex> lock(mOpMutex);
		return mDataList.size();
	}
	void release()
	{
		std::lock_guard<std::mutex> lock(mOpMutex);
		if (!mDataList.empty())
			mDataList.clear();
	}
private:
	std::list<H264Data*> mDataList;
	std::mutex mOpMutex;
};


static H264Data* pack_h264_data(uint8_t* d, size_t l, uint32_t p, uint32_t f, uint32_t t)
{
	H264Data* re = new H264Data();
	re->len = l;
	re->frm_no = f;
	re->type = t;
	re->pts = p;
	if (l > 0 && d != NULL)
	{
		re->data = new uint8_t[l];
		memcpy(re->data, d, l);
	}
	else
	{
		re->len = 0;
		re->data = NULL;
	}
	return re;
}
static void release_h264_data(H264Data* data)
{
	if (data != NULL)
	{
		delete data;
		data = NULL;
	}
}