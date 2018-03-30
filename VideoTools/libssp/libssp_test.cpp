// libssp_test.cpp : 定义控制台应用程序的入口点。
//

#include <functional>
#include <thread>
#include "imf/net/loop.h"
#include "imf/net/threadloop.h"
#include "imf/ssp/sspclient.h"

using namespace std::placeholders;

#ifdef _DEBUG
#pragma comment (lib, "libsspd.lib")
#else
#pragma comment (lib, "libssp.lib")
#endif
auto start = clock();
imf::ThreadLoop * threadLooper = nullptr;
imf::SspClient * client = nullptr;
static void on_264_1(uint8_t * data, size_t len, uint64_t pts, uint32_t frm_no, uint32_t type)
{
	printf("on 1 H264 pts=%d, frm_no =%d delta=%f(ms).  type=%d \n", pts, frm_no, (double)(clock() - start) / CLOCKS_PER_SEC * 1000, type);
	start = clock();
    //printf("on 1 H264 %d [%d] [%lld]\n", frm_no, type, len);
    //std::this_thread::sleep_for(std::chrono::seconds(45));
}

static void on_264_2(uint8_t * data, size_t len, uint64_t pts, uint32_t frm_no, uint32_t type)
{
    //printf("on 2 264 %d [%d] [%lld]\n", frm_no, type, len);
    //std::this_thread::sleep_for(std::chrono::seconds(45));
}

//static void on_264_3(uint8_t * data, size_t len, uint64_t pts, uint32_t frm_no, uint32_t type)
//{
//    printf("on 3 264 %d [%d] [%lld]\n", frm_no, type, len);
//    //std::this_thread::sleep_for(std::chrono::seconds(45));
//}
//
//static void on_264_4(uint8_t * data, size_t len, uint64_t pts, uint32_t frm_no, uint32_t type)
//{
//    printf("on 4 264 %d [%d] [%lld]\n", frm_no, type, len);
//    //std::this_thread::sleep_for(std::chrono::seconds(45));
//}

static void on_meta(struct imf::SspVideoMeta *v, struct imf::SspAudioMeta *a, struct imf::SspMeta *s)
{
    printf("on meta\n %d %d %d %d",v->width,v->height,v->timescale,v->unit,v->gop);
}
static void on_disconnect()
{
    printf("on disconnect\n");
}

static void setup(imf::Loop *loop)
{
    std::string ip = "172.29.1.83";
    imf::SspClient * client = new imf::SspClient(ip, loop, 0x400000);
    client->init();
    client->setOnH264DataCallback(std::bind(on_264_1, _1, _2, _3, _4, _5));
    client->setOnMetaCallback(std::bind(on_meta, _1, _2, _3));
    client->setOnDisconnectedCallback(std::bind(on_disconnect));

    client->start();
	//Sleep(10000);
    ////////////////////
    //client = new imf::SspClient(std::string("172.29.2.69"), loop, 0x400000);
    //client->init();

    //client->setOnH264DataCallback(std::bind(on_264_2, _1, _2, _3, _4, _5));
    //client->setOnMetaCallback(std::bind(on_meta, _1, _2, _3));
    //client->setOnDisconnectedCallback(std::bind(on_disconnect));

    //client->start();
#if 0
    ////////////////////
    client = new imf::SspClient(std::string("10.98.32.3"), loop, 0x400000);
    client->init();

    client->setOnH264DataCallback(std::bind(on_264_3, _1, _2, _3, _4, _5));
    client->setOnMetaCallback(std::bind(on_meta, _1, _2));
    client->setOnDisconnectedCallback(std::bind(on_disconnect));

    client->start();

    ////////////////////
    client = new imf::SspClient(std::string("10.98.32.4"), loop, 0x400000);
    client->init();

    client->setOnH264DataCallback(std::bind(on_264_4, _1, _2, _3, _4, _5));
    client->setOnMetaCallback(std::bind(on_meta, _1, _2));
    client->setOnDisconnectedCallback(std::bind(on_disconnect));

    client->start();
#endif
    // add more SspClient if you need
}

int main()
{
	threadLooper = new imf::ThreadLoop(std::bind(setup, _1));
    threadLooper->start();
	
    while (1);
	//Sleep(10000);
	//threadLooper->stop();
	system("pause");
    return 0;
}

