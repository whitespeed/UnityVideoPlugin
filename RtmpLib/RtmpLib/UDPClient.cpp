#include <stdio.h>  
#include <WINSOCK2.H>  
#include<iostream>
#include<string.h>
#include "pthread.h"
#include "UDPClient.h"
#include "P2PLog.h"

//#pragma comment(lib,"WS2_32.lib")
//#pragma comment(lib,"pthreadVC2.lib")

UDPClient::UDPClient()
{
}

UDPClient::~UDPClient()
{
}
void UDPClient::OnMsg(OutputType nType, ResultType nResult, char *msg)
{
    LOGI("UDPclient OnMsg: nType %d, nResult %d, msg %s", nType, nResult, msg);

    //m_pCallback->OnMsg(nType, nResult, msg);
}

void UDPClient::IntToChar(int value,char* src) {
    src[0] = (char)((value >> 24) & 0xFF);
    src[1] = (char)((value >> 16) & 0xFF);
    src[2] = (char)((value >> 8) & 0xFF);
    src[3] = (char)(value & 0xFF);
}

void UDPClient::ShortToChar(short value,char* src) {
    src[0] = (char)((value >> 8) & 0xFF);
    src[1] = (char)(value & 0xFF);
}

void UDPClient::HbMsgGenerator(int heartbeat_type)
{
    char temp_t[4] = {0};
    char temp_ht[4] = {0};

    memset(temp_t, 0, sizeof(temp_t));
    memset(temp_ht, 0, sizeof(temp_ht));
    //clear pre hbmsg:
    memset(bufsend, 0, sizeof(bufsend));
    IntToChar(htonl(UDPPACKET_CHECK_FLAG),temp_t);
    bufsend[0] = temp_t[0];
    bufsend[1] = temp_t[1];
    bufsend[2] = temp_t[2];
    bufsend[3] = temp_t[3];

    IntToChar(htonl(heartbeat_type), temp_ht);
    bufsend[4] = temp_ht[0];
    bufsend[5] = temp_ht[1];
    bufsend[6] = temp_ht[2];
    bufsend[7] = temp_ht[3];
}
int UDPClient::charToInt(char* src, int offset) {
	int value;
	value = (int)((src[offset+3] & 0xFF)
		| ((src[offset + 2] & 0xFF) << 8)
		| ((src[offset + 1] & 0xFF) << 16)
		| ((src[offset] & 0xFF) << 24));
	return value;
}

short UDPClient::charToshort(char* src, int offset) {
	short value;
	value = (short)((src[offset+1] & 0xFF)
		| ((src[offset] & 0xFF) << 8));
	return value;
}

int UDPClient::DecodeUDPpacket(char *buf, int len)
{
    int n = len;
	int blocksSize = 0;
    m_UDPPackt.check_flag = ntohl(charToInt(buf, 0));
    if (m_UDPPackt.check_flag == UDPPACKET_CHECK_FLAG)
    {
        m_UDPPackt.message_type = ntohl(charToInt(buf, 4));
        m_UDPPackt.message_index = ntohl(charToInt(buf, 8));
		LOGI("packet: check_flag = %d,message_type= %d,msg_index=%d", m_UDPPackt.check_flag, m_UDPPackt.message_type, m_UDPPackt.message_index);
		if (MSG_TYPE_SERVER_EXCEPTION == m_UDPPackt.message_type)
		{
			m_pre_index = m_UDPPackt.message_index;
			return 0;
		}
        //if ((n - UDPPACKET_MIN_SIZE >= 0) && ((n - UDPPACKET_MIN_SIZE) % 4 == 0))
        //{
            if (m_UDPPackt.message_index > m_pre_index)//index 重复，只取第一个的值,类似pts
            {
                if (m_UDPPackt.message_type == MSG_TYPE_DRAW)
                {
					if ((n - UDPPACKET_MIN_SIZE >= 0) && ((n - UDPPACKET_MIN_SIZE) % 4 == 0))
					{
						//m_UDPPackt.check_flag = ntohl(charToInt(buf, 0));
						//m_UDPPackt.message_type = ntohl(charToInt(buf, 4));
						//m_UDPPackt.message_index = ntohl(charToInt(buf, 8));
						m_UDPPackt.x = ntohs(charToshort(buf, 12));
						m_UDPPackt.y = ntohs(charToshort(buf, 14));
						if (m_UDPPackt.x < 0 || m_UDPPackt.y < 0)
						{
							LOGI("Error vip block x and y...");
							return 0;
						}
						m_UDPPackt.width = ntohs(charToshort(buf, 16));
						m_UDPPackt.height = ntohs(charToshort(buf, 18));
						if (m_UDPPackt.width <= 0 || m_UDPPackt.height <= 0)
						{
							LOGI("Error vip block width and height...");
							return 0;
						}
						m_UDPPackt.pitch = ntohl(charToInt(buf, 20));
						if (m_UDPPackt.pitch < 0)
						{
							LOGI("Error vip block pitch...");
							return 0;
						}
						blocksSize = len - PIS_HEADER_SIZE * sizeof(int);
						memcpy(m_UDPPackt.blocks, buf, blocksSize);
						for (int i = 0; i < blocksSize / sizeof(int); i++)
						{
							m_UDPPackt.blocks[i] = ntohl(charToInt(buf, 24 + i * 4));
						}
						LOGI("packet: area_x= %d,y=%d,w=%d,h=%d,pitch=%d, blockSize=%d", m_UDPPackt.x, m_UDPPackt.y, m_UDPPackt.width, m_UDPPackt.height, m_UDPPackt.pitch, blocksSize);
						m_infos.type = m_UDPPackt.message_type;
						m_infos.px = m_UDPPackt.x;
						m_infos.py = m_UDPPackt.y;
						m_infos.width = m_UDPPackt.width;
						m_infos.height = m_UDPPackt.height;
						m_infos.pitch = m_UDPPackt.pitch;
						m_infos.block_count = blocksSize / sizeof(int);
						memcpy(m_infos.blocks, m_UDPPackt.blocks, blocksSize);
						writeInteractivePIs(&m_infos);
					}
                }
                else if (m_UDPPackt.message_type == MSG_TYPE_CLEAR)
                {
                    m_infos.px = 0;
                    m_infos.py = 0;
                    m_infos.width = 0;
                    m_infos.height = 0;
                    m_infos.pitch = 0;
                    m_infos.block_count = 0;
                    memset(m_infos.blocks, 0, PIS_BLOCKS_MAX_SIZE);
                    m_infos.type = m_UDPPackt.message_type;
                    writeInteractivePIs(&m_infos);
                }
				m_pre_index = m_UDPPackt.message_index;
            }
        //}
    }
    return 0;
}
void* UDPClient::heartbeat(void *data)
{
    UDPClient* udpclient = (UDPClient*)data;
    udpclient->doheartbeat();
    return 0;
}
void* UDPClient::doheartbeat()
{
    LOGI("Thread: enter doheartbeat");
    int64_t time_begin, time_end;
    time_begin = av_gettime();
    time_end = time_begin;
    HbMsgGenerator(m_heartbeat_type);
    bool flag = true;
    while (m_isRunning && m_heartbeat_type == MSG_TYPE_HEART)
    {
        //心跳间隔1000ms
        if ((int64_t)((time_end - time_begin)/1000) >= HEARTBEAT_INTERVAL_MICROSEC && (int64_t)((time_end - time_begin) / 1000) <= HEARTBEAT_INTERVAL_MICROSEC_MAX)
        {
            time_begin = time_end;
            flag = true;
        }

        if (flag)
        {
            flag = false;
            int send_len = sendto(sockfd, bufsend, sizeof(bufsend), 0, (SOCKADDR *)&serverAddr, socklen);
            LOGI("doheartbeat Thread: sendto buf len = %d", send_len);
        }
        Sleep(10);
        time_end = av_gettime();
    }
    //最后断开前一定要保证能发送出去：
    if (m_heartbeat_type == MSG_TYPE_DISCONNECT)
    {
        HbMsgGenerator(m_heartbeat_type);
        sendto(sockfd, bufsend, sizeof(bufsend), 0, (SOCKADDR *)&serverAddr, socklen);
        OnMsg(OT_UDPCLIENT, RT_END, "stop doheartbeat end");
        return 0;
    }
    OnMsg(OT_UDPCLIENT, RT_END, "doheartbeat end");
    return 0;
}
void* UDPClient::client(void *data)
{
	UDPClient* udpclient = (UDPClient*)data;
	udpclient->doClient();
	return 0;
}
void* UDPClient::doClient()
{
	LOGI("Thread: enter client");
    int buflen = 0;
    char buf_recv[UDPPACKET_MAX_SIZE];

    while (m_isRunning)
    {
        buflen = recvfrom(sockfd, buf_recv, UDPPACKET_MAX_SIZE, 0, (SOCKADDR *)&serverAddr, &socklen);
        if (buflen > 0) {
            DecodeUDPpacket(buf_recv, buflen);
        }
    }
    if (m_heartbeat_type == MSG_TYPE_DISCONNECT)
    {
        OnMsg(OT_UDPCLIENT, RT_END, "stop heartbeat doClient end");
        return 0;
    }
    OnMsg(OT_UDPCLIENT, RT_END, "doClient end");
    return 0;
}
int UDPClient::start_client(char *myip, unsigned short port)
{
	serverport = port;
    strcpy(server_ip,myip);
    OnMsg(OT_UDPCLIENT, RT_OK, "do vip stream start_client");
    WSADATA wsaData;//初始化Socket
    int init_ret = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (init_ret != 0) 
    {
        OnMsg(OT_UDPCLIENT, RT_FAIL, "start_client WSAStartup fail...");
        return -1;
    }
    sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sockfd == INVALID_SOCKET)
    {
        OnMsg(OT_UDPCLIENT, RT_FAIL, "start_client sockfd fail...");
        WSACleanup();
        return -1;
    }
    socklen = sizeof(serverAddr);
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(serverport);
    serverAddr.sin_addr.s_addr = inet_addr(server_ip);//server IP

    //设置timeout :
    int timeout = CLIENT_SEND_TIMEOUT;//5s
    int timeout_rec = CLIENT_RECV_TIMEOUT; //超时时间影响线程结束等待时间，recv阻塞的，需要设置短些，不然界面停止会卡住。
    int ret_send = setsockopt(sockfd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&timeout, sizeof(timeout));
    int ret_recv = setsockopt(sockfd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout_rec, sizeof(timeout_rec));

	m_isRunning = true;
    m_heartbeat_type = MSG_TYPE_HEART;
    pthread_create(&thread_heart, NULL, heartbeat, this);
    pthread_create(&thread, NULL, client, this);
    return 0;
}
int UDPClient::stop_client()
{
	m_isRunning = false;
    m_heartbeat_type = MSG_TYPE_DISCONNECT;
    pthread_join(thread_heart, NULL);
    pthread_join(thread, NULL);
    closesocket(sockfd);
    //释放资源，退出
	LOGI("Exiting udpclient");
    WSACleanup();
    return 0;
}