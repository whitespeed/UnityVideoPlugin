/**
中超节目信息交互定义

交互流程
1.	字幕机启动，确定IP与监听接口
2.	VIP工作站启动，填入字幕机IP信息
3.	VIP工作站定期向字幕机发送心跳信息
4.	字幕机将节目信息发送给VIP机位
5.	VIP接收节目信息后，渲染到画面中

字幕机停止向某VIP机位发送信息条件：
1.	收到VIP机位主动发送的断开信号
2.	心跳超时

交互方式采用UDP协议，每个消息发送3次以保证消息可达。

信息格式：
0                   1                   2                   3
0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+ -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| check flag |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| message type |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| message index |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


所有信息采用网络字节序

checkflag 0xffffffff
message type： 0 为心跳， 1为断开， 2为绘画信息, 3为清除绘画信息
message index：指令index，从0 开始递增。VIP机位收到重复的index时，只处理第一个。

0                   1                   2                   3
0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+ -+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| check flag |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| message type |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| message index |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| x | y |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| width | height |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| pitch |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| block 1 |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| block 2 |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| block 3 |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| block n |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

绘画信息描述字段
x，y 绘画信息叠加到视频区域的位置
width，height 每个块的宽高
pitch 每一行块的数量
block 采用RGBA表示的颜色//交互数据包
**/
#include "InteractivePIsFilter.h"
#include <WINSOCK2.H>  
//#include "Constants.h"
#define HEARTBEAT_INTERVAL_MICROSEC 3000//心跳间隔误差3000 - 3030ms
#define HEARTBEAT_INTERVAL_MICROSEC_MAX 3030
#define CLIENT_SEND_TIMEOUT 5000   //5s
#define CLIENT_RECV_TIMEOUT 1000   //1s

#define UDPPACKET_MAX_SIZE (sizeof(UDPpacket))
#define UDPPACKET_MIN_SIZE (sizeof(UDPpacket) - (PIS_BLOCKS_MAX_SIZE - 1)*sizeof(int))
#define UDPPACKET_CHECK_FLAG 0xffffffff

typedef struct UDPpacket
{
 
    int check_flag;      //0xffffffff
    int message_type;    //0 为心跳， 1为断开， 2为绘画信息, 3为清除绘画信息
    int message_index;   //指令index，从0 开始递增。VIP机位收到重复的index时，只处理第一个。
    short x;               //绘画信息叠加到视频区域的位置x
    short y;               //绘画信息叠加到视频区域的位置y
    short width;           //每个块的宽高
    short height;
    int pitch;           //每一行块的数量  
	int blocks[PIS_BLOCKS_MAX_SIZE]; //采用RGBA表示的颜色
} UDPpacket;

class UDPClient:public InteractivePIsSource, public VROutputCallback
{
private:

    int loop_flag = 0;
	char server_ip[32];
	bool m_isRunning = false;
	unsigned short serverport;
    int m_pre_index = -0x7FFFFFFF;
	SOCKET sockfd;
	pthread_t thread;
    pthread_t thread_heart;
	UDPpacket m_UDPPackt;
	InteractivePIs m_infos;

    char bufsend[8] = { 0 };
    int m_heartbeat_type = MSG_TYPE_HEART;

    sockaddr_in serverAddr;
    int socklen;
public:
	UDPClient();
	~UDPClient();
	int start_client(char *myip, unsigned short port);
	int stop_client();
private:
	int DecodeUDPpacket(char *buf, int len);
    static void* heartbeat(void *data);
	static void* client(void *data);
	void* doClient();
    void* doheartbeat();
	int charToInt(char* src, int offset);
	short charToshort(char* src, int offset);

    void IntToChar(int value, char* src);
    void ShortToChar(short value, char* src);
    void HbMsgGenerator(int heartbeat_type); //0 for heartbeat ,1 for stop beat
    void OnMsg(OutputType nType, ResultType nResult, char *msg);
};