#pragma once

//区分360度和180度直播的类型
typedef enum VrType
{
    VR180 = 0,
    VR360,
}VrType;


typedef enum SourceType
{
    NONE_SOURCE = -1,
    PANO_SOURCE = 0,
    PROCPOST_SOURCE = 1,
}SourceType;

typedef enum SdiScaleType
{
    ORI_SCALE = 0, //保持原始比例不拉伸之后填充，360度保持底部色，180度默认底部和两边色其他色可选 填充
    CHANGE_SCALE, //拉伸至高1080 or 2160后填充，360度是满屏，180度两边默认黑色其他色可选 拉伸
}SdiScaleType;

struct StreamOpt {
    int w;//分辨率
    int h;
    int vb;//视频码率
    int ab;//音频码率
    int vc;//视频编码
    int ac;//音频编码
    int sdiad = 0; //整体延迟
    int devid = 0; //SDI编号
    long adms; //编码推流延迟 音频延迟0
    long fillcol = 0; //vr180填充颜色
    SourceType source = PROCPOST_SOURCE;
    char uri[1024] = { 0 };
    //char file[1024];
    char sdilogo[1024] = { 0 };
    SdiScaleType scaletype = ORI_SCALE;
    bool isvip = false;
    char ip[64] = { 0 };
    short port;
};


typedef enum OUTPUT_STATUS
{
    STATUS_IDLE = 0,
    STATUS_PREPARING,
    STATUS_PREPARED,
    STATUS_START,
    STATUS_RETRYING,
    STATUS_STOP,
    STATUS_END,
    STATUS_FAIL
}OUTPUT_STATUS;


//
typedef enum OutputType
{
    OT_UDPCLIENT = -4,
    OT_VIPSTREAMER = -3,
    OT_MANAGER = -2,
    OT_ENCODER = -1,
    OT_FILE = 0,
    OT_RTMP = 1,
    OT_SDI = 2
}OutputType;

//
typedef enum ResultType
{
    RT_FAIL = -1,
    RT_OK = 0,
    RT_RETRYING,
    RT_END
}ResultType;

class VROutputCallback
{
public:
    virtual void OnMsg(OutputType nType, ResultType nResult, char *msg) {};
};