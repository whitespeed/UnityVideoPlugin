//#include "stdafx.h"
#include "MessageDelivery.h"

MessageHandlerInterface* MessageDelivery::mHandler;

MessageDelivery::MessageDelivery()
{
}

MessageDelivery::~MessageDelivery()
{
}

void MessageDelivery::sendMessage(int what, int extra)
{
    if (mHandler)
        mHandler->onMessage(what, extra);
}

void MessageDelivery::setMessageHander(MessageHandlerInterface* handler)
{
    mHandler = handler;
}

std::string MessageDelivery::messageToStr(int what, int extra)
{
    switch (what)
    {
    case 0:
        return "";
    case EventDeviceStartSuc:
        return "设备启动成功";
    case EventDeviceStartFailed:
        return "设备启动失败";
    case EventDeviceConnectSuc:
        return "设备连接成功";
	case EventDeviceDisConnect:
		return "设备连接断开";
    case EventStreamPrepare:
        return "输出准备";
    case EventStreamStart:
        return "输出开始";
    case EventStreamEnd:
        return "输出结束";
    case EventStreamFailed:
        return "输出失败";
    case EventEncodePrepare:
        return "编码准备";
    case EventEncodeStart:
        return "编码开始";
    case EventEncodeEnd:
        return "编码结束";
    case EventEncodeFailed:
        return "编码失败";
    case EventOutputEnd:
        return "所有输出结束";
    case EventStandardizationEnd:
        return "镜头标定完成......";
    case EventGeometricEnd:
        return "几何位置计算完成......";
    case EventMapEnd:
        return "映射表生成完成......";
    case EventSeamcutEnd:
        return "拼接缝计算完成......";
    case EventCalibrationEnd:
        return "校准过程完成,进入融合过程......";
    case EventStitchinitEnd:
        return "融合初始化完成......";
    case EventInputDeivceParamError:
        return "输入设备参数错误";
    case ErrorStreamIOOpen:
        return "输出流打开失败";
    case ErrorStreamIOSend:
        return "输出流写入失败";
    case ErrorEncoderVideo:
        return "ErrorEncoderVideo";
    case ErrorEncoderVideoInit:
        return "视频编码器初始化失败";
    case ErrorEncoderVideoEncode:
        return "视频编码失败";
	case ErrorStreamVideoInit:
		return "视频输出流初始化失败";
	case ErrorStreamAudioInit:
		return "音频输出流初始化失败";
    case ErrorOutputVideoQueue:
        return "OutputManager视频帧缓冲池已满";
    case ErrorEncoderVideoQueue:
        return "Encoder视频帧缓冲池已满";
    case ErrorStreamVideoQueue:
        return "Stream视频帧缓冲池已满";
    case ErrorEncoderAudio:
        return "音频重采样失败";
    case ErrorEncoderAudioInit:
        return "音频编码器初始化失败";
    case ErrorEncoderAudioEncode:
        return "音频编码失败";
    case ErrorSdiInit:
        return "SDI初始化失败";
    case ErrorSdiDevInit:
        return "SDI卡初始化失败";
    case ErrorSdiInputWidth:
        return "SDI输入源width过大";
    case ErrorGeometric:
        return "几何位置计算结果不理想，可能是因为控制点不足或错误导致的，请尝试重新进行计算......";
    case ErrorState:
        return "状态错误";
	case ErrorVipStreamerQueue:
		return "VIP视频帧缓冲池已满";
	case EventDeviceConnectFailed:
		return "连接失败";
    case EventDeviceConnect:
        return "正在尝试连接设备";
    case ErrorInvalidCube:
        return "非法的CUBE文件";
    default:
        return "";
    }
}