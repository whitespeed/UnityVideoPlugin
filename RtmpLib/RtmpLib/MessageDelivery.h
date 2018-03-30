#pragma once

#include <string>

#define EVENTERROR(n) 0-n

//COMMON EVENT
#define ErrorState EVENTERROR(1000)
//DEVICE EVENT
#define EventDeviceStartSuc 100001
#define EventDeviceStartFailed EVENTERROR(100002)
#define EventDeviceConnect  100003
#define EventDeviceConnectSuc 100004
#define EventDeviceConnectFailed EVENTERROR(100005)
#define EventDeviceDisConnect 100005
#define EventInputDeivceParamError EVENTERROR(100006)
#define EventCacheError EVENTERROR(100007)
#define EventDeivceChange 100008

//MIDDLE EVENT
#define EventStandardizationEnd 200001
#define EventGeometricEnd		200002
#define EventMapEnd				200003
#define EventSeamcutEnd			200004
#define EventCalibrationEnd		200005
#define EventStitchinitEnd		200006
#define EventStitchOver			200007
#define EventStitchStop			200008

#define EventPostprocStart		300001
#define EventPostprocStop		300002
#define EventPostproc_z			300003
#define EventPostproc_x			300004
#define EventPostproc_y			300005

//STREAM EVENT
#define EventStreamPrepare 400000
#define EventStreamStart 400001
#define EventStreamEnd 400002
#define EventStreamFailed 400003

#define EventEncodePrepare 500000
#define EventEncodeStart 500001
#define EventEncodeEnd 500002
#define EventEncodeFailed 500003

#define EventEncodeFileFPS 600000
#define EventEncodeFileBPS 600001
#define EventEncodeRtmpFPS 600002
#define EventEncodeRtmpBPS 600003

#define EventOutputStart 700001
#define EventOutputEnd 700002
#define EventOutputFailed 700003





#define EventFpsShow 800006
#define EventFileRecord 800007
#define EventRtmpPush 800008
#define EventSDIPush 800009
#define EventPIPEPush 800010

#define EventStopViewCallBack 900010
#define EventStartViewCallBack 900011



#define ErrorStreamIOOpen -400001
#define ErrorStreamIOSend -400002
#define ErrorStreamIOStop -400003

#define ErrorEncoderVideo     -410001
#define ErrorEncoderVideoInit -410002
#define ErrorEncoderVideoEncode -410003
#define ErrorEncoderVideoQueue    -410004

#define ErrorEncoderVideoPacket     -410005
#define ErrorEncoderVideoPacketInit -410006
#define ErrorEncoderVideoPacketEncode -410007
#define ErrorEncoderVideoPacketQueue    -410008

#define ErrorEncoderAudio -420001
#define ErrorEncoderAudioInit -420002
#define ErrorEncoderAudioEncode -420003

#define ErrorEncoderAudioPacket -420004
#define ErrorEncoderAudioPacketInit -420005
#define ErrorEncoderAudioPacketEncode -420006
#define ErrorEncoderAudioPacketQueue    -410007

#define ErrorOutputAudio -490001
#define ErrorOutputAudioInit -490002
#define ErrorOutputVideoQueue -490003

#define ErrorStreamVideo     -510001
#define ErrorStreamVideoInit -510002
#define ErrorStreamVideoEncode -510003
#define ErrorStreamVideoQueue    -510004

#define ErrorStreamAudio -520001
#define ErrorStreamAudioInit -520002
#define ErrorStreamAudioEncode -520003
#define ErrorStreamAudioQueue    -520004

#define ErrorVipStreamerQueue -530001

#define ErrorSdiInit -430001
#define ErrorSdiDevInit -430002
#define ErrorSdiInputWidth -430003
#define ErrorGeometric -900000
#define ErrorInvalidCube -900001

class MessageHandlerInterface
{
public:
    virtual void onMessage(int what, int extra) = 0;
};

class MessageDelivery
{
public:
    MessageDelivery();
    ~MessageDelivery();

    static void sendMessage(int what, int extra);
    static void setMessageHander(MessageHandlerInterface* handler);
    static std::string messageToStr(int what, int extra = 0);

private:
    static MessageHandlerInterface* mHandler;
};
