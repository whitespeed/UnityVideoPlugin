#pragma once

#include <functional>

#ifdef FULL_SOURCE
#define LIBSSP_API
#else
#ifdef LIBSSP_EXPORTS
#define LIBSSP_API __declspec(dllexport)
#else
#define LIBSSP_API __declspec(dllimport)
#endif
#endif

#define ERROR_SSP_PROTOCOL_VERSION_GT_SERVER        (-1000)
#define ERROR_SSP_PROTOCOL_VERSION_LT_SERVER        (-1001)
#define ERROR_SSP_CONNECTION_FAILED                 (-1002)
#define ERROR_SSP_CONNECTION_EXIST                  (-1003)

namespace imf
{

struct LIBSSP_API SspVideoMeta {
    uint32_t width;
    uint32_t height;
    uint32_t timescale;
    uint32_t unit;
    uint32_t gop;
};

struct LIBSSP_API SspAudioMeta {
    uint32_t timescale;
    uint32_t unit;
    uint32_t sample_rate;
    uint32_t sample_size;
    uint32_t channel;
    uint32_t bitrate;
};

struct LIBSSP_API SspMeta {
    bool pts_is_wall_clock;
};
typedef std::function <void(void)> OnRecvBufferFullCallback;
typedef std::function <void(void)> OnDisconnectedCallback;
typedef std::function <void(uint8_t * data, size_t len, uint64_t pts, uint32_t frm_no, uint32_t type)> OnH264DataCallback;
typedef std::function <void(uint8_t * data, size_t len, uint64_t pts)> OnAudioDataCallback;
typedef std::function <void(struct SspVideoMeta*, struct SspAudioMeta*, struct SspMeta *)> OnMetaCallback;
typedef std::function <void(int code, const char* description)> OnExceptionCallback;

class Loop;
class SspClientImp;
class LIBSSP_API SspClient
{
public:
    SspClient(std::string & ip, Loop *loop, size_t bufSize);
    ~SspClient();

    int init(void);
    int start(void);
    int stop(void);

    void setOnRecvBufferFullCallback(const OnRecvBufferFullCallback & cb);
    void setOnH264DataCallback(const OnH264DataCallback & cb);
    void setOnAudioDataCallback(const OnAudioDataCallback & cb);
    void setOnMetaCallback(const OnMetaCallback & cb);
    void setOnDisconnectedCallback(const OnDisconnectedCallback & cb);
    void setOnExceptionCallback(const OnExceptionCallback & cb);
private:
    SspClientImp *imp_;
};
}