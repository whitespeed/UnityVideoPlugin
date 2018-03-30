#ifndef __MORETV_UTIL_P2PLOG_H__
#define __MORETV_UTIL_P2PLOG_H__

#define LOGTOFILE

#ifdef ANDROID

#include <android/log.h>
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

#else //ANDROID
extern void p2p_log_start();
extern void p2p_log_end();
extern int p2p_log_print(int prio, const char *tag, int line, const char *fmt, ...);

#define P2P_LOG_VERBOSE 1
#define P2P_LOG_DEBUG 2
#define P2P_LOG_INFO 3
#define P2P_LOG_WARN 4
#define P2P_LOG_ERROR 5

#ifdef _DEBUG
#define P2P_LOG_LEVEL P2P_LOG_DEBUG
#else
#define P2P_LOG_LEVEL P2P_LOG_INFO
#endif

#if P2P_LOG_VERBOSE < P2P_LOG_LEVEL
#define LOGV(...)
#else
#define LOGV(...) p2p_log_print(P2P_LOG_VERBOSE, __FILE__, __LINE__, __VA_ARGS__)
#endif

#if P2P_LOG_DEBUG < P2P_LOG_LEVEL
#define LOGD(...)
#else
#define LOGD(...) p2p_log_print(P2P_LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#endif

#if P2P_LOG_INFO < P2P_LOG_LEVEL
#define LOGI(...)
#else
#define LOGI(...) p2p_log_print(P2P_LOG_INFO, __FILE__, __LINE__, __VA_ARGS__)
#endif

#if P2P_LOG_WARN < P2P_LOG_LEVEL
#define LOGW(...)
#else
#define LOGW(...) p2p_log_print(P2P_LOG_WARN, __FILE__, __LINE__, __VA_ARGS__)
#endif

#if P2P_LOG_ERROR < P2P_LOG_LEVEL
#define LOGE(...)
#else
#define LOGE(...) p2p_log_print(P2P_LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#endif

#endif//ANDROID

#endif //__MORETV_UTIL_P2PLOG_H__
