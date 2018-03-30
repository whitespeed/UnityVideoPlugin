using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.UI;

namespace UnityPlugin.Multimedia
{
    public class NativeClass
    {
        public enum DecoderState
        {
            INIT_FAIL = -2,
            STOP,
            NOT_INITIALIZED,
            INITIALIZING,
            INITIALIZED,
            START,
            PAUSE,
            SEEK_FRAME,
            BUFFERING,
            EOF
        }

        public const string NATIVE_LIBRARY_NAME = "NativeDecoder";


        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern int nativeCreateDecoder(string filePath, ref int id);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern int nativeCreateDecoderAsync(string filePath, ref int id);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern void nativeCreateTexture(int id, ref IntPtr tex0, ref IntPtr tex1, ref IntPtr tex2);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern void nativeDestroyDecoder(int id);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern void nativeFreeAudioData(int id);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern float nativeGetAudioData(int id, ref IntPtr output, ref int lengthPerChannel);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern void nativeGetAudioFormat(int id, ref int channel, ref int frequency, ref float totalTime);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern int nativeGetDecoderState(int id);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern int nativeGetMetaData(string filePath, out IntPtr key, out IntPtr value);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern void nativeGetVideoFormat(int id, ref int width, ref int height, ref float totalTime);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern bool nativeIsAudioEnabled(int id);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern bool nativeIsContentReady(int id);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern bool nativeIsEOF(int id);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern bool nativeIsSeekOver(int id);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern bool nativeIsVideoBufferEmpty(int id);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern bool nativeIsVideoBufferFull(int id);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern bool nativeIsVideoEnabled(int id);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern void nativeLoadThumbnail(int id, float time, IntPtr texY, IntPtr texU, IntPtr texV);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern void nativeSetAudioAllChDataEnable(int id, bool isEnable);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern void nativeSetAudioEnable(int id, bool isEnable);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern void nativeSetSeekTime(int id, float sec);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern void nativeSetVideoEnable(int id, bool isEnable);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern void nativeSetVideoTime(int id, float currentTime);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern bool nativeStartDecoding(int id);

        [DllImport(NATIVE_LIBRARY_NAME)]
        public static extern void nativeRegistLogHandler(IntPtr f);

        public static void nativeLogHandler(string str)
        {
            Debug.LogFormat("[Native] {0}", str);
        }

        [DllImport(NativeClass.NATIVE_LIBRARY_NAME)]
        public static extern IntPtr GetRenderEventFunc();
    }


    public class MediaDecoder : MonoBehaviour
    {
        private const string VERSION = "1.2";

        private const string LOG_TAG = "[MediaDecoder]";

        private const int AUDIO_FRAME_SIZE = 2048; //  Audio clip data size. Packed from audioDataBuff.
        private const int SWAP_BUFFER_NUM = 4; //	How many audio source to swap.
        private const double OVERLAP_TIME = 0.02; //  Our audio clip is defined as: [overlay][audio data][overlap].
        public bool playOnAwake = false;
        public string mediaPath; //	Assigned outside.
        public UnityEvent onInitComplete = null; //  Initialization is asynchronized. Invoked after initialization.
        public UnityEvent onVideoEnd = null; //  Invoked on video end.
        private NativeClass.DecoderState lastState = NativeClass.DecoderState.NOT_INITIALIZED;
        public NativeClass.DecoderState decoderState = NativeClass.DecoderState.NOT_INITIALIZED;
        private int decoderID = -1;
        private bool isAllAudioChEnabled;

        private bool useDefault = true; //  To set default texture before video initialized.
        private bool seekPreview; //  To preview first frame of seeking when seek under paused state.
        private Texture2D videoTexYch;
        private Texture2D videoTexUch;
        private Texture2D videoTexVch;
        private int videoWidth = -1;
        private int videoHeight = -1;
        private readonly AudioSource[] audioSource = new AudioSource[SWAP_BUFFER_NUM];
        private List<float> audioDataBuff; //  Buffer to keep audio data decoded from native.
        private int audioOverlapLength; //  OVERLAP_TIME * audioFrequency.
        private int audioDataLength; //  (AUDIO_FRAME_SIZE + 2 * audioOverlapLength) * audioChannel.
        private float volume = 1.0f;

        //	Time control
        private double globalStartTime; //  Video and audio progress are based on this start time.
        private bool isVideoReadyToReplay;
        private bool isAudioReadyToReplay;
        private double audioProgressTime = -1.0;
        private double hangTime = -1.0f; //  Used to set progress time after seek/resume.
        private double firstAudioFrameTime = -1.0;

        private BackgroundWorker backgroundWorker;
        private readonly object _lock = new object();

        public bool isVideoEnabled { get; private set; }
        public bool isAudioEnabled { get; private set; }
        public int audioFrequency { get; private set; }
        public int audioChannels { get; private set; }
        public float videoTotalTime { get; private set; } //  Video duration.
        public float audioTotalTime { get; private set; } //  Audio duration.

        public void getAllAudioChannelData(out float[] data, out double time, out int samplesPerChannel)
        {
            if (!isAllAudioChEnabled)
            {
                print(LOG_TAG + " this function only works for isAllAudioEnabled == true.");
                data = null;
                time = 0;
                samplesPerChannel = 0;
                return;
            }
            var dataPtr = new IntPtr();
            var lengthPerChannel = 0;
            double audioNativeTime = NativeClass.nativeGetAudioData(decoderID, ref dataPtr, ref lengthPerChannel);
            float[] buff = null;
            if (lengthPerChannel > 0)
            {
                buff = new float[lengthPerChannel * audioChannels];
                Marshal.Copy(dataPtr, buff, 0, buff.Length);
                NativeClass.nativeFreeAudioData(decoderID);
            }
            data = buff;
            time = audioNativeTime;
            samplesPerChannel = lengthPerChannel;
        }

        public NativeClass.DecoderState getDecoderState()
        {
            return decoderState;
        }

        public static void getMetaData(string filePath, out string[] key, out string[] value)
        {
            var keyptr = IntPtr.Zero;
            var valptr = IntPtr.Zero;
            var metaCount = NativeClass.nativeGetMetaData(filePath, out keyptr, out valptr);
            var keys = new IntPtr[metaCount];
            var vals = new IntPtr[metaCount];
            Marshal.Copy(keyptr, keys, 0, metaCount);
            Marshal.Copy(valptr, vals, 0, metaCount);
            var keyArray = new string[metaCount];
            var valArray = new string[metaCount];
            for (var i = 0; i < metaCount; i++)
            {
                keyArray[i] = Marshal.PtrToStringAnsi(keys[i]);
                valArray[i] = Marshal.PtrToStringAnsi(vals[i]);
                Marshal.FreeCoTaskMem(keys[i]);
                Marshal.FreeCoTaskMem(vals[i]);
            }
            Marshal.FreeCoTaskMem(keyptr);
            Marshal.FreeCoTaskMem(valptr);
            key = keyArray;
            value = valArray;
        }

        public float getVideoCurrentTime()
        {
            if (decoderState == NativeClass.DecoderState.PAUSE || decoderState == NativeClass.DecoderState.SEEK_FRAME)
            {
                return (float)hangTime;
            }
            return (float)(curRealTime - globalStartTime);
        }

        public void getVideoResolution(ref int width, ref int height)
        {
            width = videoWidth;
            height = videoHeight;
        }

        public float getVolume()
        {
            return volume;
        }

        public void initDecoder(string path, bool enableAllAudioCh = false)
        {
            isAllAudioChEnabled = enableAllAudioCh;
            StartCoroutine(initDecoderAsync(path));
        }

        public bool isSeeking()
        {
            return decoderState >= NativeClass.DecoderState.INITIALIZED && (decoderState == NativeClass.DecoderState.SEEK_FRAME || !NativeClass.nativeIsContentReady(decoderID));
        }

        public bool isVideoEOF()
        {
            return decoderState == NativeClass.DecoderState.EOF;
        }

        public static void loadVideoThumb(GameObject obj, string filePath, float time)
        {
            if (!File.Exists(filePath))
            {
                print(LOG_TAG + " File not found!");
                return;
            }
            var decID = -1;
            var width = 0;
            var height = 0;
            var totalTime = 0.0f;
            NativeClass.nativeCreateDecoder(filePath, ref decID);
            NativeClass.nativeGetVideoFormat(decID, ref width, ref height, ref totalTime);
            if (!NativeClass.nativeStartDecoding(decID))
            {
                print(LOG_TAG + " Decoding not start.");
                return;
            }
            var thumbY = new Texture2D(width, height, TextureFormat.Alpha8, false);
            var thumbU = new Texture2D(width / 2, height / 2, TextureFormat.Alpha8, false);
            var thumbV = new Texture2D(width / 2, height / 2, TextureFormat.Alpha8, false);
            var thumbMat = getMaterial(obj);
            if (thumbMat == null)
            {
                print(LOG_TAG + " Target has no MeshRenderer.");
                NativeClass.nativeDestroyDecoder(decID);
                return;
            }
            thumbMat.SetTexture("_YTex", thumbY);
            thumbMat.SetTexture("_UTex", thumbU);
            thumbMat.SetTexture("_VTex", thumbV);
            NativeClass.nativeLoadThumbnail(decID, time, thumbY.GetNativeTexturePtr(), thumbU.GetNativeTexturePtr(), thumbV.GetNativeTexturePtr());
            NativeClass.nativeDestroyDecoder(decID);
        }

        protected static Material getMaterial(GameObject obj)
        {
            var meshRenderer = obj.GetComponent<MeshRenderer>();
            if (meshRenderer)
                return meshRenderer.material;
            else
            {
                var image = obj.GetComponent<Image>();
                if (image)
                    return image.material;
                else
                {
                    var rawImage = obj.GetComponent<RawImage>();
                    if (rawImage)
                        return rawImage.material;
                    else
                    {
                        Debug.LogError("Please attach YUV2RGBA shader to material");
                        return new Material(Shader.Find("Unlit/YUV2RGBA"));
                    }
                }
            }
        }

        public void mute()
        {
            var temp = volume;
            setVolume(0.0f);
            volume = temp;
        }

        public void replay()
        {
            if (setSeekTime(0.0f))
            {
                globalStartTime = curRealTime;
                isVideoReadyToReplay = isAudioReadyToReplay = false;
            }
        }

        public void setAudioEnable(bool isEnable)
        {
            NativeClass.nativeSetAudioEnable(decoderID, isEnable);
            if (isEnable)
            {
                setSeekTime(getVideoCurrentTime());
            }
        }

        public void setPause()
        {
            if (decoderState == NativeClass.DecoderState.START)
            {
                hangTime = curRealTime - globalStartTime;
                decoderState = NativeClass.DecoderState.PAUSE;
                if (isAudioEnabled)
                {
                    foreach (var src in audioSource)
                    {
                        src.Pause();
                    }
                }
            }
        }

        public void setResume()
        {
            if (decoderState == NativeClass.DecoderState.PAUSE)
            {
                globalStartTime = curRealTime - hangTime;
                decoderState = NativeClass.DecoderState.START;
                if (isAudioEnabled)
                {
                    foreach (var src in audioSource)
                    {
                        src.UnPause();
                    }
                }
            }
        }

        public bool setSeekTime(float seekTime)
        {
            if (decoderState != NativeClass.DecoderState.SEEK_FRAME && decoderState >= NativeClass.DecoderState.START)
            {
                lastState = decoderState;
                decoderState = NativeClass.DecoderState.SEEK_FRAME;
                var setTime = 0.0f;
                if ((isVideoEnabled && seekTime > videoTotalTime) || (isAudioEnabled && seekTime > audioTotalTime) || isVideoReadyToReplay || isAudioReadyToReplay || seekTime < 0.0f)
                {
                    print(LOG_TAG + " Seek over end. ");
                    setTime = 0.0f;
                }
                else
                {
                    setTime = seekTime;
                }
                print(LOG_TAG + " set seek time: " + setTime);
                hangTime = setTime;
                NativeClass.nativeSetSeekTime(decoderID, setTime);
                NativeClass.nativeSetVideoTime(decoderID, setTime);
                if (isAudioEnabled)
                {
                    lock (_lock)
                    {
                        audioDataBuff.Clear();
                    }
                    audioProgressTime = firstAudioFrameTime = -1.0;
                    foreach (var src in audioSource)
                    {
                        src.Stop();
                    }
                }
                return true;
            }
            return false;
        }

        public void setStepBackward(float sec)
        {
            var targetTime = curRealTime - globalStartTime - sec;
            if (setSeekTime((float)targetTime))
            {
                print(LOG_TAG + " set backward : " + sec);
            }
        }

        public void setStepForward(float sec)
        {
            var targetTime = curRealTime - globalStartTime + sec;
            if (setSeekTime((float)targetTime))
            {
                print(LOG_TAG + " set forward : " + sec);
            }
        }

        public void setVideoEnable(bool isEnable)
        {
            NativeClass.nativeSetVideoEnable(decoderID, isEnable);
            if (isEnable)
            {
                setSeekTime(getVideoCurrentTime());
            }
        }

        public void setVolume(float vol)
        {
            volume = Mathf.Clamp(vol, 0.0f, 1.0f);
            foreach (var src in audioSource)
            {
                if (src != null)
                {
                    src.volume = volume;
                }
            }
        }

        public void startDecoding()
        {
            if (decoderState == NativeClass.DecoderState.INITIALIZED)
            {
                if (!NativeClass.nativeStartDecoding(decoderID))
                {
                    print(LOG_TAG + " Decoding not start.");
                    return;
                }
                decoderState = NativeClass.DecoderState.BUFFERING;
                globalStartTime = curRealTime;
                hangTime = curRealTime - globalStartTime;
                isVideoReadyToReplay = isAudioReadyToReplay = false;
                if (isAudioEnabled && !isAllAudioChEnabled)
                {
                    StartCoroutine("audioPlay");
                    backgroundWorker = new BackgroundWorker();
                    backgroundWorker.WorkerSupportsCancellation = true;
                    backgroundWorker.DoWork += pullAudioData;
                    backgroundWorker.RunWorkerAsync();
                }
            }
        }

        public void stopDecoding()
        {
            if (decoderState >= NativeClass.DecoderState.INITIALIZING)
            {
                print(LOG_TAG + " stop decoding.");
                decoderState = NativeClass.DecoderState.STOP;
                releaseTextures();
                if (isAudioEnabled)
                {
                    StopCoroutine("audioPlay");
                    backgroundWorker.CancelAsync();
                    if (audioSource != null)
                    {
                        for (var i = 0; i < SWAP_BUFFER_NUM; i++)
                        {
                            if (audioSource[i] != null)
                            {
                                Destroy(audioSource[i].clip);
                                Destroy(audioSource[i]);
                                audioSource[i] = null;
                            }
                        }
                    }
                }
                NativeClass.nativeDestroyDecoder(decoderID);
                decoderID = -1;
                decoderState = NativeClass.DecoderState.NOT_INITIALIZED;
                isVideoEnabled = isAudioEnabled = false;
                isVideoReadyToReplay = isAudioReadyToReplay = false;
                isAllAudioChEnabled = false;
            }
        }

        public void unmute()
        {
            setVolume(volume);
        }

        private double curRealTime
        {
            get { return DateTime.Now.TimeOfDay.TotalSeconds; }
        }
        private IEnumerator audioPlay()
        {
            print(LOG_TAG + " start audio play coroutine.");
            var swapIndex = 0; //	Swap between audio sources.
            var audioDataTime = (double)AUDIO_FRAME_SIZE / audioFrequency;
            var playedAudioDataLength = AUDIO_FRAME_SIZE * audioChannels; //  Data length exclude the overlap length.
            print(LOG_TAG + " audioDataTime " + audioDataTime);
            audioProgressTime = -1.0; //  Used to schedule each audio clip to be played.
            while (decoderState >= NativeClass.DecoderState.START)
            {
                if (decoderState == NativeClass.DecoderState.START)
                {
                    var currentTime = curRealTime - globalStartTime;
                    if (currentTime < audioTotalTime || audioTotalTime == -1.0f)
                    {
                        if (audioDataBuff != null && audioDataBuff.Count >= audioDataLength)
                        {
                            if (audioProgressTime == -1.0)
                            {
                                //  To simplify, the first overlap data would not be played.
                                //  Correct the audio progress time by adding OVERLAP_TIME.
                                audioProgressTime = firstAudioFrameTime + OVERLAP_TIME;
                                globalStartTime = curRealTime - audioProgressTime;
                            }
                            while (audioSource[swapIndex].isPlaying || decoderState == NativeClass.DecoderState.SEEK_FRAME)
                            {
                                yield return null;
                            }

                            //  Re-check data length if audioDataBuff is cleared by seek.
                            if (audioDataBuff.Count >= audioDataLength)
                            {
                                var playTime = audioProgressTime + globalStartTime;
                                var endTime = playTime + audioDataTime;

                                //  If audio is late, adjust start time and re-calculate audio clip time.
                                if (playTime <= curRealTime)
                                {
                                    globalStartTime = curRealTime - audioProgressTime;
                                    playTime = audioProgressTime + globalStartTime;
                                    endTime = playTime + audioDataTime;
                                }
                                audioSource[swapIndex].clip.SetData(audioDataBuff.GetRange(0, audioDataLength).ToArray(), 0);
                                audioSource[swapIndex].PlayScheduled(playTime);
                                audioSource[swapIndex].SetScheduledEndTime(endTime);
                                audioSource[swapIndex].time = (float)OVERLAP_TIME;
                                audioProgressTime += audioDataTime;
                                swapIndex = (swapIndex + 1) % SWAP_BUFFER_NUM;
                                lock (_lock)
                                {
                                    audioDataBuff.RemoveRange(0, playedAudioDataLength);
                                }
                            }
                        }
                    }
                    else
                    {
                        //print(LOG_TAG + " Audio reach EOF. Prepare replay.");
                        isAudioReadyToReplay = true;
                        audioProgressTime = firstAudioFrameTime = -1.0;
                        if (audioDataBuff != null)
                        {
                            lock (_lock)
                            {
                                audioDataBuff.Clear();
                            }
                        }
                    }
                }
                yield return new WaitForFixedUpdate();
            }
        }

        private void registNativeLog()
        {
            var logDel = new Action<string>(NativeClass.nativeLogHandler);
            var intptr_delegate = Marshal.GetFunctionPointerForDelegate(logDel);
            NativeClass.nativeRegistLogHandler(intptr_delegate);
        }
        private void Awake()
        {
            print(LOG_TAG);
            registNativeLog();
            if (playOnAwake)
            {
                print(LOG_TAG + " play on wake.");
                onInitComplete.AddListener(startDecoding);
                initDecoder(mediaPath);
            }
        }

        private void getAudioFormat()
        {
            var channels = 0;
            var freqency = 0;
            var duration = 0.0f;
            NativeClass.nativeGetAudioFormat(decoderID, ref channels, ref freqency, ref duration);
            audioChannels = channels;
            audioFrequency = freqency;
            audioTotalTime = duration > 0 ? duration : -1.0f;
            print(LOG_TAG + " audioChannel " + audioChannels);
            print(LOG_TAG + " audioFrequency " + audioFrequency);
            print(LOG_TAG + " audioTotalTime " + audioTotalTime);
        }

        //  Render event

        private void getTextureFromNative()
        {
            releaseTextures();
            var nativeTexturePtrY = new IntPtr();
            var nativeTexturePtrU = new IntPtr();
            var nativeTexturePtrV = new IntPtr();
            NativeClass.nativeCreateTexture(decoderID, ref nativeTexturePtrY, ref nativeTexturePtrU, ref nativeTexturePtrV);
            videoTexYch = Texture2D.CreateExternalTexture(videoWidth, videoHeight, TextureFormat.Alpha8, false, false, nativeTexturePtrY);
            videoTexUch = Texture2D.CreateExternalTexture(videoWidth / 2, videoHeight / 2, TextureFormat.Alpha8, false, false, nativeTexturePtrU);
            videoTexVch = Texture2D.CreateExternalTexture(videoWidth / 2, videoHeight / 2, TextureFormat.Alpha8, false, false, nativeTexturePtrV);
        }

        private void initAudioSource()
        {
            getAudioFormat();
            audioOverlapLength = (int)(OVERLAP_TIME * audioFrequency + 0.5f);
            audioDataLength = (AUDIO_FRAME_SIZE + 2 * audioOverlapLength) * audioChannels;
            for (var i = 0; i < SWAP_BUFFER_NUM; i++)
            {
                if (audioSource[i] == null)
                {
                    audioSource[i] = gameObject.AddComponent<AudioSource>();
                }
                audioSource[i].clip = AudioClip.Create("testSound" + i, audioDataLength, audioChannels, audioFrequency, false);
                audioSource[i].playOnAwake = false;
                audioSource[i].volume = volume;
                audioSource[i].minDistance = audioSource[i].maxDistance;
            }
        }

        private IEnumerator initDecoderAsync(string path)
        {
            print(LOG_TAG + " init Decoder.");
            decoderState = NativeClass.DecoderState.INITIALIZING;
            mediaPath = path;
            decoderID = -1;
            NativeClass.nativeCreateDecoderAsync(mediaPath, ref decoderID);
            var result = 0;
            do
            {
                yield return null;
                result = NativeClass.nativeGetDecoderState(decoderID);
            } while (!(result == 1 || result == -1));

            //  Init success.
            if (result == 1)
            {
                print(LOG_TAG + " Init success.");
                isVideoEnabled = NativeClass.nativeIsVideoEnabled(decoderID);
                if (isVideoEnabled)
                {
                    var duration = 0.0f;
                    NativeClass.nativeGetVideoFormat(decoderID, ref videoWidth, ref videoHeight, ref duration);
                    videoTotalTime = duration > 0 ? duration : -1.0f;
                    print(LOG_TAG + " Video format: (" + videoWidth + ", " + videoHeight + ")");
                    if (videoTotalTime > 0) print(LOG_TAG + " Total time: " + videoTotalTime);
                    setTextures(null, null, null);
                    useDefault = true;
                }

                //	Initialize audio.
                isAudioEnabled = NativeClass.nativeIsAudioEnabled(decoderID);
                print(LOG_TAG + " isAudioEnabled = " + isAudioEnabled);
                if (isAudioEnabled)
                {
                    if (isAllAudioChEnabled)
                    {
                        NativeClass.nativeSetAudioAllChDataEnable(decoderID, isAllAudioChEnabled);
                        getAudioFormat();
                    }
                    else
                    {
                        getAudioFormat();
                        initAudioSource();
                    }
                }
                decoderState = NativeClass.DecoderState.INITIALIZED;
                if (onInitComplete != null)
                {
                    onInitComplete.Invoke();
                }
            }
            else
            {
                print(LOG_TAG + " Init fail.");
                decoderState = NativeClass.DecoderState.INIT_FAIL;
            }
        }



        private void OnApplicationQuit()
        {
            //print(LOG_TAG + " OnApplicationQuit");
            stopDecoding();
        }

        private void OnDestroy()
        {
            //print(LOG_TAG + " OnDestroy");
            stopDecoding();
            NativeClass.nativeRegistLogHandler(IntPtr.Zero);
        }

        private void pullAudioData(object sender, DoWorkEventArgs e)
        {
            var dataPtr = IntPtr.Zero; //	Pointer to get audio data from native.
            var tempBuff = new float[0]; //	Buffer to copy audio data from dataPtr to audioDataBuff.
            var audioFrameLength = 0;
            double lastTime = -1.0f; //	Avoid to schedule the same audio data set.
            audioDataBuff = new List<float>();
            while (decoderState >= NativeClass.DecoderState.START)
            {
                if (decoderState != NativeClass.DecoderState.SEEK_FRAME)
                {
                    double audioNativeTime = NativeClass.nativeGetAudioData(decoderID, ref dataPtr, ref audioFrameLength);
                    if (0 < audioNativeTime && lastTime != audioNativeTime && decoderState != NativeClass.DecoderState.SEEK_FRAME && audioFrameLength != 0)
                    {
                        if (firstAudioFrameTime == -1.0)
                        {
                            firstAudioFrameTime = audioNativeTime;
                        }
                        lastTime = audioNativeTime;
                        audioFrameLength *= audioChannels;
                        if (tempBuff.Length != audioFrameLength)
                        {
                            //  For dynamic audio data length, reallocate the memory if needed.
                            tempBuff = new float[audioFrameLength];
                        }
                        Marshal.Copy(dataPtr, tempBuff, 0, audioFrameLength);
                        lock (_lock)
                        {
                            audioDataBuff.AddRange(tempBuff);
                        }
                    }
                    if (audioNativeTime != -1.0)
                    {
                        NativeClass.nativeFreeAudioData(decoderID);
                    }
                    Thread.Sleep(2);
                }
            }
            lock (_lock)
            {
                audioDataBuff.Clear();
                audioDataBuff = null;
            }
        }

        private void releaseTextures()
        {
            setTextures(null, null, null);
            videoTexYch = null;
            videoTexUch = null;
            videoTexVch = null;
            useDefault = true;
        }

        private void setTextures(Texture ytex, Texture utex, Texture vtex)
        {
            var texMaterial = getMaterial(gameObject);
            texMaterial.SetTexture("_YTex", ytex);
            texMaterial.SetTexture("_UTex", utex);
            texMaterial.SetTexture("_VTex", vtex);
        }

        //  Video progress is triggered using Update. Progress time would be set by nativeSetVideoTime.
        private void Update()
        {
            switch (decoderState)
            {
                case NativeClass.DecoderState.START:
                    if (isVideoEnabled)
                    {
                        //  Prevent empty texture generate green screen.(default 0,0,0 in YUV which is green in RGB)
                        if (useDefault && NativeClass.nativeIsContentReady(decoderID))
                        {
                            getTextureFromNative();
                            setTextures(videoTexYch, videoTexUch, videoTexVch);
                            useDefault = false;
                        }

                        //	Update video frame by dspTime.
                        var setTime = curRealTime - globalStartTime;

                        //	Normal update frame.
                        if (setTime < videoTotalTime || videoTotalTime <= 0)
                        {
                            if (seekPreview && NativeClass.nativeIsContentReady(decoderID))
                            {
                                setPause();
                                seekPreview = false;
                                unmute();
                            }
                            else
                            {
                                NativeClass.nativeSetVideoTime(decoderID, (float)setTime);
                                GL.IssuePluginEvent(NativeClass.GetRenderEventFunc(), decoderID);
                            }
                        }
                        else
                        {
                            isVideoReadyToReplay = true;
                        }
                    }
                    if (NativeClass.nativeIsVideoBufferEmpty(decoderID) && !NativeClass.nativeIsEOF(decoderID))
                    {
                        decoderState = NativeClass.DecoderState.BUFFERING;
                        hangTime = curRealTime - globalStartTime;
                    }
                    break;
                case NativeClass.DecoderState.SEEK_FRAME:
                    if (NativeClass.nativeIsSeekOver(decoderID))
                    {
                        globalStartTime = curRealTime - hangTime;
                        decoderState = NativeClass.DecoderState.START;
                        if (lastState == NativeClass.DecoderState.PAUSE)
                        {
                            seekPreview = true;
                            mute();
                        }
                    }
                    break;
                case NativeClass.DecoderState.BUFFERING:
                    if (!NativeClass.nativeIsVideoBufferEmpty(decoderID) || NativeClass.nativeIsEOF(decoderID))
                    {
                        decoderState = NativeClass.DecoderState.START;
                        globalStartTime = curRealTime - hangTime;
                    }
                    break;
                case NativeClass.DecoderState.PAUSE:
                case NativeClass.DecoderState.EOF:
                default:
                    break;
            }
            if (isVideoEnabled || isAudioEnabled)
            {
                if ((!isVideoEnabled || isVideoReadyToReplay) && (!isAudioEnabled || isAudioReadyToReplay))
                {
                    decoderState = NativeClass.DecoderState.EOF;
                    isVideoReadyToReplay = isAudioReadyToReplay = false;
                    if (onVideoEnd != null)
                    {
                        onVideoEnd.Invoke();
                    }
                }
            }
        }
    }
}