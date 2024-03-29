﻿//========= Copyright 2015-2018, WhaleyVR. All rights reserved. ===========
//========= Written by whitespeed =========

using UnityEngine;
using UnityEngine.Events;
using System.Collections;

namespace UnityPlugin.Multimedia
{
	[RequireComponent(typeof(MediaDecoder))]
	public class VideoSourceController : MonoBehaviour {
		protected string LOG_TAG = "[VideoSourceController]";

		public string folderPath;
		public string filter;
		public bool isAdaptToResolution;
		public UnityEvent onInitComplete;
		public UnityEvent onChangeVideo;
		protected bool isInitialized = false;
		protected FileSeeker fileSeeker;

		protected MediaDecoder decoder;
		protected Vector3 oriScale;

		protected virtual void Start () {
			decoder = GetComponent<MediaDecoder>();
			initFileSeeker();
		}

		public void initFileSeeker() {
			if (folderPath == null) {
				Debug.Log(LOG_TAG + "Folder path is null.");
				return;
			}

			isInitialized = false;

			fileSeeker = new FileSeeker();
			if (!fileSeeker.loadFolder(folderPath, filter)) {
				Debug.Log(LOG_TAG + " content not found.");
				fileSeeker = null;
				return;
			}

			oriScale = transform.localScale;

			isInitialized = true;

			onInitComplete.Invoke();
		}

		public void startVideoPlay() {
			if (!isInitialized) {
				Debug.Log(LOG_TAG + " not initialized.");
				return;
			}

			if (isAdaptToResolution) {
				decoder.onInitComplete.AddListener(adaptResolution);
			}
			decoder.onInitComplete.AddListener(decoder.startDecoding);
			decoder.onInitComplete.AddListener(decoder.onInitComplete.RemoveAllListeners);
			decoder.initDecoder(fileSeeker.getPath());
		}

		public void nextVideo() {
			if (!isInitialized) {
				Debug.Log(LOG_TAG + " not initialized.");
				return;
			}

			decoder.stopDecoding();
			fileSeeker.toNext();

			onChangeVideo.Invoke();
		}

		public void prevVideo() {
			if (!isInitialized) {
				Debug.Log(LOG_TAG + " not initialized.");
				return;
			}

			decoder.stopDecoding();
			fileSeeker.toPrev();

			onChangeVideo.Invoke();
		}

		public void stopVideo() {
			if (!isInitialized) {
				Debug.Log(LOG_TAG + " not initialized.");
				return;
			}

			decoder.stopDecoding();
		}

		protected virtual void adaptResolution() {
			int width = 1;
			int height = 1;
			decoder.getVideoResolution(ref width, ref height);
			Vector3 adaptReso = oriScale;
			adaptReso.x *= ((float) width / height);
			transform.localScale = adaptReso;
		}
	}
}