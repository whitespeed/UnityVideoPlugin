//========= Copyright 2015-2018, WhaleyVR. All rights reserved. ===========
//========= Written by whitespeed =========

using UnityEngine;

namespace UnityPlugin.Multimedia
{
	[System.Serializable]
	public class StereoProperty {
		public bool isLeftFirst;
		public StereoHandler.StereoType stereoType = StereoHandler.StereoType.TOP_DOWN;

		public GameObject left;
		public GameObject right;
	}
}