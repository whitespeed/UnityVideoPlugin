///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Camera Transitions.
//
// Copyright (c) Ibuprogames <hello@ibuprogames.com>. All rights reserved.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// http://unity3d.com/support/documentation/Components/SL-Shader.html
Shader "Hidden/Camera Transitions/Doom"
{
  // http://unity3d.com/support/documentation/Components/SL-Properties.html
  Properties
  {
    _MainTex("Base (RGB)", 2D) = "white" {}

    _SecondTex("Second (RGB)", 2D) = "white" {}

	  // Transition.
    _T("Amount", Range(0.0, 1.0)) = 1.0
  }

  CGINCLUDE
  #include "UnityCG.cginc"
  #include "CameraTransitionsCG.cginc"

  sampler2D _MainTex;
  sampler2D _SecondTex;

  fixed _T;
  int _BarWidth;
  fixed _Amplitude;
  fixed _Noise;
  fixed _Frequency;

  inline float Wave(int num)
  {
    float fn = float(num) * _Frequency * 0.1 * float(_BarWidth);
    
    return cos(fn * 0.5) * cos(fn * 0.13) * sin((fn + 10.0) * 0.3) / 2.0 + 0.5;
  }

  inline float BarPosition(int num)
  {
    if (_Noise == 0.0)
      return Wave(num);
    
    return lerp(Wave(num), Rand01(num), _Noise);
  }

  float4 frag_gamma(v2f_img i) : COLOR
  {
    float2 uv = i.uv * _ScreenParams.xy;

    int bar = uv.x / (float)_BarWidth;
    float scale = 1.0 + BarPosition(bar) * _Amplitude;
    float phase = _T * scale;

    float2 p;
    float3 pixel;
    
    if (phase + i.uv.y < 1.0)
    {
      p = float2(uv.x, uv.y + lerp(0.0, _ScreenParams.y, phase)) / _ScreenParams.xy;
      pixel = tex2D(_MainTex, p).rgb;
    }
    else
      pixel = tex2D(_SecondTex, RenderTextureUV(i.uv)).rgb;

    return float4(pixel, 1.0);
  }

  float4 frag_linear(v2f_img i) : COLOR
  {
    float2 uv = i.uv * _ScreenParams.xy;

    int bar = uv.x / (float)_BarWidth;
    float scale = 1.0 + BarPosition(bar) * _Amplitude;
    float phase = _T * scale;

    float2 p;
    float3 pixel;
    
    if (phase + i.uv.y < 1.0)
    {
      p = float2(uv.x, uv.y + lerp(0.0, _ScreenParams.y, phase)) / _ScreenParams.xy;
      pixel = tex2D(_MainTex, p).rgb;
    }
    else
      pixel = tex2D(_SecondTex, RenderTextureUV(i.uv)).rgb;

    return float4(pixel, 1.0);
  }
  ENDCG

  // Techniques (http://unity3d.com/support/documentation/Components/SL-SubShader.html).
  SubShader
  {
    // Tags (http://docs.unity3d.com/Manual/SL-CullAndDepth.html).
    ZTest Always
    Cull Off
    ZWrite Off
    Fog { Mode off }

    // Pass 0: Color Space Gamma.
    Pass
    {
      CGPROGRAM
      #pragma fragmentoption ARB_precision_hint_fastest
      #pragma target 3.0
      #pragma multi_compile ___ INVERT_RENDERTEXTURE
      #pragma vertex vert_img
      #pragma fragment frag_gamma
      ENDCG
    }

    // Pass 1: Color Space Linear.
    Pass
    {
      CGPROGRAM
      #pragma fragmentoption ARB_precision_hint_fastest
      #pragma target 3.0
      #pragma multi_compile ___ INVERT_RENDERTEXTURE
      #pragma vertex vert_img
      #pragma fragment frag_linear
      ENDCG
    }
  }

  Fallback "Transition Fallback"
}