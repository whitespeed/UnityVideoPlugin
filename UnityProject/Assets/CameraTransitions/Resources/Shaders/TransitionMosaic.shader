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
Shader "Hidden/Camera Transitions/Mosaic"
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
  int _StepX;
  int _StepY;
  int _Rotate;

  inline float2 Rotate(float2 v, float a)
  {
    return mul(v, float2x2(cos(a), -sin(a), sin(a), cos(a)));
  }

  float4 frag_gamma(v2f_img i) : COLOR
  {
    float2 uv = i.uv - 0.5;
    float st = _T * 2.0 - 1.0;

    uv *= abs(-(st * st * 2.0) + 3.0);
    uv += lerp(float2(0.5, 0.5), float2(float(_StepX) + 0.5, float(_StepY) + 0.5), _T * _T);
    
    float2 mrp = Mod(uv, 1.0);
    bool onEnd = int(floor(uv.x)) == _StepX && int(floor(uv.y)) == _StepY;

    if (onEnd == false && _Rotate == 1)
    {
      float ang = float(int(Rand01(floor(uv)) * 4.0)) * 0.5 * _PI;
      mrp = float2(0.5, 0.5) + Rotate(mrp - float2(0.5, 0.5), ang);
    }

    float3 pixel = 0.0;
    if (onEnd == true || Rand01(floor(uv)) > 0.5)
      pixel = tex2D(_SecondTex, mrp).rgb;
    else
      pixel = tex2D(_MainTex, mrp).rgb;

    return float4(pixel, 1.0);
  }

  float4 frag_linear(v2f_img i) : COLOR
  {
    float2 uv = i.uv - 0.5;
    float st = _T * 2.0 - 1.0;

    uv *= abs(-(st * st * 2.0) + 3.0);
    uv += lerp(float2(0.5, 0.5), float2(float(_StepX) + 0.5, float(_StepY) + 0.5), _T * _T);

    float2 mrp = Mod(uv, 1.0);
    bool onEnd = int(floor(uv.x)) == _StepX && int(floor(uv.y)) == _StepY;

    if (onEnd == false && _Rotate == 1)
    {
      float ang = float(int(Rand01(floor(uv)) * 4.0)) * 0.5 * _PI;
      mrp = float2(0.5, 0.5) + Rotate(mrp - float2(0.5, 0.5), ang);
    }

    float3 pixel = 0.0;
    if (onEnd == true || Rand01(floor(uv)) > 0.5)
      pixel = tex2D(_SecondTex, mrp).rgb;
    else
      pixel = tex2D(_MainTex, mrp).rgb;

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
    Fog{ Mode off }

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