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
Shader "Hidden/Camera Transitions/Gate"
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
  fixed _GatePerspective;
  fixed _GateDepth;
  fixed _GateReflection;

  inline bool InBounds(float2 p)
  {
    return all(0.0 < p) && all(p < 1.0);
  }

  inline float3 BackgroundColorGamma(float2 p, float2 pto, sampler2D to)
  {
    float3 pixel = 0.0; // Black.

    pto *= float2(1.0, -1.2);
    pto += float2(0.0, -0.02);

    if (InBounds(pto))
      pixel += lerp(0.0, tex2D(to, RenderTextureUV(pto)).rgb, _GateReflection * lerp(1.0, 0.0, pto.y));

    return pixel;
  }

  inline float3 BackgroundColorLinear(float2 p, float2 pto, sampler2D to)
  {
    float3 pixel = 0.0; // Black.

    pto *= float2(1.0, -1.2);
    pto.y -= 0.02;

    if (InBounds(pto))
      pixel += lerp(0.0, sRGB(tex2D(to, RenderTextureUV(pto)).rgb), _GateReflection * lerp(1.0, 0.0, pto.y));

    return pixel;
  }

  float4 frag_gamma(v2f_img i) : COLOR
  {
    float2 pfr = -1.0;
    float2 pto = -1.0;

    float middleSlit = 2.0 * abs(i.uv.x - 0.5) - _T;
    if (middleSlit > 0.0)
    {
      pfr = i.uv + (i.uv.x > 0.5 ? -1.0 : 1.0) * float2(0.5 * _T, 0.0);
      float d = 1.0 / (1.0 + _GatePerspective * _T * (1.0 - middleSlit));
      pfr.y -= d / 2.0;
      pfr.y *= d;
      pfr.y += d / 2.0;
    }

    float size = lerp(1.0, _GateDepth, 1.0 - _T);
    pto = (i.uv - 0.5) * size + 0.5;

    if (InBounds(pfr))
      return tex2D(_MainTex, pfr);
    else if (InBounds(pto))
      return tex2D(_SecondTex, RenderTextureUV(pto));

    return float4(BackgroundColorGamma(i.uv, pto, _SecondTex), 1.0);
  }

  float4 frag_linear(v2f_img i) : COLOR
  {
    float2 pfr = -1.0;
    float2 pto = -1.0;

    float middleSlit = 2.0 * abs(i.uv.x - 0.5) - _T;
    if (middleSlit > 0.0)
    {
      pfr = i.uv + (i.uv.x > 0.5 ? -1.0 : 1.0) * float2(0.5 * _T, 0.0);
      float d = 1.0 / (1.0 + _GatePerspective * _T * (1.0 - middleSlit));
      pfr.y -= d / 2.0;
      pfr.y *= d;
      pfr.y += d / 2.0;
    }

    float size = lerp(1.0, _GateDepth, 1.0 - _T);
    pto = (i.uv - 0.5) * size + 0.5;

    if (InBounds(pfr))
      return tex2D(_MainTex, pfr);
    else if (InBounds(pto))
      return tex2D(_SecondTex, RenderTextureUV(pto));

    return float4(Linear(BackgroundColorLinear(i.uv, pto, _SecondTex)), 1.0);
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