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
Shader "Hidden/Camera Transitions/Swap"
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
  fixed _SwapPerspective;
  fixed _SwapDepth;
  fixed _SwapReflection;

  inline bool InBounds(float2 p)
  {
    return all(0.0 < p) && all(p < 1.0);
  }

  inline float3 BackgroundColorGamma(float2 p, float2 pfr, float2 pto, sampler2D from, sampler2D to)
  {
    float3 pixel = 0.0; // Black.

    pfr *= float2(1.0, -1.0);

    if (InBounds(pfr))
      pixel += lerp(0.0, tex2D(from, pfr), _SwapReflection * lerp(1.0, 0.0, pfr.y));

    pto *= float2(1.0, -1.0);

    if (InBounds(pto))
      pixel += lerp(0.0, tex2D(to, pto), _SwapReflection * lerp(1.0, 0.0, pto.y));

    return pixel;
  }

  inline float3 BackgroundColorLinear(float2 p, float2 pfr, float2 pto, sampler2D from, sampler2D to)
  {
    float3 pixel = 0.0; // Black.

    pfr *= float2(1.0, -1.0);

    if (InBounds(pfr))
      pixel += lerp(0.0, sRGB(tex2D(from, pfr).rgb), _SwapReflection * lerp(1.0, 0.0, pfr.y));

    pto *= float2(1.0, -1.0);

    if (InBounds(pto))
      pixel += lerp(0.0, sRGB(tex2D(to, pto).rgb), _SwapReflection * lerp(1.0, 0.0, pto.y));

    return pixel;
  }

  float4 frag_gamma(v2f_img i) : COLOR
  {
    float2 pfr = -1.0;
    float2 pto = -1.0;
    float size = lerp(1.0, _SwapDepth, _T);
    float persp = _SwapPerspective * _T;

    pfr = (i.uv + float2(0.0, -0.5)) * float2(size / (1.0 - _SwapPerspective * _T), size / (1.0 - size * persp * i.uv.x)) + float2(0.0, 0.5);
    
    size = lerp(1.0, _SwapDepth, 1.0 - _T);
    persp = _SwapPerspective * (1.0 - _T);

    pto = (i.uv + float2(-1.0, -0.5)) * float2(size / (1.0 - _SwapPerspective * (1.0 - _T)), size / (1.0 - size * persp * (0.5 - i.uv.x))) + float2(1.0, 0.5);

    if (_T < 0.5)
    {
      if (InBounds(pfr))
        return tex2D(_MainTex, pfr);
      else if (InBounds(pto))
        return tex2D(_SecondTex, RenderTextureUV(pto));
      else
        return float4(BackgroundColorGamma(i.uv, pfr, pto, _MainTex, _SecondTex), 1.0);
    }

    if (InBounds(pto))
      return tex2D(_SecondTex, RenderTextureUV(pto));
    else if (InBounds(pfr))
      return tex2D(_MainTex, pfr);

    return float4(BackgroundColorGamma(i.uv, pfr, pto, _MainTex, _SecondTex), 1.0);
  }

  float4 frag_linear(v2f_img i) : COLOR
  {
    float2 pfr = -1.0;
    float2 pto = -1.0;
    float size = lerp(1.0, _SwapDepth, _T);
    float persp = _SwapPerspective * _T;

    pfr = (i.uv + float2(0.0, -0.5)) * float2(size / (1.0 - _SwapPerspective * _T), size / (1.0 - size * persp * i.uv.x)) + float2(0.0, 0.5);

    size = lerp(1.0, _SwapDepth, 1.0 - _T);
    persp = _SwapPerspective * (1.0 - _T);

    pto = (i.uv + float2(-1.0, -0.5)) * float2(size / (1.0 - _SwapPerspective * (1.0 - _T)), size / (1.0 - size * persp * (0.5 - i.uv.x))) + float2(1.0, 0.5);

    if (_T < 0.5)
    {
      if (InBounds(pfr))
        return tex2D(_MainTex, pfr);
      else if (InBounds(pto))
        return tex2D(_SecondTex, RenderTextureUV(pto));
      else
        return float4(Linear(BackgroundColorLinear(i.uv, pfr, pto, _MainTex, _SecondTex)), 1.0);
    }

    if (InBounds(pto))
      return tex2D(_SecondTex, RenderTextureUV(pto));
    else if (InBounds(pfr))
      return tex2D(_MainTex, pfr);

    return float4(Linear(BackgroundColorLinear(i.uv, pfr, pto, _MainTex, _SecondTex)), 1.0);
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