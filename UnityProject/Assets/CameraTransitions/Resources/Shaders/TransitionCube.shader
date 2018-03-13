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
Shader "Hidden/Camera Transitions/Cube"
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
  fixed _CubePerspective;
  fixed _CubeZoom;
  fixed _CubeReflection;
  fixed _CubeElevantion;

  inline float2 Project(float2 p)
  {
    return p * float2(1.0, -1.2) + float2(0.0, _CubeElevantion);
  }

  inline bool InBounds(float2 p)
  {
    return all(0.0 < p) && all(p < 1.0);
  }

  inline float3 BackgroundColorGamma(float2 p, float2 pfr, float2 pto, sampler2D from, sampler2D to)
  {
    float3 pixel = 0.0; // Black.

    pfr = Project(pfr);
  
    if (InBounds(pfr))
      pixel += lerp(0.0, tex2D(from, pfr), _CubeReflection * lerp(1.0, 0.0, pfr.y));
  
    pto = Project(pto);
  
    if (InBounds(pto))
      pixel += lerp(0.0, tex2D(to, pto), _CubeReflection * lerp(1.0, 0.0, pto.y));
    
    return pixel;
  }

  inline float3 BackgroundColorLinear(float2 p, float2 pfr, float2 pto, sampler2D from, sampler2D to)
  {
    float3 pixel = 0.0; // Black.

    pfr = Project(pfr);

    if (InBounds(pfr))
      pixel += lerp(0.0, sRGB(tex2D(from, pfr).rgb), _CubeReflection * lerp(1.0, 0.0, pfr.y));

    pto = Project(pto);

    if (InBounds(pto))
      pixel += lerp(0.0, sRGB(tex2D(to, RenderTextureUV(pto)).rgb), _CubeReflection * lerp(1.0, 0.0, pto.y));

    return pixel;
  }

  inline float2 XSkew(float2 p, float persp, float center)
  {
    float x = lerp(p.x, 1.0 - p.x, center);
  
    return ((float2(x, (p.y - 0.5 * (1.0 - persp) * x) / (1.0 + (persp - 1.0) * x)) - float2(0.5 - distance(center, 0.5), 0.0)) *
             float2(0.5 / distance(center, 0.5) * (center < 0.5 ? 1.0 : -1.0), 1.0) + float2(center < 0.5 ? 0.0 : 1.0, 0.0));
  }

  float4 frag_gamma(v2f_img i) : COLOR
  {
    float uz = _CubeZoom * 2.0 * (0.5 - distance(0.5, _T));
    float2 p = -uz * 0.5 + (1.0 + uz) * i.uv;
    
    float2 fromP = XSkew((p - float2(_T, 0.0)) / float2(1.0 - _T, 1.0), 1.0 - lerp(_T, 0.0, _CubePerspective), 0.0);
    float2 toP = XSkew(p / float2(_T, 1.0), lerp(pow(_T, 2.0), 1.0, _CubePerspective), 1.0);

    if (InBounds(fromP))
      return tex2D(_MainTex, fromP);
    else if (InBounds(toP))
      return tex2D(_SecondTex, RenderTextureUV(toP));
  
    return float4(BackgroundColorGamma(i.uv, fromP, toP, _MainTex, _SecondTex), 1.0);
  }

  float4 frag_linear(v2f_img i) : COLOR
  {
    float uz = _CubeZoom * 2.0 * (0.5 - distance(0.5, _T));
    float2 p = -uz * 0.5 + (1.0 + uz) * i.uv;

    float2 fromP = XSkew((p - float2(_T, 0.0)) / float2(1.0 - _T, 1.0), 1.0 - lerp(_T, 0.0, _CubePerspective), 0.0);
    float2 toP = XSkew(p / float2(_T, 1.0), lerp(pow(_T, 2.0), 1.0, _CubePerspective), 1.0);

    if (InBounds(fromP))
      return tex2D(_MainTex, fromP);
    else if (InBounds(toP))
      return tex2D(_SecondTex, RenderTextureUV(toP));

    return float4(Linear(BackgroundColorLinear(i.uv, fromP, toP, _MainTex, _SecondTex)), 1.0);
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