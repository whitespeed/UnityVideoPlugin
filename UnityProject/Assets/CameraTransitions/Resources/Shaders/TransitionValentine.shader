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
Shader "Hidden/Camera Transitions/Valentine"
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
  fixed _ValentineBorder;
  fixed3 _ValentineColor;

  inline bool Heart(float2 p, float2 center, float size)
  {
    if (size == 0.0)
      return false;

    float2 o = (p - center) / (1.6 * size);

    return pow(o.x * o.x + o.y * o.y - 0.3, 3.0) < o.x * o.x * pow(o.y, 3.0);
  }

  float4 frag_gamma(v2f_img i) : COLOR
  {
    fixed3 from = tex2D(_MainTex, i.uv).rgb;
    fixed3 to = tex2D(_SecondTex, RenderTextureUV(i.uv)).rgb;

    float h1 = Heart(i.uv, float2(0.5, 0.4), _T) ? 1.0 : 0.0;
    float h2 = Heart(i.uv, float2(0.5, 0.4), _T + 0.001 * _ValentineBorder) ? 1.0 : 0.0;

    float border = max(h2 - h1, 0.0);

    return float4(lerp(from, to, h1) * (1.0 - border) + _ValentineColor * border, 1.0);
  }

  float4 frag_linear(v2f_img i) : COLOR
  {
    fixed3 from = sRGB(tex2D(_MainTex, i.uv).rgb);
    fixed3 to = sRGB(tex2D(_SecondTex, RenderTextureUV(i.uv)).rgb);

    float h1 = Heart(i.uv, float2(0.5, 0.4), _T) ? 1.0 : 0.0;
    float h2 = Heart(i.uv, float2(0.5, 0.4), _T + 0.001 * _ValentineBorder) ? 1.0 : 0.0;

    float border = max(h2 - h1, 0.0);

    return float4(Linear(lerp(from, to, h1) * (1.0 - border) + _ValentineColor * border), 1.0);
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