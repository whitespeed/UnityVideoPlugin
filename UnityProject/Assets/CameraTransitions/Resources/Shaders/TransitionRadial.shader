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
Shader "Hidden/Camera Transitions/Radial"
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
  fixed _Clockwise;

  float4 frag_gamma(v2f_img i) : COLOR
  {
    fixed2 ruv = i.uv * 2.0 - 1.0;
  	fixed a = atan2(ruv.x, ruv.y) * _Clockwise;
  	fixed pa = _T * _PI * 2.5 - _PI * 1.25;

	  fixed3 to = tex2D(_SecondTex, RenderTextureUV(i.uv)).rgb;

    return float4(a > pa ? lerp(to, tex2D(_MainTex, i.uv).rgb, smoothstep(0.0, 1.0, (a - pa))) : to, 1.0);
  }

  float4 frag_linear(v2f_img i) : COLOR
  {
    fixed2 ruv = i.uv * 2.0 - 1.0;
  	fixed a = atan2(ruv.x, ruv.y) * _Clockwise;
  	fixed pa = _T * _PI * 2.5 - _PI * 1.25;

  	fixed3 to = sRGB(tex2D(_SecondTex, RenderTextureUV(i.uv)).rgb);

    return float4(a > pa ? Linear(lerp(to, sRGB(tex2D(_MainTex, i.uv).rgb), smoothstep(0.0, 1.0, (a - pa)))) : Linear(to), 1.0);
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