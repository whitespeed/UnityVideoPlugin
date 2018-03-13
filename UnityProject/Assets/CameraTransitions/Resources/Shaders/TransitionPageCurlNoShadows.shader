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
Shader "Hidden/Camera Transitions/Page Curl NoShadows"
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
	fixed _Radius;
	fixed2 _Angle;
  int _Obtuse;

	float2 PageCurl(float t, float maxt, float cyl)
	{
		float2 ret = float2(t, 1.0);

		if (t < cyl - _Radius)
			return ret;

		if (t > cyl + _Radius)
      return float2(-1.0, -1.0);

		float a = asin((t - cyl) / _Radius);
		float ca = -a + _PI;

		ret.x = cyl + ca * _Radius;
		ret.y = cos(ca);

		if (ret.x < maxt)
			return ret;

		if (t < cyl)
			return float2(t, 1.0);

		ret.x = cyl + a * _Radius;
		ret.y = cos(a);

		return (ret.x < maxt) ? ret : float2(-1.0, -1.0);
	}

	float4 frag_gamma(v2f_img i) : COLOR
	{
    float2 uv = (_Obtuse == 0) ? i.uv : float2(1.0 - i.uv.x, i.uv.y);

    float2 angle = _Angle * _T;
    float d = length(angle * (1.0 + 4.0 * _Radius)) - 2.0 * _Radius;
    float3 cyl = float3(normalize(angle), d);

    d = dot(uv, cyl.xy);
    float2 end = abs((1.0 - uv) / cyl.xy);
    float maxt = d + min(end.x, end.y);
    float2 cf = PageCurl(d, maxt, cyl.z);
    float2 tuv = i.uv + cyl.xy * (cf.x - d);

		float3 from = tex2D(_SecondTex, RenderTextureUV(tuv)).rgb;
		from = cf.y > 0.0 ? from : (from * 0.25 + 0.75);

		return float4(cf.x > 0.0 ? from : tex2D(_MainTex, i.uv).rgb, 1.0);
	}

  float4 frag_linear(v2f_img i) : COLOR
	{
    float2 uv = (_Obtuse == 0) ? i.uv : float2(1.0 - i.uv.x, i.uv.y);

    float2 angle = _Angle * _T;
    float d = length(angle * (1.0 + 4.0 * _Radius)) - 2.0 * _Radius;
    float3 cyl = float3(normalize(angle), d);

    d = dot(uv, cyl.xy);
    float2 end = abs((1.0 - uv) / cyl.xy);
    float maxt = d + min(end.x, end.y);
    float2 cf = PageCurl(d, maxt, cyl.z);
    float2 tuv = i.uv + cyl.xy * (cf.x - d);

		float3 from = sRGB(tex2D(_SecondTex, RenderTextureUV(tuv)).rgb);
    from = cf.y > 0.0 ? from : (from * 0.25 + 0.75);

		return float4(Linear(cf.x > 0.0 ? from : sRGB(tex2D(_MainTex, i.uv).rgb)), 1.0);
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