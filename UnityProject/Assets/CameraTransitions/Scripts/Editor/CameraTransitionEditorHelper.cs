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
using UnityEngine;
using UnityEditor;

namespace CameraTransitions
{
  /// <summary>
  /// Utilities for the Editor.
  /// </summary>
  public static class CameraTransitionEditorHelper
  {
    /// <summary>
    /// Misc.
    /// </summary>
    public static readonly string DocumentationURL = @"http://www.ibuprogames.com/2015/11/10/camera-transitions/";

    /// <summary>
    /// A slider with a reset button.
    /// </summary>
    public static float SliderWithReset(string label, string tooltip, float value, float minValue, float maxValue, float defaultValue)
    {
      EditorGUILayout.BeginHorizontal();
      {
        value = EditorGUILayout.Slider(new GUIContent(label, tooltip), value, minValue, maxValue);

        if (GUILayout.Button(new GUIContent(@"R", string.Format("Reset to '{0}'", defaultValue)), GUILayout.Width(18.0f), GUILayout.Height(17.0f)) == true)
          value = defaultValue;
      }
      EditorGUILayout.EndHorizontal();

      return value;
    }

    /// <summary>
    /// A slider with a reset button.
    /// </summary>
    public static int IntSliderWithReset(string label, string tooltip, int value, int minValue, int maxValue, int defaultValue)
    {
      EditorGUILayout.BeginHorizontal();
      {
        value = EditorGUILayout.IntSlider(new GUIContent(label, tooltip), value, minValue, maxValue);

        if (GUILayout.Button(new GUIContent(@"R", string.Format("Reset to '{0}'", defaultValue)), GUILayout.Width(18.0f), GUILayout.Height(17.0f)) == true)
          value = defaultValue;
      }
      EditorGUILayout.EndHorizontal();

      return value;
    }

    /// <summary>
    /// Range with a reset button.
    /// </summary>
    public static void MinMaxSliderWithReset(string label, string tooltip, ref float minValue, ref float maxValue, float minLimit, float maxLimit, float defaultMinLimit, float defaultMaxLimit)
    {
      EditorGUILayout.BeginHorizontal();
      {
        EditorGUILayout.MinMaxSlider(new GUIContent(label, tooltip), ref minValue, ref maxValue, minLimit, maxLimit);

        if (GUILayout.Button(new GUIContent(@"R", string.Format("Reset to '{0}-{1}'", defaultMinLimit, defaultMaxLimit)), GUILayout.Width(18.0f), GUILayout.Height(17.0f)) == true)
        {
          minValue = defaultMinLimit;
          maxValue = defaultMaxLimit;
        }
      }
      EditorGUILayout.EndHorizontal();
    }

		/// <summary>
		/// Vector2 field with reset button.
		/// </summary>
		public static Vector2 Vector2WithReset(string label, string tooltip, Vector2 value, Vector2 defaultValue)
		{
			EditorGUILayout.BeginHorizontal();
			{
				EditorGUILayout.LabelField(new GUIContent(label, tooltip));
				
				float oldLabelWidth = EditorGUIUtility.labelWidth;
				
				EditorGUIUtility.labelWidth = 20.0f;
				
				value.x = EditorGUILayout.FloatField("X", value.x);
				
				value.y = EditorGUILayout.FloatField("Y", value.y);
				
				EditorGUIUtility.labelWidth = oldLabelWidth;
				
        if (GUILayout.Button(new GUIContent(@"R", string.Format("Reset to '{0}'", defaultValue)), GUILayout.Width(18.0f), GUILayout.Height(17.0f)) == true)
					value = defaultValue;
			}
			EditorGUILayout.EndHorizontal();
			
			return value;
		}

		/// <summary>
		/// Color field with reset button.
		/// </summary>
		public static Color ColorWithReset(string label, string tooltip, Color value, Color defaultValue)
		{
			EditorGUILayout.BeginHorizontal();
			{
				EditorGUILayout.LabelField(new GUIContent(label, tooltip));
				
				value = EditorGUILayout.ColorField(value);

				if (GUILayout.Button(new GUIContent(@"R", string.Format("Reset to '{0}'", defaultValue)), GUILayout.Width(18.0f), GUILayout.Height(17.0f)) == true)
					value = defaultValue;
			}
			EditorGUILayout.EndHorizontal();
			
			return value;
		}
  }
}