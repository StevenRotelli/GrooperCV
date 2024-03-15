using Emgu.CV;
using Emgu.CV.Structure;
using Grooper;
using System;
using System.Drawing;

namespace GrooperCV
{
  /// <summary>
  /// Extension Classes fro Grooper Types
  /// </summary>
  public static class GrooperImageExtensions
  {
    /// <summary>
    /// Returns a Mat from a GrooperImage
    /// </summary>
    /// <param name="image"></param>
    /// <returns></returns>
    public static Mat ToMat(this GrooperImage image)
    {
      Bitmap bmp = image.ToBmp();
      Image<Bgr, Byte> img = bmp.ToImage<Bgr, Byte>();
      return img.Mat;
    }
    /// <summary>
    /// Returns a GrooperImage from a Emgu.CV.Mat
    /// </summary>
    /// <param name="image"></param>
    /// <returns></returns>
    public static GrooperImage ToGrooperImage(this Mat image)
    {
      Bitmap bmp = image.ToBitmap();
      GrooperImage grooperImage = new GrooperImage(bmp);
      return grooperImage;
    }
  }
}
