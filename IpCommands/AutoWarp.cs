using System;
using System.Drawing;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Runtime.Serialization;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Grooper;
using Grooper.IP;

namespace GrooperCV.IpCommands
{
  /// <summary>Finds the page within captured image by automatically locating the rectangular shape</summary>
  [DataContract, IconResource("AutoWarp"), Category("OpenCV")]
  public class AutoWarp : IpCommand
  {
    #region initialization & properties
    /// <inheritdoc/>
    public AutoWarp() { }
    ///<inheritdoc/>
    public AutoWarp(ConnectedObject owner) : base(owner) { }
    /// <summary> The image quality to be used for the image processing and detection alogrithms. This does not affect the image quality of the final output </summary>
    /// <remarks>Images captured from mobile devices often poses high resolution causing slowness. </remarks>
    [DataMember, Viewable, DisplayName("Image Quality"), DV(20), ValueRange(LowValue: 20, HighValue: 100)]
    public int Quality { get; set; }
    #endregion
    /// <inheritdoc/>
    protected override IpCommandResult ApplyCommand(GrooperImage image)
    {
      IpCommandResult result = new IpCommandResult(this, image);
      Mat input, morph, grabCut, grayScale, contours;
      input = DownscaleImage(image, result);
      morph = ApplyMorphologicalClose(input, result);
      grabCut = ApplyGrabCut(morph, result);
      grayScale = ConverToGray(grabCut, result);
      contours = CountourDetection(grayScale, result);
      VectorOfPoint approxContours = AproximateContours(contours);
      PointF[] points = DetectCornerPoints(approxContours);
      _ = DrawPoints(contours, result, OrderPoints(approxContours.ToArray()), approxContours);
      Mat transform = ApplyPerspectiveTransform(image.ToMat(), points);
      result.Image = transform.ToGrooperImage();

      return result;
    }

    /// <inheritdoc/>
    public Mat DownscaleImage(GrooperImage image, IpCommandResult result)
    {
      Mat inputImage = image.ToMat();
      if (Quality == 100) { return inputImage; }
      double scale = Quality / 100.0;
      Size newSize = new Size((int)(inputImage.Width * scale), (int)(inputImage.Height * scale));
      Mat downscaledImage = new Mat();
      CvInvoke.Resize(inputImage, downscaledImage, newSize, 0, 0, Inter.Linear);
      if (DiagMode) { result.AddDiagImage("Downscale", downscaledImage.ToGrooperImage()); }
      return downscaledImage;
    }

    /// <inheritdoc/>
    public Mat ApplyMorphologicalClose(Mat inputImage, IpCommandResult result)
    {
      Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(5, 5), new Point(-1, -1));
      Mat morphedImage = new Mat();
      CvInvoke.MorphologyEx(inputImage, morphedImage, MorphOp.Close, kernel, new Point(-1, -1), 3, BorderType.Reflect, new MCvScalar());
      if (DiagMode) { result.AddDiagImage("Morph", morphedImage.ToGrooperImage()); }
      return morphedImage;
    }

    /// <inheritdoc/>
    public Mat ApplyGrabCut(Mat img, IpCommandResult result)
    {
      // Initialize the mask to probable background
      Mat mask = new Mat(img.Size, DepthType.Cv8U, 1);
      mask.SetTo(new MCvScalar(2)); // Probable background

      Mat bgdModel = new Mat();
      Mat fgdModel = new Mat();
      // Define the rectangle for the foreground region (excluding 20-pixel margin)
      Rectangle rect = new Rectangle(20, 20, img.Width - 40, img.Height - 40);
      // Apply GrabCut
      CvInvoke.GrabCut(img, mask, rect, bgdModel, fgdModel, 5, GrabcutInitType.InitWithRect);
      // Process the mask, set background and probable background to 0, foreground and probable foreground to 1
      Mat foregroundMask = new Mat(mask.Size, DepthType.Cv8U, 1);
      mask.CopyTo(foregroundMask);
      _ = CvInvoke.Threshold(mask, foregroundMask, 2, 1, ThresholdType.Binary);
      // Use the mask to extract the foreground
      Mat resultImage = new Mat();
      img.CopyTo(resultImage, foregroundMask);
      if (DiagMode) { result.AddDiagImage("GrabCut", resultImage.ToGrooperImage()); }
      return resultImage;
    }

    /// <inheritdoc/>
    public Mat CountourDetection(Mat inputImage, IpCommandResult result)
    {
      // Edge Detection
      Mat canny = new Mat();
      CvInvoke.Canny(inputImage, canny, 0, 200);

      // Dilate to enhance edges
      Mat kernel = CvInvoke.GetStructuringElement(ElementShape.Ellipse, new Size(5, 5), new Point(-1, -1));
      CvInvoke.Dilate(canny, canny, kernel, new Point(-1, -1), 1, BorderType.Default, new MCvScalar());

      // Find Contours
      VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
      Mat hierarchy = new Mat();
      CvInvoke.FindContours(canny, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxNone);

      // Check if any contours were found before proceeding
      if (contours.Size == 0)
      {
        throw new Exception("nope"); // Consider handling this more gracefully
      }

      // Sort contours by area in descending order and keep only the largest
      int largestContourIndex = 0;
      double largestArea = 0.0;
      for (int i = 0; i < contours.Size; i++)
      {
        double area = CvInvoke.ContourArea(contours[i]);
        if (area > largestArea)
        {
          largestArea = area;
          largestContourIndex = i;
        }
      }

      // Draw the largest contour on a blank canvas
      Mat contourImage = new Mat(inputImage.Size, DepthType.Cv8U, 3);
      contourImage.SetTo(new MCvScalar(0, 0, 0)); // Set background to black
      CvInvoke.DrawContours(contourImage, contours, largestContourIndex, new MCvScalar(0, 255, 255), 3);

      if (DiagMode) { result.AddDiagImage("Contours", contourImage.ToGrooperImage()); }

      return contourImage;
    }

    /// <inheritdoc/>
    public Mat ApplyPerspectiveTransform(Mat image, PointF[] srcCorners)
    {
      AdjustPoints(srcCorners, Quality);
      PointF[] dstCorners = FindDestinationCoordinates(srcCorners);
      // Calculate the perspective transform matrix
      Mat perspectiveTransform = CvInvoke.GetPerspectiveTransform(srcCorners, dstCorners);
      // Apply the perspective transformation
      Mat warpedImage = new Mat();
      CvInvoke.WarpPerspective(image, warpedImage, perspectiveTransform, new Size((int)dstCorners[2].X, (int)dstCorners[2].Y), Inter.Linear, Emgu.CV.CvEnum.Warp.Default, BorderType.Reflect101);
      return warpedImage;
    }

    /// <inheritdoc/>
    public static PointF[] FindDestinationCoordinates(PointF[] pts)
    {
      // Unpack the points
      (PointF tl, PointF tr, PointF br, PointF bl) = (pts[0], pts[1], pts[2], pts[3]);

      // Finding the maximum width.
      float widthA = (float)Math.Sqrt(Math.Pow(br.X - bl.X, 2) + Math.Pow(br.Y - bl.Y, 2));
      float widthB = (float)Math.Sqrt(Math.Pow(tr.X - tl.X, 2) + Math.Pow(tr.Y - tl.Y, 2));
      int maxWidth = (int)Math.Max(widthA, widthB);

      // Finding the maximum height.
      float heightA = (float)Math.Sqrt(Math.Pow(tr.X - br.X, 2) + Math.Pow(tr.Y - br.Y, 2));
      float heightB = (float)Math.Sqrt(Math.Pow(tl.X - bl.X, 2) + Math.Pow(tl.Y - bl.Y, 2));
      int maxHeight = (int)Math.Max(heightA, heightB);

      // Final destination coordinates.
      PointF[] destinationCorners = new PointF[]
      {
        new PointF(0, 0),
        new PointF(maxWidth - 1, 0),
        new PointF(maxWidth - 1, maxHeight - 1),
        new PointF(0, maxHeight - 1)
      };

      return destinationCorners;
    }

    private static void AdjustPoints(PointF[] points, int Scale)
    {
      for (int i = 0; i < points.Length; i++)
      {
        PointF point = points[i];
        point.X /= Scale / 100.0f;
        point.Y /= Scale / 100.0f;
        points[i] = point;
      }
    }

    /// <inheritdoc/>
    public VectorOfPoint AproximateContours(Mat inputImage)
    {
      Mat gray = inputImage;
      CvInvoke.CvtColor(inputImage, gray, ColorConversion.Bgr2Gray);
      Mat binaryImage = new Mat();
      _ = CvInvoke.Threshold(inputImage, binaryImage, 1, 255, ThresholdType.Binary);
      VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
      Mat hierarchy = new Mat();
      CvInvoke.FindContours(binaryImage, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxSimple);
      VectorOfPoint approxContour = null;

      for (int i = 0; i < contours.Size; i++)
      {
        VectorOfPoint contour = contours[i];
        double epsilon = 0.02 * CvInvoke.ArcLength(contour, true);
        VectorOfPoint tmpApproxContour = new VectorOfPoint();
        CvInvoke.ApproxPolyDP(contour, tmpApproxContour, epsilon, true);

        // Check if the approximated contour has four points
        if (tmpApproxContour.Size == 4)
        {
          approxContour = tmpApproxContour;
          break; // Stop at the first detected quadrilateral
        }
      }

      return approxContour;
    }

    /// <inheritdoc/>
    public PointF[] DetectCornerPoints(VectorOfPoint approxContour)
    {
      List<PointF> cornerPoints = new List<PointF>();
      Point[] points = approxContour.ToArray();

      if (approxContour != null)
      {
        points = OrderPoints(points);
        foreach (Point p in points)
        {
          cornerPoints.Add(p);
        }
      }
      return cornerPoints.ToArray();
    }

    /// <inheritdoc/>
    public Mat DrawPoints(Mat inputImage, IpCommandResult result, Point[] points, VectorOfPoint approxContour)
    {
      Mat con = new Mat(inputImage.Size, DepthType.Cv8U, 3);
      con.SetTo(new MCvScalar(0, 0, 0)); // Making canvas black

      if (approxContour != null)
      {
        CvInvoke.DrawContours(con, new VectorOfVectorOfPoint(approxContour), -1, new MCvScalar(0, 255, 255), 3); // Draw the quadrilateral
        foreach (Point p in points)
        {
          CvInvoke.Circle(con, p, 10, new MCvScalar(0, 255, 0), -1); // Draw corners
        }
        for (int index = 0; index < points.Length; index++)
        {
          string character = ((char)(65 + index)).ToString();
          CvInvoke.PutText(con, character, points[index], FontFace.HersheySimplex, 1, new MCvScalar(255, 0, 0), 2);
        }
      }

      if (DiagMode) { result.AddDiagImage("Corners", con.ToGrooperImage()); }
      return con;
    }

    /// <inheritdoc/>
    public static Point[] OrderPoints(Point[] pts)
    {
      if (pts.Length != 4) { throw new ArgumentException("The array must contain exactly 4 points"); }
      Point[] orderedPts = new Point[4];
      IEnumerable<int> sum = pts.Select(p => p.X + p.Y);
      orderedPts[0] = pts[sum.ToList().IndexOf(sum.Min())]; // Top-left
      orderedPts[2] = pts[sum.ToList().IndexOf(sum.Max())]; // Bottom-right

      IEnumerable<int> diff = pts.Select(p => p.Y - p.X);
      orderedPts[1] = pts[diff.ToList().IndexOf(diff.Min())]; // Top-right
      orderedPts[3] = pts[diff.ToList().IndexOf(diff.Max())]; // Bottom-left

      return orderedPts;
    }

    /// <inheritdoc/>
    public Mat ConverToGray(Mat inputImage, IpCommandResult result)
    {
      Mat gray = new Mat();
      CvInvoke.CvtColor(inputImage, gray, ColorConversion.Bgr2Gray);
      CvInvoke.GaussianBlur(gray, gray, new Size(11, 11), 0);

      if (DiagMode) { result.AddDiagImage("Gray", gray.ToGrooperImage()); }

      return gray;
    }
  }
}
