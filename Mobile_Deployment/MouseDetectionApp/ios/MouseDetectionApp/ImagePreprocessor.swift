import Foundation
import UIKit
import Accelerate
import React

/// Native image preprocessor for PaddleDetection YOLOv3 model.
/// Performs JPEG decode → Resize (608×608) → Normalize → CHW permute
/// entirely in native code using Accelerate framework for maximum speed.
@objc(ImagePreprocessor)
class ImagePreprocessor: NSObject {

  // PaddleDetection normalization constants (ImageNet)
  private static let mean: [Float] = [0.485, 0.456, 0.406]
  private static let std: [Float]  = [0.229, 0.224, 0.225]
  private static let inputSize: Int = 608

  /// Main entry point called from React Native.
  /// Takes a JPEG file path, returns preprocessed CHW float array + metadata.
  @objc
  func preprocess(_ filePath: String,
                  resolver resolve: @escaping RCTPromiseResolveBlock,
                  rejecter reject: @escaping RCTPromiseRejectBlock) {

    DispatchQueue.global(qos: .userInitiated).async {
      let startTime = CFAbsoluteTimeGetCurrent()

      // 1. Load image from file
      // Handle both file:// URI and plain file paths
      let fileURL: URL
      if filePath.hasPrefix("file://") {
        guard let url = URL(string: filePath) else {
          reject("E_FILE", "Invalid file URL: \(filePath)", nil)
          return
        }
        fileURL = url
      } else {
        fileURL = URL(fileURLWithPath: filePath)
      }

      guard let imageData = try? Data(contentsOf: fileURL) else {
        reject("E_FILE", "Cannot read file: \(filePath)", nil)
        return
      }

      guard let uiImage = UIImage(data: imageData) else {
        reject("E_DECODE", "Cannot decode image from file", nil)
        return
      }

      let originalWidth = Int(uiImage.size.width * uiImage.scale)
      let originalHeight = Int(uiImage.size.height * uiImage.scale)

      let decodeTime = CFAbsoluteTimeGetCurrent()

      // 2. Resize to 608×608 using Core Graphics (bilinear interpolation)
      let size = Self.inputSize
      guard let resizedPixels = Self.resizeImage(uiImage, toWidth: size, toHeight: size) else {
        reject("E_RESIZE", "Failed to resize image", nil)
        return
      }

      let resizeTime = CFAbsoluteTimeGetCurrent()

      // 3. Normalize and convert to CHW Float32Array
      let chwData = Self.normalizeAndPermute(resizedPixels, size: size)

      let normalizeTime = CFAbsoluteTimeGetCurrent()

      // 4. Convert Float32Array to base64 string for efficient transfer to JS
      let base64 = chwData.withUnsafeBufferPointer { bufferPtr -> String in
        let rawPtr = UnsafeRawBufferPointer(bufferPtr)
        return Data(rawPtr).base64EncodedString()
      }

      let totalTime = CFAbsoluteTimeGetCurrent()

      let scaleX = Float(size) / Float(originalWidth)
      let scaleY = Float(size) / Float(originalHeight)

      let timings = String(format: "decode=%.0fms resize=%.0fms norm=%.0fms total=%.0fms",
                           (decodeTime - startTime) * 1000,
                           (resizeTime - decodeTime) * 1000,
                           (normalizeTime - resizeTime) * 1000,
                           (totalTime - startTime) * 1000)

      let result: [String: Any] = [
        "data": base64,
        "originalWidth": originalWidth,
        "originalHeight": originalHeight,
        "scaleX": scaleX,
        "scaleY": scaleY,
        "timings": timings,
      ]

      resolve(result)
    }
  }

  /// Resize UIImage to target size, return raw RGBA pixel buffer.
  private static func resizeImage(_ image: UIImage, toWidth width: Int, toHeight height: Int) -> [UInt8]? {
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    var pixels = [UInt8](repeating: 0, count: width * height * 4)

    guard let context = CGContext(
      data: &pixels,
      width: width,
      height: height,
      bitsPerComponent: 8,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    ) else {
      return nil
    }

    context.interpolationQuality = .medium
    guard let cgImage = image.cgImage else { return nil }
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    return pixels
  }

  /// Normalize RGBA pixels and convert to CHW Float32 array.
  /// Uses Accelerate framework for vectorized operations.
  private static func normalizeAndPermute(_ rgba: [UInt8], size: Int) -> [Float] {
    let planeSize = size * size

    // Output CHW buffer
    var chw = [Float](repeating: 0, count: 3 * planeSize)

    // Use vDSP_vfltu8 with stride=4 to extract each channel directly from RGBA
    // This converts UInt8→Float and de-interleaves in one pass per channel
    rgba.withUnsafeBufferPointer { srcBuf in
      chw.withUnsafeMutableBufferPointer { dstBuf in
        let src = srcBuf.baseAddress!
        let dst = dstBuf.baseAddress!

        // Extract R (offset 0, stride 4) → plane 0
        vDSP_vfltu8(src + 0, 4, dst, 1, vDSP_Length(planeSize))
        // Extract G (offset 1, stride 4) → plane 1
        vDSP_vfltu8(src + 1, 4, dst + planeSize, 1, vDSP_Length(planeSize))
        // Extract B (offset 2, stride 4) → plane 2
        vDSP_vfltu8(src + 2, 4, dst + 2 * planeSize, 1, vDSP_Length(planeSize))

        // Now normalize each channel: result = pixel * (1/(255*std)) + (-mean/std)
        var sR: Float = 1.0 / (255.0 * std[0])
        var oR: Float = -mean[0] / std[0]
        vDSP_vsmsa(dst, 1, &sR, &oR, dst, 1, vDSP_Length(planeSize))

        var sG: Float = 1.0 / (255.0 * std[1])
        var oG: Float = -mean[1] / std[1]
        let gPtr = dst + planeSize
        vDSP_vsmsa(gPtr, 1, &sG, &oG, gPtr, 1, vDSP_Length(planeSize))

        var sB: Float = 1.0 / (255.0 * std[2])
        var oB: Float = -mean[2] / std[2]
        let bPtr = dst + 2 * planeSize
        vDSP_vsmsa(bPtr, 1, &sB, &oB, bPtr, 1, vDSP_Length(planeSize))
      }
    }

    return chw
  }

  /// Required for React Native module
  @objc
  static func requiresMainQueueSetup() -> Bool {
    return false
  }
}
