import jpeg from 'jpeg-js';
import { Buffer } from 'buffer';
import { ModelConfig, PreprocessResult } from '../types';

export class ImageProcessor {
  /**
   * Preprocess image from base64 JPEG for PaddleDetection YOLOv3 model.
   *
   * The infer_cfg.yml specifies:
   *   Resize: target_size=[608,608], keep_ratio=false
   *   NormalizeImage: is_scale=true, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
   *   Permute: HWC -> CHW
   */
  static preprocessFromBase64(
    base64: string,
    config: ModelConfig
  ): PreprocessResult {
    const cleanBase64 = base64.includes(',') ? base64.split(',')[1] : base64;
    let raw;
    try {
      raw = jpeg.decode(Buffer.from(cleanBase64, 'base64'), {
        useTArray: true,
        formatAsRGBA: true,
      });
    } catch (error) {
      throw new Error(
        '当前仅支持JPEG图片，请将相机格式设置为"最兼容"(JPEG)或选择JPG图片。'
      );
    }
    if (!raw || !raw.data) {
      throw new Error('图片解析失败，请更换图片重试。');
    }
    const originalWidth = raw.width;
    const originalHeight = raw.height;
    const { inputSize, mean, std } = config;

    const scaleY = inputSize / originalHeight;
    const scaleX = inputSize / originalWidth;

    const imageData = ImageProcessor.resizeNormalizeCHW(
      raw.data,
      originalWidth,
      originalHeight,
      inputSize,
      mean,
      std
    );

    return {
      imageData,
      originalWidth,
      originalHeight,
      scaleX,
      scaleY,
    };
  }

  /**
   * Optimized: Resize RGBA image to inputSize x inputSize, normalize, and output as CHW Float32Array.
   * Uses pre-computed lookup tables to minimize per-pixel computation.
   */
  private static resizeNormalizeCHW(
    rgba: Uint8Array,
    srcW: number,
    srcH: number,
    size: number,
    mean: number[],
    std: number[]
  ): Float32Array {
    const planeSize = size * size;
    const chw = new Float32Array(3 * planeSize);

    // Pre-compute normalization constants: pixel / 255 - mean / std = pixel * invScale - meanNorm
    const invScaleR = 1.0 / (255.0 * std[0]);
    const invScaleG = 1.0 / (255.0 * std[1]);
    const invScaleB = 1.0 / (255.0 * std[2]);
    const meanNormR = mean[0] / std[0];
    const meanNormG = mean[1] / std[1];
    const meanNormB = mean[2] / std[2];

    // Pre-compute source Y indices for each destination Y
    const srcYLookup = new Int32Array(size);
    for (let y = 0; y < size; y++) {
      srcYLookup[y] = Math.min(srcH - 1, (y * srcH) >> 0 | 0);
    }
    // Fix: use floating point division for accuracy
    for (let y = 0; y < size; y++) {
      srcYLookup[y] = Math.min(srcH - 1, Math.floor((y / size) * srcH));
    }

    // Pre-compute source X indices for each destination X
    const srcXLookup = new Int32Array(size);
    for (let x = 0; x < size; x++) {
      srcXLookup[x] = Math.min(srcW - 1, Math.floor((x / size) * srcW));
    }

    // Pre-compute row offsets for source image (srcY * srcW * 4)
    const srcRowOffset = new Int32Array(size);
    for (let y = 0; y < size; y++) {
      srcRowOffset[y] = srcYLookup[y] * srcW * 4;
    }

    // Pre-compute pixel offsets for source X (srcX * 4)
    const srcXOffset = new Int32Array(size);
    for (let x = 0; x < size; x++) {
      srcXOffset[x] = srcXLookup[x] * 4;
    }

    const planeG = planeSize;
    const planeB = planeSize * 2;

    for (let y = 0; y < size; y++) {
      const rowOff = srcRowOffset[y];
      const dstRowOff = y * size;

      for (let x = 0; x < size; x++) {
        const srcIdx = rowOff + srcXOffset[x];
        const dstIdx = dstRowOff + x;

        // Read RGB from RGBA source, normalize and write to CHW planes
        chw[dstIdx] = rgba[srcIdx] * invScaleR - meanNormR;
        chw[planeG + dstIdx] = rgba[srcIdx + 1] * invScaleG - meanNormG;
        chw[planeB + dstIdx] = rgba[srcIdx + 2] * invScaleB - meanNormB;
      }
    }

    return chw;
  }
}
